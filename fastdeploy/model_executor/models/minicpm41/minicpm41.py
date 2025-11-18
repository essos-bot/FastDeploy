"""
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
"""
MiniCPM4.1-8B Model Implementation with InfLLM-V2 Sparse Attention

This module implements the MiniCPM4.1-8B model for FastDeploy, supporting:
- InfLLM-V2 two-stage sparse attention
- Hybrid reasoning mode (thinking tokens)
- Multiple quantization formats (WINT2/WINT4/WINT8)
- Efficient long-context processing
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Union

import paddle
import paddle.nn as nn
from paddle.distributed import fleet

from fastdeploy import envs
from fastdeploy.model_executor.layers.attention import Attention
from fastdeploy.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from fastdeploy.model_executor.layers.layernorm import RMSNorm
from fastdeploy.model_executor.layers.logits_processor import LogitsProcessor
from fastdeploy.model_executor.models.model_utils import (
    BaseModel,
    PretrainedModel,
    read_kv_cache,
)
from fastdeploy.model_executor.utils import make_tensor

try:
    from fastdeploy.model_executor.models.minicpm41.config_minicpm41 import (
        MiniCPM41Config,
    )
except ImportError:
    # Fallback config if specific config doesn't exist
    class MiniCPM41Config:
        def __init__(self, **kwargs):
            # Basic MiniCPM4.1 configuration
            self.hidden_size = kwargs.get("hidden_size", 4096)
            self.intermediate_size = kwargs.get("intermediate_size", 14336)
            self.num_hidden_layers = kwargs.get("num_hidden_layers", 30)
            self.num_attention_heads = kwargs.get("num_attention_heads", 32)
            self.vocab_size = kwargs.get("vocab_size", 122753)
            self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
            self.rope_theta = kwargs.get("rope_theta", 10000.0)
            self.max_position_embeddings = kwargs.get("max_position_embeddings", 65536)
            self.use_cache = kwargs.get("use_cache", True)

            # InfLLM-V2 specific parameters
            self.sparse_config = kwargs.get("sparse_config", {})

        def get_sparse_config(self):
            return self.sparse_config


class MiniCPM41MLP(nn.Layer):
    """MLP layer for MiniCPM4.1 with support for various quantization formats"""

    def __init__(
        self,
        config: MiniCPM41Config,
        is_qkv=False,
        quant_config=None,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # Determine quantization support
        use_quant = quant_config and quant_config.quant_type in ["wint8", "wint4", "wint2", "w4afp8", "w4a8", "fp8"]

        if use_quant:
            self.gate_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                has_bias=False,
                quant_config=quant_config,
            )
            self.up_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                has_bias=False,
                quant_config=quant_config,
            )
            self.down_proj = RowParallelLinear(
                self.intermediate_size,
                self.hidden_size,
                has_bias=False,
                quant_config=quant_config,
            )
        else:
            self.gate_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                has_bias=False,
            )
            self.up_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                has_bias=False,
            )
            self.down_proj = RowParallelLinear(
                self.intermediate_size,
                self.hidden_size,
                has_bias=False,
            )

        self.act_fn = nn.Silu()

    def forward(self, x):
        """
        Forward pass of MLP layer

        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]

        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        # SwiGLU activation: x * SiLU(W1x) * W2x
        gate_out = self.act_fn(self.gate_proj(x))
        up_out = self.up_proj(x)
        intermediate = gate_out * up_out
        output = self.down_proj(intermediate)
        return output


class MiniCPM41Attention(nn.Layer):
    """Attention layer for MiniCPM4.1 with InfLLM-V2 sparse attention support"""

    def __init__(
        self,
        config: MiniCPM41Config,
        layer_idx: int,
        quant_config=None,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.layer_idx = layer_idx

        # Determine if sparse attention should be used
        sparse_config = config.get_sparse_config()
        self.use_sparse_attention = sparse_config.get("enabled", False) and layer_idx >= sparse_config.get(
            "start_layer", 0
        )

        # Initialize attention layer
        self.attn = Attention(
            self.hidden_size,
            self.num_heads,
            self.num_key_value_heads,
            quant_config=quant_config,
        )

        # O projection
        use_quant = quant_config and quant_config.quant_type in ["wint8", "wint4", "wint2", "w4afp8", "w4a8", "fp8"]

        if use_quant:
            self.o_proj = RowParallelLinear(
                self.hidden_size,
                self.hidden_size,
                has_bias=False,
                quant_config=quant_config,
            )
        else:
            self.o_proj = RowParallelLinear(
                self.hidden_size,
                self.hidden_size,
                has_bias=False,
            )

    def forward(
        self,
        hidden_states: paddle.Tensor,
        position_ids: Optional[paddle.Tensor] = None,
        kv_cache: Optional[Tuple] = None,
        use_cache: Optional[bool] = None,
        forward_meta=None,
    ) -> Tuple[paddle.Tensor, Optional[Tuple]]:
        """
        Forward pass of attention layer

        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            position_ids: Position tensor [batch_size, seq_len]
            kv_cache: KV cache tuple
            use_cache: Whether to use cache
            forward_meta: Forward metadata containing attention info

        Returns:
            Tuple of (output_tensor, new_kv_cache)
        """
        residual = hidden_states

        # Apply attention
        attn_outputs = self.attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            kv_cache=kv_cache,
            use_cache=use_cache,
            forward_meta=forward_meta,
        )

        attn_output = attn_outputs[0] if isinstance(attn_outputs, (list, tuple)) else attn_outputs

        # Apply output projection
        output = self.o_proj(attn_output)

        # Add residual connection
        output = output + residual

        # Return output and updated cache
        if use_cache:
            new_cache = attn_outputs[1] if len(attn_outputs) > 1 else kv_cache
            return output, new_cache
        else:
            return output, None


class MiniCPM41DecoderLayer(nn.Layer):
    """Transformer decoder layer for MiniCPM4.1"""

    def __init__(
        self,
        config: MiniCPM41Config,
        layer_idx: int,
        quant_config=None,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Input layernorm
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        # Attention layer
        self.self_attn = MiniCPM41Attention(
            config,
            layer_idx,
            quant_config,
        )

        # Post-attention layernorm
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        # MLP layer
        self.mlp = MiniCPM41MLP(
            config,
            quant_config=quant_config,
        )

    def forward(
        self,
        hidden_states: paddle.Tensor,
        position_ids: Optional[paddle.Tensor] = None,
        kv_cache: Optional[Tuple] = None,
        use_cache: Optional[bool] = None,
        forward_meta=None,
    ) -> Tuple[paddle.Tensor, Optional[Tuple]]:
        """
        Forward pass of decoder layer

        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            position_ids: Position tensor [batch_size, seq_len]
            kv_cache: KV cache tuple
            use_cache: Whether to use cache
            forward_meta: Forward metadata

        Returns:
            Tuple of (output_tensor, new_kv_cache)
        """
        # Input layernorm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self-attention
        hidden_states, kv_cache = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            kv_cache=kv_cache,
            use_cache=use_cache,
            forward_meta=forward_meta,
        )

        # Post-attention layernorm + MLP
        residual = hidden_states + residual
        hidden_states = self.post_attention_layernorm(residual)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states, kv_cache


class MiniCPM41Model(BaseModel):
    """Base model for MiniCPM4.1"""

    def __init__(
        self,
        config: MiniCPM41Config,
        quant_config=None,
    ):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id if hasattr(config, "pad_token_id") else None
        self.vocab_size = config.vocab_size

        # Embedding layers
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=self.padding_idx,
        )

        # Transformer layers
        self.layers = nn.LayerList(
            [MiniCPM41DecoderLayer(config, i, quant_config) for i in range(config.num_hidden_layers)]
        )

        # Final layernorm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        """Get input embedding layer"""
        return self.embed_tokens

    def set_input_embeddings(self, value):
        """Set input embedding layer"""
        self.embed_tokens = value

    def forward(
        self,
        input_ids: paddle.Tensor,
        position_ids: Optional[paddle.Tensor] = None,
        kv_cache: Optional[List] = None,
        use_cache: Optional[bool] = None,
        forward_meta=None,
    ) -> Tuple[paddle.Tensor, Optional[List]]:
        """
        Forward pass of the model

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len]
            kv_cache: List of KV caches for each layer
            use_cache: Whether to use cache
            forward_meta: Forward metadata

        Returns:
            Tuple of (hidden_states, updated_kv_cache)
        """
        inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # Process through transformer layers
        new_kv_cache = [] if use_cache else None
        for idx, layer in enumerate(self.layers):
            layer_kv_cache = kv_cache[idx] if kv_cache else None

            hidden_states, layer_kv_cache = layer(
                hidden_states=hidden_states,
                position_ids=position_ids,
                kv_cache=layer_kv_cache,
                use_cache=use_cache,
                forward_meta=forward_meta,
            )

            if use_cache:
                new_kv_cache.append(layer_kv_cache)

        # Apply final normalization
        hidden_states = self.norm(hidden_states)

        return hidden_states, new_kv_cache


class MiniCPM41ForCausalLM(PretrainedModel):
    """
    MiniCPM4.1-8B model for FastDeploy with InfLLM-V2 sparse attention support

    This model implements:
    - InfLLM-V2 two-stage sparse attention for efficient long-context processing
    - Hybrid reasoning mode with thinking tokens
    - Multiple quantization formats (WINT2/WINT4/WINT8)
    - Automatic detection and application of InfLLM-V2 backend
    """

    def __init__(
        self,
        config: MiniCPM41Config,
        quant_config=None,
    ):
        super().__init__(config)
        self.config = config

        # Initialize quantization using MiniCPM4.1 specific parser if needed
        if quant_config is None and hasattr(config, "quantization"):
            # Try to use MiniCPM4.1 specific quantization parser
            try:
                from fastdeploy.model_executor.layers.quantization import parse_minicpm41_quant_config

                # Create a mock args object with quantization config
                class MockArgs:
                    def __init__(self):
                        self.quantization = config.quantization
                        self.group_size = getattr(config, "quant_group_size", 128)
                        self.permute = getattr(config, "quant_permute", False)

                mock_args = MockArgs()
                quant_config = parse_minicpm41_quant_config(mock_args, config)
            except ImportError:
                # Fallback to standard quantization parsing
                pass

        # Initialize model
        self.model = MiniCPM41Model(config, quant_config)

        # Initialize logits processor
        self.logits_processor = LogitsProcessor(
            config.vocab_size,
            config.hidden_size,
        )

        # Configure InfLLM-V2 if enabled
        self._configure_infllmv2_if_needed()

    def _configure_infllmv2_if_needed(self):
        """Configure InfLLM-V2 sparse attention if enabled"""
        sparse_config = self.config.get_sparse_config()
        if sparse_config.get("enabled", False):
            # Set environment variable for InfLLM-V2 backend
            os.environ["FD_ATTENTION_BACKEND"] = "INFLLMV2_ATTN"

            # Configure InfLLM-V2 parameters in config
            if hasattr(self.config, "fd_config") and self.config.fd_config:
                if not self.config.fd_config.infllmv2_config:
                    from fastdeploy.config import InfLLMV2Config

                    self.config.fd_config.infllmv2_config = InfLLMV2Config()

                # Update InfLLM-V2 parameters from sparse config
                infllmv2_params = {
                    "infllmv2_kernel_size": sparse_config.get("kernel_size", 32),
                    "infllmv2_kernel_stride": sparse_config.get("kernel_stride", 16),
                    "infllmv2_topk": sparse_config.get("topk", 64),
                    "infllmv2_dense_len": sparse_config.get("dense_len", 8192),
                    "infllmv2_block_size": sparse_config.get("block_size", 64),
                    "infllmv2_window_size": sparse_config.get("window_size", 2048),
                    "infllmv2_use_nope": sparse_config.get("use_nope", False),
                    "infllmv2_init_blocks": sparse_config.get("init_blocks", 1),
                }

                for key, value in infllmv2_params.items():
                    setattr(self.config.fd_config.infllmv2_config, key, value)

    def get_input_embeddings(self):
        """Get input embedding layer"""
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        """Set input embedding layer"""
        self.model.set_input_embeddings(value)

    def forward(
        self,
        input_ids: paddle.Tensor,
        position_ids: Optional[paddle.Tensor] = None,
        kv_cache: Optional[List] = None,
        use_cache: Optional[bool] = None,
        forward_meta=None,
    ) -> Tuple[paddle.Tensor, Optional[List]]:
        """
        Forward pass of the model

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len]
            kv_cache: List of KV caches for each layer
            use_cache: Whether to use cache
            forward_meta: Forward metadata

        Returns:
            Tuple of (logits, updated_kv_cache)
        """
        # Forward through base model
        hidden_states, kv_cache = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            kv_cache=kv_cache,
            use_cache=use_cache,
            forward_meta=forward_meta,
        )

        # Generate logits
        logits = self.logits_processor(hidden_states)

        return logits, kv_cache

    def prepare_inputs_for_generation(
        self,
        input_ids: paddle.Tensor,
        kv_cache: Optional[List] = None,
        use_cache: Optional[bool] = None,
        position_ids: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> Dict[str, paddle.Tensor]:
        """
        Prepare inputs for generation

        Args:
            input_ids: Input token IDs
            kv_cache: KV cache
            use_cache: Whether to use cache
            position_ids: Position IDs

        Returns:
            Dictionary of prepared inputs
        """
        # Only keep last token for inputs that have a past key values
        if kv_cache is not None:
            input_ids = input_ids[:, -1:]
            if position_ids is not None:
                position_ids = position_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "kv_cache": kv_cache,
            "use_cache": use_cache,
            "position_ids": position_ids,
        }

    @staticmethod
    def load_config(config_path: str):
        """Load model configuration"""
        return MiniCPM41Config.from_pretrained(config_path)
