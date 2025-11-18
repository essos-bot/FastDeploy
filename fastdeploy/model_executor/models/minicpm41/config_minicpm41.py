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

MiniCPM4.1 Configuration
"""

from typing import Dict, Optional, Union
from dataclasses import dataclass, field

from transformers.configuration_utils import PretrainedConfig


@dataclass
class MiniCPM41SparseConfig:
    """
    Configuration for MiniCPM4.1 sparse attention (InfLLM-V2)
    """

    # Whether to enable sparse attention
    enabled: bool = True

    # InfLLM-V2 core parameters
    kernel_size: int = 32  # Size of semantic kernels
    kernel_stride: int = 16  # Stride between adjacent kernels
    topk: int = 64  # Number of top-k blocks to select
    block_size: int = 64  # Size of KV cache blocks
    window_size: int = 2048  # Local sliding window size
    dense_len: int = 8192  # Sequence length threshold for sparse attention
    init_blocks: int = 1  # Number of initial blocks
    use_nope: bool = False  # Whether to use NOPE technique

    # Layer configuration
    start_layer: int = 0  # Layer index to start using sparse attention
    end_layer: Optional[int] = None  # Layer index to end using sparse attention (None means all layers)

    # Performance tuning
    use_cuda_graph: bool = True  # Whether to use CUDA graph optimization
    chunk_size: Optional[int] = None  # Chunk size for processing (None means auto)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "enabled": self.enabled,
            "kernel_size": self.kernel_size,
            "kernel_stride": self.kernel_stride,
            "topk": self.topk,
            "block_size": self.block_size,
            "window_size": self.window_size,
            "dense_len": self.dense_len,
            "init_blocks": self.init_blocks,
            "use_nope": self.use_nope,
            "start_layer": self.start_layer,
            "end_layer": self.end_layer,
            "use_cuda_graph": self.use_cuda_graph,
            "chunk_size": self.chunk_size,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "MiniCPM41SparseConfig":
        """Create from dictionary"""
        return cls(**config_dict)


@dataclass
class MiniCPM41HybridReasoningConfig:
    """
    Configuration for MiniCPM4.1 hybrid reasoning mode
    """

    enabled: bool = False  # Whether to enable hybrid reasoning mode
    reasoning_token: str = "thinking"  # Token to indicate reasoning mode
    max_thinking_length: int = 512  # Maximum length of thinking tokens
    thinking_probability: float = 0.8  # Probability of entering thinking mode
    force_thinking: bool = False  # Force thinking mode for all queries

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "enabled": self.enabled,
            "reasoning_token": self.reasoning_token,
            "max_thinking_length": self.max_thinking_length,
            "thinking_probability": self.thinking_probability,
            "force_thinking": self.force_thinking,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "MiniCPM41HybridReasoningConfig":
        """Create from dictionary"""
        return cls(**config_dict)


class MiniCPM41Config(PretrainedConfig):
    """
    MiniCPM4.1-8B Configuration

    This configuration class includes all parameters needed for the MiniCPM4.1 model,
    including support for InfLLM-V2 sparse attention and hybrid reasoning mode.
    """

    model_type = "minicpm41"

    def __init__(
        self,
        # Basic model parameters
        vocab_size: int = 122753,
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        num_hidden_layers: int = 30,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 32,
        max_position_embeddings: int = 65536,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        use_cache: bool = True,
        pad_token_id: int = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        # RoPE scaling for long context
        rope_scaling: Optional[Dict] = None,
        # Sparse attention configuration
        sparse_config: Optional[Union[Dict, MiniCPM41SparseConfig]] = None,
        # Hybrid reasoning configuration
        hybrid_reasoning_config: Optional[Union[Dict, MiniCPM41HybridReasoningConfig]] = None,
        # Model specific configurations
        tie_word_embeddings: bool = False,
        use_flash_attention: bool = True,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            rope_scaling=rope_scaling,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        # Initialize sparse attention configuration
        if sparse_config is None:
            self.sparse_config = MiniCPM41SparseConfig()
        elif isinstance(sparse_config, dict):
            self.sparse_config = MiniCPM41SparseConfig.from_dict(sparse_config)
        else:
            self.sparse_config = sparse_config

        # Initialize hybrid reasoning configuration
        if hybrid_reasoning_config is None:
            self.hybrid_reasoning_config = MiniCPM41HybridReasoningConfig()
        elif isinstance(hybrid_reasoning_config, dict):
            self.hybrid_reasoning_config = MiniCPM41HybridReasoningConfig.from_dict(hybrid_reasoning_config)
        else:
            self.hybrid_reasoning_config = hybrid_reasoning_config

        # Additional model configurations
        self.use_flash_attention = use_flash_attention

    def get_sparse_config(self) -> MiniCPM41SparseConfig:
        """Get sparse attention configuration"""
        return self.sparse_config

    def get_hybrid_reasoning_config(self) -> MiniCPM41HybridReasoningConfig:
        """Get hybrid reasoning configuration"""
        return self.hybrid_reasoning_config

    def set_sparse_config(self, sparse_config: Union[Dict, MiniCPM41SparseConfig]):
        """Set sparse attention configuration"""
        if isinstance(sparse_config, dict):
            self.sparse_config = MiniCPM41SparseConfig.from_dict(sparse_config)
        else:
            self.sparse_config = sparse_config

    def set_hybrid_reasoning_config(self, hybrid_reasoning_config: Union[Dict, MiniCPM41HybridReasoningConfig]):
        """Set hybrid reasoning configuration"""
        if isinstance(hybrid_reasoning_config, dict):
            self.hybrid_reasoning_config = MiniCPM41HybridReasoningConfig.from_dict(hybrid_reasoning_config)
        else:
            self.hybrid_reasoning_config = hybrid_reasoning_config

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        config_dict = super().to_dict()

        # Add sparse config
        if self.sparse_config:
            config_dict["sparse_config"] = self.sparse_config.to_dict()

        # Add hybrid reasoning config
        if self.hybrid_reasoning_config:
            config_dict["hybrid_reasoning_config"] = self.hybrid_reasoning_config.to_dict()

        return config_dict

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "MiniCPM41Config":
        """Create configuration from dictionary"""
        # Extract sparse and hybrid reasoning configs if present
        sparse_config = config_dict.pop("sparse_config", None)
        hybrid_reasoning_config = config_dict.pop("hybrid_reasoning_config", None)

        # Create base config
        config = cls(
            sparse_config=sparse_config,
            hybrid_reasoning_config=hybrid_reasoning_config,
            **config_dict,
        )

        return config

    def validate_configuration(self):
        """Validate configuration parameters"""
        # Validate basic model parameters
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        # Validate sparse configuration
        if self.sparse_config.enabled:
            if self.sparse_config.kernel_size <= 0:
                raise ValueError("sparse_config.kernel_size must be positive")
            if self.sparse_config.kernel_stride <= 0:
                raise ValueError("sparse_config.kernel_stride must be positive")
            if self.sparse_config.topk <= 0:
                raise ValueError("sparse_config.topk must be positive")
            if self.sparse_config.block_size <= 0:
                raise ValueError("sparse_config.block_size must be positive")
            if self.sparse_config.kernel_stride > self.sparse_config.kernel_size:
                raise ValueError("sparse_config.kernel_stride should not exceed kernel_size")
            if self.sparse_config.dense_len < 0:
                raise ValueError("sparse_config.dense_len cannot be negative")

        # Validate hybrid reasoning configuration
        if self.hybrid_reasoning_config.enabled:
            if self.hybrid_reasoning_config.max_thinking_length <= 0:
                raise ValueError("hybrid_reasoning_config.max_thinking_length must be positive")
            if not 0 <= self.hybrid_reasoning_config.thinking_probability <= 1:
                raise ValueError("hybrid_reasoning_config.thinking_probability must be between 0 and 1")

    def get_model_size_info(self) -> Dict:
        """Get model size information"""
        # Calculate parameters count (approximate)
        embedding_params = self.vocab_size * self.hidden_size

        # Attention parameters
        qkv_params = self.hidden_size * self.hidden_size * 3  # Q, K, V
        o_proj_params = self.hidden_size * self.hidden_size  # Output projection
        attention_params = qkv_params + o_proj_params

        # MLP parameters
        gate_proj_params = self.hidden_size * self.intermediate_size
        up_proj_params = self.hidden_size * self.intermediate_size
        down_proj_params = self.intermediate_size * self.hidden_size
        mlp_params = gate_proj_params + up_proj_params + down_proj_params

        # Layer norm parameters
        layernorm_params = self.hidden_size * 2  # Input + post-attention

        # Final norm parameters
        final_norm_params = self.hidden_size

        # Parameters per layer
        params_per_layer = attention_params + mlp_params + layernorm_params

        # Total parameters
        total_params = embedding_params + params_per_layer * self.num_hidden_layers + final_norm_params

        return {
            "total_parameters": total_params,
            "parameters_per_layer": params_per_layer,
            "attention_parameters_per_layer": attention_params,
            "mlp_parameters_per_layer": mlp_params,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "head_dim": self.hidden_size // self.num_attention_heads,
        }

    def supports_sparse_attention(self) -> bool:
        """Check if model supports sparse attention"""
        return self.sparse_config.enabled

    def supports_hybrid_reasoning(self) -> bool:
        """Check if model supports hybrid reasoning"""
        return self.hybrid_reasoning_config.enabled

    def get_attention_backend_name(self) -> str:
        """Get recommended attention backend name"""
        if self.supports_sparse_attention():
            return "INFLLMV2_ATTN"
        elif self.use_flash_attention:
            return "FLASH_ATTN"
        else:
            return "APPEND_ATTN"
