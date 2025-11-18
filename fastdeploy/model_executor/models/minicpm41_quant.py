"""
MiniCPM4.1-8B Quantization Support

This module provides quantization support for MiniCPM4.1-8B model in FastDeploy,
including various quantization methods and optimized implementations.
"""

import math
from typing import Dict, Optional, Tuple, Union

import paddle
import paddle.nn.functional as F
from paddle import nn

from fastdeploy.config import FDConfig
from fastdeploy.config.minicpm41_config import MiniCPM41ModelConfig
from fastdeploy.model_executor.layers.quantization.quant_base import (
    QuantConfigBase,
    QuantMethodBase,
)
from fastdeploy.model_executor.layers.quantization.weight_only import (
    WINT4Config,
    WINT8Config,
    WINT2Config,
)
from fastdeploy.model_executor.layers.quantization.fp8_quant import (
    W4AFP8Config,
    W4A8Config,
    WFP8AFP8Config,
)
from fastdeploy.model_executor.models.minicpm41 import (
    MiniCPM41ForCausalLM as MiniCPM41Base,
    MiniCPM41DecoderLayer as MiniCPM41DecoderLayerBase,
    MiniCPM41Attention as MiniCPM41AttentionBase,
    MiniCPM41MLP as MiniCPM41MLPBase,
)


class MiniCPM41QuantizationConfig(QuantConfigBase):
    """
    MiniCPM4.1-8B specific quantization configuration
    """

    def __init__(
        self,
        quant_type: str = "w4afp8",
        weight_bits: int = 4,
        activation_bits: int = 8,
        group_size: int = 128,
        is_permuted: bool = True,
        hadamard_block_size: int = 128,
        enable_kv_cache_quant: bool = False,
        kv_cache_quant_type: str = "fp8",
        **kwargs
    ):
        super().__init__()
        self.quant_type = quant_type
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.group_size = group_size
        self.is_permuted = is_permuted
        self.hadamard_block_size = hadamard_block_size
        self.enable_kv_cache_quant = enable_kv_cache_quant
        self.kv_cache_quant_type = kv_cache_quant_type

        # Validate configuration
        self._validate_config()

    def _validate_config(self):
        """Validate quantization configuration"""
        # 扩展支持的量化类型，包括新的INT量化类型
        supported_quant_types = [
            "w4afp8", "w4a8", "wint8", "wint4", "wint2", "fp8",
            "w8a16", "w4a16"  # 添加新的量化类型支持
        ]
        if self.quant_type not in supported_quant_types:
            raise ValueError(f"Unsupported quantization type: {self.quant_type}")

        # 验证与InfLLM-V2稀疏注意力的兼容性
        self._validate_infllmv2_compatibility()

        if self.weight_bits not in [2, 4, 8]:
            raise ValueError(f"Unsupported weight bits: {self.weight_bits}")

        if self.activation_bits not in [8, 16]:
            raise ValueError(f"Unsupported activation bits: {self.activation_bits}")

        if self.group_size <= 0 or (self.group_size & (self.group_size - 1)) != 0:
            raise ValueError("Group size must be positive and power of 2")

    def _validate_infllmv2_compatibility(self):
        """验证与InfLLM-V2稀疏注意力的兼容性"""
        # 某些量化类型可能与稀疏注意力有兼容性问题
        compatibility_issues = {
            "wint2": "WINT2量化可能限制InfLLM-V2稀疏注意力的性能",
            "wint4": "WINT4量化与InfLLM-V2基本兼容，可能有轻微性能影响",
        }

        if self.quant_type in compatibility_issues:
            import warnings
            warnings.warn(f"{compatibility_issues[self.quant_type]}")

    def name(self) -> str:
        """Return quantization method name"""
        return f"minicpm41_{self.quant_type}"

    @classmethod
    def from_config(cls, config: dict) -> "MiniCPM41QuantizationConfig":
        """Create configuration from dictionary"""
        return cls(**config)

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "quant_type": self.quant_type,
            "weight_bits": self.weight_bits,
            "activation_bits": self.activation_bits,
            "group_size": self.group_size,
            "is_permuted": self.is_permuted,
            "hadamard_block_size": self.hadamard_block_size,
            "enable_kv_cache_quant": self.enable_kv_cache_quant,
            "kv_cache_quant_type": self.kv_cache_quant_type,
        }


class QuantizedMiniCPM41MLP(MiniCPM41MLPBase):
    """
    Quantized MiniCPM4.1-8B MLP with quantization support
    """

    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = ""):
        # Override parent initialization to use quantized layers
        self.fd_config = fd_config
        self.model_config = fd_config.model_config
        self.layer_id = layer_id
        self.prefix = prefix
        self.quant_config = getattr(fd_config, 'quant_config', None)

        # Initialize quantized layers
        self._init_quantized_layers()

    def _init_quantized_layers(self):
        """Initialize quantized MLP layers"""
        from fastdeploy.model_executor.layers.linear import (
            QuantizedRowParallelLinear,
            RowParallelLinear,
        )

        if self.quant_config and self.quant_config.quant_type in ["w4afp8", "w4a8", "wint8", "wint4"]:
            # Use quantized linear layers
            linear_class = QuantizedRowParallelLinear
        else:
            # Use standard linear layers
            linear_class = RowParallelLinear

        # Gate projection
        self.gate_proj = linear_class(
            self.fd_config,
            prefix=f"{self.prefix}.gate_proj",
            input_size=self.model_config.hidden_size,
            output_size=self.model_config.intermediate_size,
            layer_id=self.layer_id,
        )

        # Up projection
        self.up_proj = linear_class(
            self.fd_config,
            prefix=f"{self.prefix}.up_proj",
            input_size=self.model_config.hidden_size,
            output_size=self.model_config.intermediate_size,
            layer_id=self.layer_id,
        )

        # Down projection
        self.down_proj = linear_class(
            self.fd_config,
            prefix=f"{self.prefix}.down_proj",
            input_size=self.model_config.intermediate_size,
            output_size=self.model_config.hidden_size,
            layer_id=self.layer_id,
        )


class QuantizedMiniCPM41Attention(MiniCPM41AttentionBase):
    """
    Quantized MiniCPM4.1-8B Attention with quantization support
    """

    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = ""):
        # Override parent initialization to use quantized layers
        self.fd_config = fd_config
        self.model_config = fd_config.model_config
        self.layer_id = layer_id
        self.prefix = prefix
        self.quant_config = getattr(fd_config, 'quant_config', None)

        # Compute head dimension
        self.head_dim = self.model_config.hidden_size // self.model_config.num_attention_heads

        # Initialize quantized layers
        self._init_quantized_layers()

        # Optional Q/K normalization
        self._init_qk_norm()

    def _init_quantized_layers(self):
        """Initialize quantized attention layers"""
        from fastdeploy.model_executor.layers.linear import (
            QuantizedQKVParallelLinear,
            QKVParallelLinear,
            QuantizedRowParallelLinear,
            RowParallelLinear,
        )

        if self.quant_config and self.quant_config.quant_type in ["w4afp8", "w4a8", "wint8", "wint4"]:
            # Use quantized linear layers
            qkv_linear_class = QuantizedQKVParallelLinear
            output_linear_class = QuantizedRowParallelLinear
        else:
            # Use standard linear layers
            qkv_linear_class = QKVParallelLinear
            output_linear_class = RowParallelLinear

        # QKV parallel linear projection
        self.qkv_proj = qkv_linear_class(
            self.fd_config,
            prefix=f"{self.prefix}.qkv_proj",
            with_bias=False,
        )

        # Output projection
        self.o_proj = output_linear_class(
            self.fd_config,
            prefix=f"{self.prefix}.o_proj",
            input_size=self.model_config.hidden_size,
            output_size=self.model_config.hidden_size,
            layer_id=self.layer_id,
        )

        # Attention mechanism
        self.attn = self._create_attention_layer()

    def _create_attention_layer(self):
        """Create attention layer with optional KV cache quantization"""
        from fastdeploy.model_executor.layers.attention.attention import Attention

        # Check if KV cache quantization is enabled
        kv_cache_config = None
        if (self.quant_config and
            self.quant_config.enable_kv_cache_quant and
            self.quant_config.kv_cache_quant_type == "fp8"):
            from fastdeploy.model_executor.layers.quantization.kv_cache_quant import (
                KVCacheQuantConfig,
            )
            kv_cache_config = KVCacheQuantConfig(
                quant_type="fp8",
                enable_quant=True,
            )

        return Attention(
            self.fd_config,
            layer_id=self.layer_id,
            prefix=self.prefix,
            use_neox_rotary_style=True,
            kv_cache_config=kv_cache_config,
        )

    def _init_qk_norm(self):
        """Initialize Q/K normalization layers"""
        from fastdeploy.model_executor.layers.normalization import RMSNorm

        # Optional Q/K normalization (if enabled in config)
        self.q_norm = None
        self.k_norm = None
        if getattr(self.model_config, "use_qk_norm", False):
            self.q_norm = RMSNorm(
                self.fd_config,
                hidden_size=self.head_dim,
                eps=self.model_config.rms_norm_eps,
                prefix=f"{self.prefix}.q_norm",
                begin_norm_axis=-1,
            )
            self.k_norm = RMSNorm(
                self.fd_config,
                hidden_size=self.head_dim,
                eps=self.model_config.rms_norm_eps,
                prefix=f"{self.prefix}.k_norm",
                begin_norm_axis=-1,
            )


class QuantizedMiniCPM41DecoderLayer(MiniCPM41DecoderLayerBase):
    """
    Quantized MiniCPM4.1-8B Decoder Layer with quantization support
    """

    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = ""):
        # Override parent initialization to use quantized layers
        self.fd_config = fd_config
        self.model_config = fd_config.model_config
        self.layer_id = layer_id
        self.prefix = prefix

        # Initialize quantized components
        self.self_attn = QuantizedMiniCPM41Attention(
            fd_config,
            layer_id=layer_id,
            prefix=f"{prefix}.self_attn"
        )

        # Input layernorm (pre-attention norm)
        from fastdeploy.model_executor.layers.normalization import RMSNorm
        self.input_layernorm = RMSNorm(
            fd_config,
            hidden_size=self.model_config.hidden_size,
            eps=self.model_config.rms_norm_eps,
            prefix=f"{prefix}.input_layernorm",
            begin_norm_axis=-1,
        )

        # Quantized MLP
        self.mlp = QuantizedMiniCPM41MLP(
            fd_config,
            layer_id=layer_id,
            prefix=f"{prefix}.mlp"
        )

        # Post-attention layernorm
        self.post_attention_layernorm = RMSNorm(
            fd_config,
            hidden_size=self.model_config.hidden_size,
            eps=self.model_config.rms_norm_eps,
            prefix=f"{prefix}.post_attention_layernorm",
            begin_norm_axis=-1,
        )


class MiniCPM41ForCausalLM(MiniCPM41Base):
    """
    MiniCPM4.1-8B model with quantization support for FastDeploy
    """

    def __init__(self, fd_config: FDConfig):
        # Store quantization config
        self.quant_config = getattr(fd_config, 'quant_config', None)

        # Initialize base model
        super().__init__(fd_config)

        # Override layers with quantized versions if quantization is enabled
        if self.quant_config:
            self._init_quantized_layers()

    def _init_quantized_layers(self):
        """Initialize model layers with quantization support"""
        # Replace decoder layers with quantized versions
        self.layers = nn.LayerList([
            QuantizedMiniCPM41DecoderLayer(
                self.fd_config,
                layer_id=i
            )
            for i in range(self.model_config.num_hidden_layers)
        ])

    def set_state_dict(self, state_dict: Dict[str, paddle.Tensor]) -> None:
        """
        Set state dict with quantization-aware weight mapping
        """
        # Process quantization-specific weight mapping
        if self.quant_config:
            state_dict = self._process_quantized_weights(state_dict)

        # Call parent method for standard weight mapping
        super().set_state_dict(state_dict)

    def _process_quantized_weights(self, state_dict: Dict[str, paddle.Tensor]) -> Dict[str, paddle.Tensor]:
        """
        Process quantized weights for proper loading
        """
        quant_type = self.quant_config.quant_type

        # Handle different quantization types
        if quant_type in ["w4afp8", "w4a8"]:
            state_dict = self._process_fp8_weights(state_dict, quant_type)
        elif quant_type in ["wint8", "wint4", "wint2"]:
            state_dict = self._process_weight_only_weights(state_dict, quant_type)

        return state_dict

    def _process_fp8_weights(self, state_dict: Dict, quant_type: str) -> Dict[str, paddle.Tensor]:
        """Process FP8 quantized weights"""
        processed_state_dict = {}

        for key, value in state_dict.items():
            # Map weight names for quantized layers
            if key.endswith(".weight"):
                # Handle quantized weight naming
                if "qkv_proj" in key:
                    new_key = key.replace(".weight", ".weight_quant")
                elif any(x in key for x in ["gate_proj", "up_proj", "down_proj", "o_proj"]):
                    new_key = key.replace(".weight", ".weight_quant")
                else:
                    new_key = key

                processed_state_dict[new_key] = value

                # Add scale tensors if not present
                scale_key = new_key.replace(".weight_quant", ".weight_scale")
                if scale_key not in state_dict and scale_key not in processed_state_dict:
                    # Create default scale tensor
                    if value.dtype == paddle.float32:
                        scale_value = paddle.ones([1], dtype=paddle.float32)
                    else:
                        scale_value = paddle.ones([1], dtype=paddle.float32)
                    processed_state_dict[scale_key] = scale_value
            else:
                processed_state_dict[key] = value

        return processed_state_dict

    def _process_weight_only_weights(self, state_dict: Dict, quant_type: str) -> Dict[str, paddle.Tensor]:
        """Process weight-only quantized weights"""
        processed_state_dict = {}

        for key, value in state_dict.items():
            # Map weight names for weight-only quantization
            if key.endswith(".weight"):
                if quant_type == "wint8":
                    # Convert to int8
                    if value.dtype != paddle.int8:
                        # Quantize to int8 (simple scaling)
                        scale = paddle.max(paddle.abs(value)).item()
                        if scale == 0:
                            scale = 1.0
                        quantized_value = paddle.clip(
                            paddle.round(value / scale * 127), -128, 127
                        ).astype(paddle.int8)

                        processed_state_dict[key] = quantized_value
                        processed_state_dict[key.replace(".weight", ".weight_scale")] = paddle.to_tensor([scale])
                    else:
                        processed_state_dict[key] = value
                elif quant_type in ["wint4", "wint2"]:
                    # Convert to int4/int2 (stored in int8)
                    bits = 4 if quant_type == "wint4" else 2
                    processed_state_dict = self._quantize_to_int_n(value, key, bits, processed_state_dict)
                else:
                    processed_state_dict[key] = value
            else:
                processed_state_dict[key] = value

        return processed_state_dict

    def _quantize_to_int_n(self, weight: paddle.Tensor, key: str, bits: int, state_dict: Dict) -> Dict:
        """Quantize weight to int-n (n=2,4) and store in int8"""
        # Calculate quantization range
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1

        # Group-wise quantization
        group_size = getattr(self.quant_config, 'group_size', 128)
        original_shape = weight.shape
        weight_2d = weight.reshape([-1, original_shape[-1]])

        # Reshape for group-wise quantization
        num_groups = (weight_2d.shape[-1] + group_size - 1) // group_size
        weight_grouped = weight_2d.reshape([-1, num_groups, group_size])

        # Calculate scales
        max_vals = paddle.max(paddle.abs(weight_grouped), axis=-1, keepdim=True)
        scales = paddle.where(max_vals > 0, max_vals, paddle.ones_like(max_vals))
        scales = scales / qmax

        # Quantize
        quantized = paddle.clip(
            paddle.round(weight_grouped / scales), qmin, qmax
        ).astype(paddle.int8)

        # Store two int4/int2 values in one int8
        if bits == 4:
            # Pack two int4 values into one int8
            quantized_packed = (quantized[:, :, ::2] << 4) | (quantized[:, :, 1::2] & 0x0F)
        else:  # bits == 2
            # Pack four int2 values into one int8
            quantized_packed = (
                (quantized[:, :, ::4] & 0x03) << 6 |
                (quantized[:, :, 1::4] & 0x03) << 4 |
                (quantized[:, :, 2::4] & 0x03) << 2 |
                (quantized[:, :, 3::4] & 0x03)
            )

        # Reshape back and store
        quantized_weight = quantized_packed.reshape(original_shape[:-1] + [-1])
        state_dict[key] = quantized_weight
        state_dict[key.replace(".weight", ".weight_scale")] = scales.reshape(original_shape[:-1] + [-1])

        return state_dict

    def get_quantization_info(self) -> Dict:
        """Get quantization information"""
        if not self.quant_config:
            return {"quantized": False}

        return {
            "quantized": True,
            "quant_type": self.quant_config.quant_type,
            "weight_bits": self.quant_config.weight_bits,
            "activation_bits": self.quant_config.activation_bits,
            "group_size": self.quant_config.group_size,
            "is_permuted": self.quant_config.is_permuted,
            "hadamard_block_size": self.quant_config.hadamard_block_size,
            "enable_kv_cache_quant": self.quant_config.enable_kv_cache_quant,
            "kv_cache_quant_type": self.quant_config.kv_cache_quant_type,
            "memory_reduction_ratio": self._calculate_memory_reduction(),
        }

    def _calculate_memory_reduction(self) -> float:
        """Calculate memory reduction ratio due to quantization"""
        if not self.quant_config:
            return 1.0

        weight_bits = self.quant_config.weight_bits
        activation_bits = self.quant_config.activation_bits

        # Approximate memory reduction (assuming fp16 baseline)
        weight_reduction = 16 / weight_bits
        activation_reduction = 16 / activation_bits

        # Weighted average (assuming 70% weights, 30% activations)
        total_reduction = 0.7 * weight_reduction + 0.3 * activation_reduction

        return total_reduction


def create_quantized_minicpm41_config(
    quant_type: str = "w4afp8",
    **kwargs
) -> Tuple[FDConfig, MiniCPM41QuantizationConfig]:
    """
    Create quantized MiniCPM4.1-8B configuration

    Args:
        quant_type: Type of quantization (w4afp8, w4a8, wint8, wint4, wint2)
        **kwargs: Additional quantization parameters

    Returns:
        Tuple of (FDConfig, QuantizationConfig)
    """
    # Create model config
    model_config = MiniCPM41ModelConfig()

    # Create quantization config
    quant_config = MiniCPM41QuantizationConfig(quant_type=quant_type, **kwargs)

    # Create FDConfig with quantization
    fd_config = FDConfig(
        model_config=model_config,
        quant_config=quant_config,
    )

    return fd_config, quant_config


def optimize_for_quantization(
    model_config: MiniCPM41ModelConfig,
    quant_config: MiniCPM41QuantizationConfig
) -> MiniCPM41ModelConfig:
    """
    Optimize model configuration for quantization

    Args:
        model_config: Base model configuration
        quant_config: Quantization configuration

    Returns:
        Optimized model configuration
    """
    # Adjust model config based on quantization type
    optimized_config = MiniCPM41ModelConfig(**model_config.__dict__)

    # Optimizations for different quantization types
    if quant_config.quant_type in ["w4afp8", "w4a8"]:
        # Enable memory optimizations for FP8
        optimized_config.use_cache = True
        optimized_config.enable_thinking = False  # Disable thinking mode for memory efficiency

    elif quant_config.quant_type in ["wint8", "wint4"]:
        # Adjust for weight-only quantization
        if quant_config.weight_bits <= 4:
            # Enable additional optimizations for extreme quantization
            optimized_config.max_position_embeddings = min(
                optimized_config.max_position_embeddings, 32768
            )

    return optimized_config