"""
MiniCPM4.1-8B specific configuration for FastDeploy

This module provides configuration classes and utilities for MiniCPM4.1-8B model,
including special configurations for sparse attention and mixed reasoning mode.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from fastdeploy.config import ModelConfig


@dataclass
class SparseAttentionConfig:
    """Configuration for trainable sparse attention (InfLLM v2)"""
    kernel_size: int = 32
    kernel_stride: int = 16
    init_blocks: int = 1
    block_size: int = 64
    window_size: int = 2048
    topk: int = 64
    use_nope: bool = False
    dense_len: int = 8192  # Switch to sparse attention beyond this length

    def __post_init__(self):
        """Validate configuration parameters"""
        if self.kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if self.kernel_stride <= 0:
            raise ValueError("kernel_stride must be positive")
        if self.init_blocks < 0:
            raise ValueError("init_blocks must be non-negative")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.topk <= 0:
            raise ValueError("topk must be positive")

    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            "kernel_size": self.kernel_size,
            "kernel_stride": self.kernel_stride,
            "init_blocks": self.init_blocks,
            "block_size": self.block_size,
            "window_size": self.window_size,
            "topk": self.topk,
            "use_nope": self.use_nope,
            "dense_len": self.dense_len,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "SparseAttentionConfig":
        """Create from dictionary"""
        return cls(**config_dict)


@dataclass
class RopeScalingConfig:
    """Configuration for RoPE scaling (LongRope)"""
    rope_type: str = "longrope"
    long_factor: List[float] = field(default_factory=list)
    short_factor: List[float] = field(default_factory=list)
    original_max_position_embeddings: int = 65536

    def __post_init__(self):
        """Validate and initialize configuration"""
        if self.rope_type not in ["longrope", "linear", "dynamic"]:
            raise ValueError(f"Unsupported rope_type: {self.rope_type}")

        # Set default factors if not provided (typical for 65536 context)
        if not self.long_factor:
            self.long_factor = [1.0] * 20  # Simplified default
        if not self.short_factor:
            self.short_factor = [1.0] * 20  # Simplified default

    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            "rope_type": self.rope_type,
            "long_factor": self.long_factor,
            "short_factor": self.short_factor,
            "original_max_position_embeddings": self.original_max_position_embeddings,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "RopeScalingConfig":
        """Create from dictionary"""
        return cls(**config_dict)


@dataclass
class MiniCPM41ModelConfig(ModelConfig):
    """MiniCPM4.1-8B specific model configuration"""

    # Basic architecture
    model_type: str = "minicpm41"
    hidden_size: int = 3584
    num_hidden_layers: int = 32
    num_attention_heads: int = 28
    num_key_value_heads: int = 28
    intermediate_size: int = 14336
    rms_norm_eps: float = 1e-6

    # Vocabulary
    vocab_size: int = 151936
    pad_token_id: Optional[int] = None
    bos_token_id: int = 1
    eos_token_id: int = 2

    # Position embeddings
    max_position_embeddings: int = 65536
    rope_theta: float = 10000.0
    rope_traditional: bool = False

    # Special features
    use_qk_norm: bool = False
    use_cache: bool = True
    tie_word_embeddings: bool = False

    # Sparse attention
    sparse_config: Optional[Dict] = None

    # RoPE scaling
    rope_scaling: Optional[Dict] = None

    # Generation settings
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_new_tokens: int = 4096

    # Mixed reasoning mode
    enable_thinking: bool = False  # Enable mixed reasoning mode

    def __post_init__(self):
        """Initialize and validate configuration"""
        super().__post_init__()

        # Process sparse attention config
        if self.sparse_config is not None:
            if isinstance(self.sparse_config, dict):
                self.sparse_config = SparseAttentionConfig.from_dict(self.sparse_config)
        else:
            # Set default sparse config for long context
            self.sparse_config = SparseAttentionConfig()

        # Process rope scaling config
        if self.rope_scaling is not None:
            if isinstance(self.rope_scaling, dict):
                self.rope_scaling = RopeScalingConfig.from_dict(self.rope_scaling)
        else:
            # Set default rope scaling for long context
            self.rope_scaling = RopeScalingConfig()

        # Validate architecture parameters
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if self.num_key_value_heads <= 0:
            raise ValueError("num_key_value_heads must be positive")
        if self.num_key_value_heads > self.num_attention_heads:
            raise ValueError("num_key_value_heads cannot exceed num_attention_heads")

        # Calculate head dimension
        self.head_dim = self.hidden_size // self.num_attention_heads
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

    def get_model_info(self) -> Dict:
        """Get comprehensive model information"""
        return {
            "model_type": self.model_type,
            "architecture": "MiniCPM41ForCausalLM",
            "hidden_size": self.hidden_size,
            "num_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "intermediate_size": self.intermediate_size,
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
            "head_dim": self.head_dim,
            "use_qk_norm": self.use_qk_norm,
            "sparse_attention": self.sparse_config is not None,
            "rope_scaling": self.rope_scaling is not None,
            "mixed_reasoning": self.enable_thinking,
            "total_params": self._calculate_params(),
        }

    def _calculate_params(self) -> int:
        """Calculate total number of parameters"""
        # Embedding parameters
        embedding_params = self.vocab_size * self.hidden_size

        # Layer parameters
        layer_params = 0
        for _ in range(self.num_hidden_layers):
            # Attention parameters
            # QKV projection
            qkv_params = self.hidden_size * (3 * self.hidden_size)
            # Output projection
            o_params = self.hidden_size * self.hidden_size
            # Attention normalization
            if self.use_qk_norm:
                q_norm_params = self.head_dim * self.num_attention_heads
                k_norm_params = self.head_dim * self.num_key_value_heads
                attention_norm_params = q_norm_params + k_norm_params
            else:
                attention_norm_params = 0

            # MLP parameters (SwiGLU)
            gate_params = self.hidden_size * self.intermediate_size
            up_params = self.hidden_size * self.intermediate_size
            down_params = self.intermediate_size * self.hidden_size

            # Layer normalization
            input_norm_params = self.hidden_size
            post_norm_params = self.hidden_size

            layer_params += (
                qkv_params + o_params + attention_norm_params +
                gate_params + up_params + down_params +
                input_norm_params + post_norm_params
            )

        # Final normalization
        final_norm_params = self.hidden_size

        # LM head
        if not self.tie_word_embeddings:
            lm_head_params = self.vocab_size * self.hidden_size
        else:
            lm_head_params = 0

        total_params = (
            embedding_params + layer_params +
            final_norm_params + lm_head_params
        )

        return total_params

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        config_dict = super().to_dict()

        # Add MiniCPM4.1 specific fields
        config_dict.update({
            "model_type": self.model_type,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "intermediate_size": self.intermediate_size,
            "rms_norm_eps": self.rms_norm_eps,
            "vocab_size": self.vocab_size,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "max_position_embeddings": self.max_position_embeddings,
            "rope_theta": self.rope_theta,
            "rope_traditional": self.rope_traditional,
            "use_qk_norm": self.use_qk_norm,
            "use_cache": self.use_cache,
            "tie_word_embeddings": self.tie_word_embeddings,
            "enable_thinking": self.enable_thinking,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_new_tokens": self.max_new_tokens,
        })

        # Add nested configs
        if self.sparse_config:
            config_dict["sparse_config"] = self.sparse_config.to_dict()
        if self.rope_scaling:
            config_dict["rope_scaling"] = self.rope_scaling.to_dict()

        return config_dict

    @classmethod
    def from_hf_config(cls, hf_config) -> "MiniCPM41ModelConfig":
        """Create MiniCPM4.1 config from HuggingFace config"""
        return cls(
            model_type=getattr(hf_config, "model_type", "minicpm41"),
            hidden_size=getattr(hf_config, "hidden_size", 3584),
            num_hidden_layers=getattr(hf_config, "num_hidden_layers", 32),
            num_attention_heads=getattr(hf_config, "num_attention_heads", 28),
            num_key_value_heads=getattr(hf_config, "num_key_value_heads", 28),
            intermediate_size=getattr(hf_config, "intermediate_size", 14336),
            rms_norm_eps=getattr(hf_config, "rms_norm_eps", 1e-6),
            vocab_size=getattr(hf_config, "vocab_size", 151936),
            pad_token_id=getattr(hf_config, "pad_token_id", None),
            bos_token_id=getattr(hf_config, "bos_token_id", 1),
            eos_token_id=getattr(hf_config, "eos_token_id", 2),
            max_position_embeddings=getattr(hf_config, "max_position_embeddings", 65536),
            rope_theta=getattr(hf_config, "rope_theta", 10000.0),
            rope_traditional=getattr(hf_config, "rope_traditional", False),
            use_qk_norm=getattr(hf_config, "use_qk_norm", False),
            use_cache=getattr(hf_config, "use_cache", True),
            tie_word_embeddings=getattr(hf_config, "tie_word_embeddings", False),
            sparse_config=getattr(hf_config, "sparse_config", None),
            rope_scaling=getattr(hf_config, "rope_scaling", None),
        )


def create_default_minicpm41_config(**kwargs) -> MiniCPM41ModelConfig:
    """Create default MiniCPM4.1-8B configuration"""
    return MiniCPM41ModelConfig(**kwargs)


def get_optimized_cache_config(seq_len: int, model_config: MiniCPM41ModelConfig) -> Dict:
    """Get optimized cache configuration based on sequence length and model config"""

    # For short sequences, use standard caching
    if seq_len <= 4096:
        return {
            "cache_mode": "normal",
            "max_batch_size": 32,
            "block_size": 16,
            "enable_prefix_cache": False,
        }

    # For medium sequences, enable prefix caching
    elif seq_len <= 16384:
        return {
            "cache_mode": "prefix",
            "max_batch_size": 16,
            "block_size": 32,
            "enable_prefix_cache": True,
            "prefix_cache_size": 1024,
        }

    # For long sequences, optimize for sparse attention
    else:
        return {
            "cache_mode": "sparse",
            "max_batch_size": 8,
            "block_size": model_config.sparse_config.block_size,
            "enable_prefix_cache": True,
            "prefix_cache_size": 2048,
            "sparse_window_size": model_config.sparse_config.window_size,
        }


def get_quantization_config(quant_type: str, **kwargs) -> Dict:
    """Get quantization configuration for MiniCPM4.1-8B"""

    base_config = {
        "quant_type": quant_type,
        "quant_algo": "weight_only",  # MiniCPM4.1 uses weight-only quantization by default
    }

    if quant_type == "W8A16":
        base_config.update({
            "weight_bits": 8,
            "activation_bits": 16,
            "group_size": kwargs.get("group_size", -1),  # Per-channel or per-group
        })
    elif quant_type == "W4A8":
        base_config.update({
            "weight_bits": 4,
            "activation_bits": 8,
            "group_size": kwargs.get("group_size", 128),
        })
    elif quant_type == "W4AFP8":
        base_config.update({
            "weight_bits": 4,
            "activation_bits": 8,
            "activation_type": "fp8",
            "group_size": kwargs.get("group_size", 128),
        })
    elif quant_type == "FP8":
        base_config.update({
            "weight_bits": 8,
            "activation_bits": 8,
            "weight_type": "fp8",
            "activation_type": "fp8",
        })

    return base_config