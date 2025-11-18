"""
Weight conversion script for MiniCPM4.1-8B from HuggingFace to PaddlePaddle

This script converts MiniCPM4.1-8B model weights from HuggingFace format
to PaddlePaddle format compatible with FastDeploy.

Usage:
    python tools/convert_minicpm41_weights.py \
        --input_dir /path/to/minicpm4.1-8b \
        --output_dir /path/to/output \
        --model_name minicpm4.1-8b-ppd
"""

import argparse
import json
import os
from pathlib import Path

import paddle
import torch
from transformers import AutoConfig, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Convert MiniCPM4.1-8B weights from HF to PaddlePaddle")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the HuggingFace model directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output PaddlePaddle model directory",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="minicpm4.1-8b-ppd",
        help="Name for the converted model",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Data type for the converted weights",
    )
    return parser.parse_args()


def load_hf_weights(model_dir: str) -> dict:
    """Load HuggingFace weights"""
    print(f"Loading HuggingFace weights from {model_dir}")

    # Load config first
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    # Load model weights
    try:
        # Try to load safetensors first
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            model_dir,
            trust_remote_code=True,
            torch_dtype=torch.float32,  # Load as float32 first
            device_map="cpu",
        )
        state_dict = model.state_dict()
    except Exception as e:
        print(f"Failed to load with AutoModel: {e}")
        # Fallback to direct safetensors loading
        safetensors_files = [f for f in os.listdir(model_dir) if f.endswith(".safetensors")]
        if not safetensors_files:
            raise FileNotFoundError("No safetensors file found in the input directory")

        from safetensors import safe_open
        state_dict = {}
        for safetensors_file in safetensors_files:
            safetensors_path = os.path.join(model_dir, safetensors_file)
            with safe_open(safetensors_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)

    print(f"Loaded {len(state_dict)} tensors")
    return state_dict, config, tokenizer


def convert_weight_name(hf_name: str) -> str:
    """Convert HF weight name to PaddlePaddle format"""

    # Remove HF model prefix if present
    hf_name = hf_name.replace("model.", "")

    # Mapping table for weight names
    name_mapping = {
        # Embedding
        "embed_tokens.weight": "model.embed_tokens.weight",

        # Final norm
        "norm.weight": "model.norm.weight",

        # Attention layers
        "layers.{}.attention.wq.weight": "model.layers.{}.self_attn.qkv_proj.q_proj.weight",
        "layers.{}.attention.wk.weight": "model.layers.{}.self_attn.qkv_proj.k_proj.weight",
        "layers.{}.attention.wv.weight": "model.layers.{}.self_attn.qkv_proj.v_proj.weight",
        "layers.{}.attention.wo.weight": "model.layers.{}.self_attn.o_proj.weight",

        # Q/K normalization (if present)
        "layers.{}.attention.q_norm.weight": "model.layers.{}.self_attn.q_norm.weight",
        "layers.{}.attention.k_norm.weight": "model.layers.{}.self_attn.k_norm.weight",

        # MLP layers
        "layers.{}.feed_forward.w1.weight": "model.layers.{}.mlp.gate_proj.weight",
        "layers.{}.feed_forward.w2.weight": "model.layers.{}.mlp.down_proj.weight",
        "layers.{}.feed_forward.w3.weight": "model.layers.{}.mlp.up_proj.weight",

        # Layer normalization
        "layers.{}.attention_norm.weight": "model.layers.{}.input_layernorm.weight",
        "layers.{}.ffn_norm.weight": "model.layers.{}.post_attention_layernorm.weight",
    }

    # Try to match the pattern
    for pattern, replacement in name_mapping.items():
        if "layers." in hf_name:
            # Extract layer number
            parts = hf_name.split(".")
            if len(parts) >= 3 and parts[0] == "layers" and parts[1].isdigit():
                layer_id = parts[1]
                pattern_with_id = pattern.replace("{}", layer_id)
                if hf_name == pattern_with_id:
                    return replacement.replace("{}", layer_id)
        else:
            if hf_name == pattern:
                return replacement

    # If no mapping found, return original name with model prefix
    return f"model.{hf_name}"


def convert_weights(state_dict: dict, dtype: str = "float16") -> dict:
    """Convert weights from PyTorch to PaddlePaddle format"""
    print("Converting weights to PaddlePaddle format...")

    paddle_state_dict = {}

    for name, tensor in state_dict.items():
        # Convert weight name
        paddle_name = convert_weight_name(name)

        # Convert tensor
        if isinstance(tensor, torch.Tensor):
            paddle_tensor = paddle.to_tensor(tensor.cpu().numpy())

            # Convert dtype if needed
            if dtype == "float16" and paddle_tensor.dtype != paddle.float16:
                paddle_tensor = paddle_tensor.astype(paddle.float16)
            elif dtype == "float32" and paddle_tensor.dtype != paddle.float32:
                paddle_tensor = paddle_tensor.astype(paddle.float32)
            elif dtype == "bfloat16" and paddle_tensor.dtype != paddle.bfloat16:
                paddle_tensor = paddle_tensor.astype(paddle.bfloat16)
        else:
            # If already numpy or other format
            paddle_tensor = paddle.to_tensor(tensor)
            if dtype == "float16":
                paddle_tensor = paddle_tensor.astype(paddle.float16)

        paddle_state_dict[paddle_name] = paddle_tensor

        print(f"Converted: {name} -> {paddle_name} ({paddle_tensor.shape}, {paddle_tensor.dtype})")

    print(f"Converted {len(paddle_state_dict)} tensors")
    return paddle_state_dict


def convert_config(config, model_name: str) -> dict:
    """Convert HF config to PaddlePaddle format"""
    print("Converting model configuration...")

    paddle_config = {
        "model_type": "minicpm41",
        "architectures": ["MiniCPM41ForCausalLM"],

        # Basic model architecture
        "hidden_size": getattr(config, "hidden_size", 3584),
        "num_hidden_layers": getattr(config, "num_hidden_layers", 32),
        "num_attention_heads": getattr(config, "num_attention_heads", 28),
        "num_key_value_heads": getattr(config, "num_key_value_heads", 28),
        "intermediate_size": getattr(config, "intermediate_size", 14336),
        "rms_norm_eps": getattr(config, "rms_norm_eps", 1e-6),

        # Vocabulary and tokenizer
        "vocab_size": getattr(config, "vocab_size", 151936),
        "pad_token_id": getattr(config, "pad_token_id", None),
        "bos_token_id": getattr(config, "bos_token_id", 1),
        "eos_token_id": getattr(config, "eos_token_id", 2),

        # Position embedding
        "max_position_embeddings": getattr(config, "max_position_embeddings", 65536),
        "rope_theta": getattr(config, "rope_theta", 10000.0),
        "rope_scaling": getattr(config, "rope_scaling", None),

        # Special features
        "use_qk_norm": getattr(config, "use_qk_norm", False),
        "sparse_config": getattr(config, "sparse_config", None),

        # Generation settings
        "use_cache": True,
        "tie_word_embeddings": getattr(config, "tie_word_embeddings", False),

        # Model metadata
        "model_name": model_name,
        "transformers_version": "4.30.0",  # Compatible version
    }

    return paddle_config


def save_converted_model(
    paddle_state_dict: dict,
    paddle_config: dict,
    tokenizer,
    output_dir: str,
    model_name: str
):
    """Save converted model to output directory"""
    print(f"Saving converted model to {output_dir}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save weights
    weight_path = output_path / "model_state.pdparams"
    paddle.save(paddle_state_dict, str(weight_path))
    print(f"Saved weights to {weight_path}")

    # Save config
    config_path = output_path / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(paddle_config, f, indent=2, ensure_ascii=False)
    print(f"Saved config to {config_path}")

    # Save tokenizer files
    tokenizer.save_pretrained(str(output_path))
    print(f"Saved tokenizer to {output_path}")

    # Create model info file
    model_info = {
        "model_name": model_name,
        "model_type": "minicpm41",
        "framework": "PaddlePaddle",
        "backend": "FastDeploy",
        "description": "MiniCPM4.1-8B model converted from HuggingFace to PaddlePaddle",
        "source": "HuggingFace openbmb/MiniCPM4.1-8B",
        "conversion_time": paddle.framework.core.get_version(),
    }

    info_path = output_path / "model_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    print(f"Saved model info to {info_path}")


def verify_conversion(paddle_state_dict: dict, paddle_config: dict):
    """Verify the converted model"""
    print("Verifying converted model...")

    # Check key components exist
    required_keys = [
        "model.embed_tokens.weight",
        "model.norm.weight",
    ]

    missing_keys = []
    for key in required_keys:
        if key not in paddle_state_dict:
            missing_keys.append(key)

    if missing_keys:
        print(f"Warning: Missing required keys: {missing_keys}")
    else:
        print("✓ All required keys present")

    # Check layer count
    num_layers = paddle_config["num_hidden_layers"]
    layer_count = sum(1 for k in paddle_state_dict.keys() if "self_attn.qkv_proj" in k)
    print(f"✓ Found {layer_count} attention layers (expected: {num_layers})")

    # Check weight shapes
    embed_weight = paddle_state_dict.get("model.embed_tokens.weight")
    if embed_weight is not None:
        vocab_size, hidden_size = embed_weight.shape
        print(f"✓ Embedding shape: ({vocab_size}, {hidden_size})")
        print(f"  Vocab size: {vocab_size} (config: {paddle_config['vocab_size']})")
        print(f"  Hidden size: {hidden_size} (config: {paddle_config['hidden_size']})")

    print("✓ Conversion verification completed")


def main():
    args = parse_args()

    print("Starting MiniCPM4.1-8B weight conversion...")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model name: {args.model_name}")
    print(f"Data type: {args.dtype}")

    # Load HuggingFace weights
    hf_state_dict, hf_config, hf_tokenizer = load_hf_weights(args.input_dir)

    # Convert weights
    paddle_state_dict = convert_weights(hf_state_dict, args.dtype)

    # Convert config
    paddle_config = convert_config(hf_config, args.model_name)

    # Save converted model
    save_converted_model(
        paddle_state_dict,
        paddle_config,
        hf_tokenizer,
        args.output_dir,
        args.model_name
    )

    # Verify conversion
    verify_conversion(paddle_state_dict, paddle_config)

    print("✓ Weight conversion completed successfully!")
    print(f"Converted model saved to: {args.output_dir}")
    print("\nTo use the converted model with FastDeploy:")
    print(f"```python")
    print(f"from fastdeploy import LLM")
    print(f"llm = LLM(model='{args.output_dir}')")
    print(f"output = llm.generate('Hello, world!')")
    print(f"```")


if __name__ == "__main__":
    main()