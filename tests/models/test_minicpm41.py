"""
Test cases for MiniCPM4.1-8B model implementation

This module contains unit tests for the MiniCPM4.1-8B model in FastDeploy,
including model loading, inference, and configuration tests.
"""

import unittest
from unittest.mock import patch, MagicMock

import paddle
import pytest
from transformers import AutoConfig

from fastdeploy.config import FDConfig
from fastdeploy.config.minicpm41_config import (
    MiniCPM41ModelConfig,
    SparseAttentionConfig,
    RopeScalingConfig,
)
from fastdeploy.model_executor.models.minicpm41 import (
    MiniCPM41ForCausalLM,
    MiniCPM41DecoderLayer,
    MiniCPM41Attention,
    MiniCPM41MLP,
)


class TestMiniCPM41Config(unittest.TestCase):
    """Test MiniCPM4.1-8B configuration"""

    def test_default_config(self):
        """Test default configuration creation"""
        config = MiniCPM41ModelConfig()

        # Check basic parameters
        self.assertEqual(config.model_type, "minicpm41")
        self.assertEqual(config.hidden_size, 3584)
        self.assertEqual(config.num_hidden_layers, 32)
        self.assertEqual(config.num_attention_heads, 28)
        self.assertEqual(config.num_key_value_heads, 28)
        self.assertEqual(config.vocab_size, 151936)

        # Check special features
        self.assertIsInstance(config.sparse_config, SparseAttentionConfig)
        self.assertIsInstance(config.rope_scaling, RopeScalingConfig)
        self.assertFalse(config.use_qk_norm)
        self.assertTrue(config.use_cache)

    def test_custom_config(self):
        """Test custom configuration parameters"""
        custom_config = {
            "hidden_size": 4096,
            "num_hidden_layers": 24,
            "num_attention_heads": 32,
            "use_qk_norm": True,
            "enable_thinking": True,
        }

        config = MiniCPM41ModelConfig(**custom_config)

        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.num_hidden_layers, 24)
        self.assertEqual(config.num_attention_heads, 32)
        self.assertTrue(config.use_qk_norm)
        self.assertTrue(config.enable_thinking)

    def test_sparse_attention_config(self):
        """Test sparse attention configuration"""
        sparse_config = SparseAttentionConfig(
            kernel_size=64,
            window_size=4096,
            topk=128,
        )

        self.assertEqual(sparse_config.kernel_size, 64)
        self.assertEqual(sparse_config.window_size, 4096)
        self.assertEqual(sparse_config.topk, 128)

        # Test dict conversion
        sparse_dict = sparse_config.to_dict()
        self.assertIn("kernel_size", sparse_dict)
        self.assertEqual(sparse_dict["kernel_size"], 64)

    def test_rope_scaling_config(self):
        """Test RoPE scaling configuration"""
        rope_config = RopeScalingConfig(
            rope_type="longrope",
            original_max_position_embeddings=131072,
        )

        self.assertEqual(rope_config.rope_type, "longrope")
        self.assertEqual(rope_config.original_max_position_embeddings, 131072)

        # Test dict conversion
        rope_dict = rope_config.to_dict()
        self.assertIn("rope_type", rope_dict)
        self.assertEqual(rope_dict["rope_type"], "longrope")

    def test_parameter_validation(self):
        """Test parameter validation"""
        # Test invalid parameters
        with self.assertRaises(ValueError):
            MiniCPM41ModelConfig(hidden_size=-1)

        with self.assertRaises(ValueError):
            MiniCPM41ModelConfig(num_attention_heads=0)

        with self.assertRaises(ValueError):
            MiniCPM41ModelConfig(num_key_value_heads=40, num_attention_heads=32)

        # Test invalid sparse config
        with self.assertRaises(ValueError):
            SparseAttentionConfig(kernel_size=0)

        with self.assertRaises(ValueError):
            SparseAttentionConfig(topk=-1)

    def test_model_info(self):
        """Test model information calculation"""
        config = MiniCPM41ModelConfig()
        model_info = config.get_model_info()

        # Check required fields
        required_fields = [
            "model_type", "architecture", "hidden_size", "num_layers",
            "vocab_size", "head_dim", "total_params"
        ]

        for field in required_fields:
            self.assertIn(field, model_info)

        # Check parameter calculation
        self.assertGreater(model_info["total_params"], 0)
        self.assertEqual(model_info["head_dim"], config.hidden_size // config.num_attention_heads)


class TestMiniCPM41Components(unittest.TestCase):
    """Test MiniCPM4.1-8B model components"""

    def setUp(self):
        """Set up test configuration"""
        self.test_config = MiniCPM41ModelConfig(
            hidden_size=512,  # Small size for testing
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=8,
            intermediate_size=2048,
            vocab_size=1000,
            max_position_embeddings=2048,
        )

        # Create FDConfig
        self.fd_config = FDConfig(
            model_config=self.test_config,
            test_mode=True,
        )

    def test_mlp_component(self):
        """Test MLP component"""
        mlp = MiniCPM41MLP(self.fd_config, layer_id=0)

        # Check component creation
        self.assertIsInstance(mlp.gate_proj, paddle.nn.Layer)
        self.assertIsInstance(mlp.up_proj, paddle.nn.Layer)
        self.assertIsInstance(mlp.down_proj, paddle.nn.Layer)

        # Test forward pass
        batch_size, seq_len, hidden_size = 2, 10, self.test_config.hidden_size
        x = paddle.randn([batch_size, seq_len, hidden_size])

        output = mlp(x)
        self.assertEqual(output.shape, [batch_size, seq_len, hidden_size])

    def test_attention_component(self):
        """Test attention component"""
        attention = MiniCPM41Attention(self.fd_config, layer_id=0)

        # Check component creation
        self.assertIsInstance(attention.qkv_proj, paddle.nn.Layer)
        self.assertIsInstance(attention.o_proj, paddle.nn.Layer)

        # Test forward pass
        batch_size, seq_len, hidden_size = 2, 10, self.test_config.hidden_size
        hidden_states = paddle.randn([batch_size, seq_len, hidden_size])

        output = attention(hidden_states)
        self.assertEqual(output[0].shape, [batch_size, seq_len, hidden_size])

    def test_decoder_layer(self):
        """Test decoder layer component"""
        decoder_layer = MiniCPM41DecoderLayer(self.fd_config, layer_id=0)

        # Check components
        self.assertIsInstance(decoder_layer.self_attn, MiniCPM41Attention)
        self.assertIsInstance(decoder_layer.mlp, MiniCPM41MLP)
        self.assertIsInstance(decoder_layer.input_layernorm, paddle.nn.Layer)
        self.assertIsInstance(decoder_layer.post_attention_layernorm, paddle.nn.Layer)

        # Test forward pass
        batch_size, seq_len, hidden_size = 2, 10, self.test_config.hidden_size
        hidden_states = paddle.randn([batch_size, seq_len, hidden_size])

        output = decoder_layer(hidden_states)
        self.assertEqual(output[0].shape, [batch_size, seq_len, hidden_size])


class TestMiniCPM41Model(unittest.TestCase):
    """Test MiniCPM4.1-8B complete model"""

    def setUp(self):
        """Set up test configuration"""
        # Small configuration for testing
        self.test_config = MiniCPM41ModelConfig(
            hidden_size=256,  # Small size for testing
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=8,
            intermediate_size=1024,
            vocab_size=1000,
            max_position_embeddings=512,
            use_qk_norm=False,  # Disable for simplicity
        )

        # Create FDConfig
        self.fd_config = FDConfig(
            model_config=self.test_config,
            test_mode=True,
        )

    def test_model_creation(self):
        """Test model creation"""
        model = MiniCPM41ForCausalLM(self.fd_config)

        # Check model components
        self.assertIsInstance(model.embed_tokens, paddle.nn.Layer)
        self.assertEqual(len(model.layers), self.test_config.num_hidden_layers)
        self.assertIsInstance(model.norm, paddle.nn.Layer)
        self.assertIsInstance(model.lm_head, paddle.nn.Layer)

    def test_model_forward(self):
        """Test model forward pass"""
        model = MiniCPM41ForCausalLM(self.fd_config)

        # Prepare inputs
        batch_size, seq_len = 2, 8
        input_ids = paddle.randint(0, self.test_config.vocab_size, [batch_size, seq_len])

        # Forward pass
        output = model(input_ids)

        # Check outputs
        self.assertIn("last_hidden_state", output)
        self.assertIn("logits", output)

        # Check output shapes
        self.assertEqual(
            output["last_hidden_state"].shape,
            [batch_size, seq_len, self.test_config.hidden_size]
        )
        self.assertEqual(
            output["logits"].shape,
            [batch_size, seq_len, self.test_config.vocab_size]
        )

    def test_model_computation(self):
        """Test model computation methods"""
        model = MiniCPM41ForCausalLM(self.fd_config)

        # Test compute_logits
        batch_size, seq_len, hidden_size = 2, 8, self.test_config.hidden_size
        hidden_states = paddle.randn([batch_size, seq_len, hidden_size])

        logits = model.compute_logits(hidden_states)
        self.assertEqual(logits.shape, [batch_size, seq_len, self.test_config.vocab_size])

    def test_weight_mapping(self):
        """Test weight state dict mapping"""
        model = MiniCPM41ForCausalLM(self.fd_config)

        # Create mock state dict with HF format
        mock_state_dict = {
            "model.embed_tokens.weight": paddle.randn([self.test_config.vocab_size, self.test_config.hidden_size]),
            "model.layers.0.attention.wq.weight": paddle.randn([self.test_config.hidden_size, self.test_config.hidden_size]),
            "model.layers.0.attention.wk.weight": paddle.randn([self.test_config.hidden_size, self.test_config.hidden_size]),
            "model.layers.0.attention.wv.weight": paddle.randn([self.test_config.hidden_size, self.test_config.hidden_size]),
            "model.layers.0.attention.wo.weight": paddle.randn([self.test_config.hidden_size, self.test_config.hidden_size]),
            "model.layers.0.feed_forward.w1.weight": paddle.randn([self.test_config.hidden_size, self.test_config.intermediate_size]),
            "model.layers.0.feed_forward.w2.weight": paddle.randn([self.test_config.intermediate_size, self.test_config.hidden_size]),
            "model.layers.0.feed_forward.w3.weight": paddle.randn([self.test_config.hidden_size, self.test_config.intermediate_size]),
            "model.norm.weight": paddle.randn([self.test_config.hidden_size]),
            "lm_head.weight": paddle.randn([self.test_config.vocab_size, self.test_config.hidden_size]),
        }

        # Test weight mapping
        model.set_state_dict(mock_state_dict)

        # Verify no errors occurred
        self.assertTrue(True)

    def test_prepare_generation_inputs(self):
        """Test generation input preparation"""
        model = MiniCPM41ForCausalLM(self.fd_config)

        batch_size, seq_len = 1, 4
        input_ids = paddle.randint(0, self.test_config.vocab_size, [batch_size, seq_len])

        # Test without past_key_values
        prepared = model.prepare_inputs_for_generation(input_ids)
        self.assertIn("input_ids", prepared)
        self.assertIn("position_ids", prepared)

        # Test with mock past_key_values
        mock_past = [(paddle.randn([1, 1, 2, 8, 32]), paddle.randn([1, 1, 2, 8, 32]))]
        prepared = model.prepare_inputs_for_generation(input_ids, past_key_values=mock_past)
        self.assertIn("past_key_values", prepared)


class TestIntegration(unittest.TestCase):
    """Integration tests for MiniCPM4.1-8B"""

    @patch('transformers.AutoConfig')
    def test_hf_config_conversion(self, mock_auto_config):
        """Test HuggingFace config conversion"""
        # Create mock HF config
        mock_hf_config = MagicMock()
        mock_hf_config.hidden_size = 3584
        mock_hf_config.num_hidden_layers = 32
        mock_hf_config.num_attention_heads = 28
        mock_hf_config.num_key_value_heads = 28
        mock_hf_config.vocab_size = 151936
        mock_hf_config.max_position_embeddings = 65536
        mock_hf_config.rms_norm_eps = 1e-6
        mock_hf_config.use_qk_norm = False
        mock_hf_config.sparse_config = {
            "kernel_size": 32,
            "window_size": 2048,
            "topk": 64,
        }
        mock_hf_config.rope_scaling = {
            "rope_type": "longrope",
            "original_max_position_embeddings": 65536,
        }

        mock_auto_config.return_value = mock_hf_config

        # Convert config
        paddle_config = MiniCPM41ModelConfig.from_hf_config(mock_hf_config)

        # Verify conversion
        self.assertEqual(paddle_config.hidden_size, 3584)
        self.assertEqual(paddle_config.num_attention_heads, 28)
        self.assertIsInstance(paddle_config.sparse_config, SparseAttentionConfig)
        self.assertIsInstance(paddle_config.rope_scaling, RopeScalingConfig)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)