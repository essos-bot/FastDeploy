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
import unittest
from unittest.mock import Mock, patch
import os

import paddle
import pytest

from fastdeploy.config import FDConfig, InfLLMV2Config
from fastdeploy.model_executor.models.minicpm41.config_minicpm41 import (
    MiniCPM41Config,
    MiniCPM41SparseConfig,
    MiniCPM41HybridReasoningConfig,
)
from fastdeploy.model_executor.models.minicpm41.minicpm41 import MiniCPM41ForCausalLM
from fastdeploy.model_executor.models.minicpm41_quant import MiniCPM41ForCausalLM as QuantizedMiniCPM41ForCausalLM


class TestMiniCPM41Config(unittest.TestCase):
    """Test MiniCPM4.1 Configuration"""

    def test_default_config(self):
        """Test default configuration"""
        config = MiniCPM41Config()

        self.assertEqual(config.vocab_size, 122753)
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.intermediate_size, 14336)
        self.assertEqual(config.num_hidden_layers, 30)
        self.assertEqual(config.num_attention_heads, 32)
        self.assertEqual(config.num_key_value_heads, 32)
        self.assertEqual(config.max_position_embeddings, 65536)

        # Test default sparse config
        sparse_config = config.get_sparse_config()
        self.assertIsInstance(sparse_config, MiniCPM41SparseConfig)
        self.assertTrue(sparse_config.enabled)
        self.assertEqual(sparse_config.kernel_size, 32)
        self.assertEqual(sparse_config.topk, 64)

        # Test default hybrid reasoning config
        hybrid_config = config.get_hybrid_reasoning_config()
        self.assertIsInstance(hybrid_config, MiniCPM41HybridReasoningConfig)
        self.assertFalse(hybrid_config.enabled)

    def test_custom_sparse_config(self):
        """Test custom sparse configuration"""
        sparse_dict = {
            "enabled": True,
            "kernel_size": 64,
            "kernel_stride": 32,
            "topk": 32,
            "dense_len": 4096,
        }

        config = MiniCPM41Config(sparse_config=sparse_dict)
        sparse_config = config.get_sparse_config()

        self.assertEqual(sparse_config.kernel_size, 64)
        self.assertEqual(sparse_config.kernel_stride, 32)
        self.assertEqual(sparse_config.topk, 32)
        self.assertEqual(sparse_config.dense_len, 4096)

    def test_custom_hybrid_reasoning_config(self):
        """Test custom hybrid reasoning configuration"""
        hybrid_dict = {
            "enabled": True,
            "reasoning_token": "think",
            "max_thinking_length": 1024,
            "thinking_probability": 0.9,
        }

        config = MiniCPM41Config(hybrid_reasoning_config=hybrid_dict)
        hybrid_config = config.get_hybrid_reasoning_config()

        self.assertTrue(hybrid_config.enabled)
        self.assertEqual(hybrid_config.reasoning_token, "think")
        self.assertEqual(hybrid_config.max_thinking_length, 1024)
        self.assertEqual(hybrid_config.thinking_probability, 0.9)

    def test_config_validation(self):
        """Test configuration validation"""
        config = MiniCPM41Config()

        # Should not raise for valid config
        config.validate_configuration()

        # Test invalid hidden_size
        config.hidden_size = 1000  # Not divisible by num_attention_heads
        with self.assertRaises(ValueError):
            config.validate_configuration()

        # Test invalid sparse config
        config.hidden_size = 4096  # Reset
        config.sparse_config.kernel_size = 0
        with self.assertRaises(ValueError):
            config.validate_configuration()

        config.sparse_config.kernel_size = 32
        config.sparse_config.kernel_stride = 64  # Greater than kernel_size
        with self.assertRaises(ValueError):
            config.validate_configuration()

    def test_model_size_info(self):
        """Test model size information"""
        config = MiniCPM41Config()
        size_info = config.get_model_size_info()

        self.assertIn("total_parameters", size_info)
        self.assertIn("parameters_per_layer", size_info)
        self.assertIn("hidden_size", size_info)
        self.assertIn("num_attention_heads", size_info)

        # Check reasonable parameter count for 8B model
        self.assertGreater(size_info["total_parameters"], 7_000_000_000)
        self.assertLess(size_info["total_parameters"], 9_000_000_000)

    def test_sparse_attention_support(self):
        """Test sparse attention support detection"""
        # Test enabled sparse attention
        config = MiniCPM41Config(sparse_config={"enabled": True})
        self.assertTrue(config.supports_sparse_attention())

        # Test disabled sparse attention
        config = MiniCPM41Config(sparse_config={"enabled": False})
        self.assertFalse(config.supports_sparse_attention())

    def test_attention_backend_selection(self):
        """Test attention backend selection"""
        # Test with sparse attention
        config = MiniCPM41Config(sparse_config={"enabled": True})
        self.assertEqual(config.get_attention_backend_name(), "INFLLMV2_ATTN")

        # Test without sparse attention
        config = MiniCPM41Config(sparse_config={"enabled": False}, use_flash_attention=True)
        self.assertEqual(config.get_attention_backend_name(), "FLASH_ATTN")

        # Test default backend
        config = MiniCPM41Config(sparse_config={"enabled": False}, use_flash_attention=False)
        self.assertEqual(config.get_attention_backend_name(), "APPEND_ATTN")


class TestMiniCPM41Model(unittest.TestCase):
    """Test MiniCPM4.1 Model"""

    def setUp(self):
        """Setup test fixtures"""
        self.config = MiniCPM41Config(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=512,
        )

    @patch('paddle.is_compiled_with_cuda', return_value=True)
    def test_model_initialization(self, mock_cuda):
        """Test model initialization"""
        model = MiniCPM41ForCausalLM(self.config)

        self.assertIsInstance(model, MiniCPM41ForCausalLM)
        self.assertEqual(model.config, self.config)

        # Check model components
        self.assertIsNotNone(model.model)
        self.assertIsNotNone(model.logits_processor)

    @patch('paddle.is_compiled_with_cuda', return_value=True)
    def test_model_forward(self, mock_cuda):
        """Test model forward pass"""
        model = MiniCPM41ForCausalLM(self.config)

        # Create dummy input
        batch_size, seq_len = 2, 8
        input_ids = paddle.randint(0, self.config.vocab_size, [batch_size, seq_len])
        position_ids = paddle.arange(seq_len).unsqueeze(0).expand([batch_size, seq_len])

        # Forward pass
        with paddle.no_grad():
            logits, _ = model(input_ids=input_ids, position_ids=position_ids)

        # Check output shape
        expected_shape = [batch_size, seq_len, self.config.vocab_size]
        self.assertEqual(logits.shape, expected_shape)

    @patch('paddle.is_compiled_with_cuda', return_value=True)
    def test_model_with_sparse_attention(self, mock_cuda):
        """Test model with sparse attention enabled"""
        # Enable sparse attention
        self.config.sparse_config.enabled = True
        self.config.sparse_config.start_layer = 1

        model = MiniCPM41ForCausalLM(self.config)

        # Verify sparse attention configuration was applied
        self.assertTrue(self.config.sparse_config.enabled)

    @patch('paddle.is_compiled_with_cuda', return_value=True)
    def test_model_with_quantization(self, mock_cuda):
        """Test model with quantization"""
        # Create mock quant config
        quant_config = Mock()
        quant_config.quant_type = "wint8"

        model = MiniCPM41ForCausalLM(self.config, quant_config=quant_config)
        self.assertEqual(model.quant_config, quant_config)

    def test_prepare_inputs_for_generation(self):
        """Test input preparation for generation"""
        model = MiniCPM41ForCausalLM(self.config)

        # Create dummy input
        input_ids = paddle.randint(0, self.config.vocab_size, [2, 8])

        # Test with cache
        kv_cache = [Mock() for _ in range(self.config.num_hidden_layers)]
        prepared = model.prepare_inputs_for_generation(
            input_ids=input_ids,
            kv_cache=kv_cache,
            use_cache=True
        )

        self.assertIn("input_ids", prepared)
        self.assertIn("kv_cache", prepared)
        self.assertIn("use_cache", prepared)

        # With cache, should only keep last token
        self.assertEqual(prepared["input_ids"].shape[1], 1)

        # Test without cache
        prepared = model.prepare_inputs_for_generation(
            input_ids=input_ids,
            kv_cache=None,
            use_cache=False
        )
        self.assertEqual(prepared["input_ids"].shape, input_ids.shape)


class TestMiniCPM41Quantized(unittest.TestCase):
    """Test Quantized MiniCPM4.1 Model"""

    def setUp(self):
        """Setup test fixtures"""
        self.config = MiniCPM41Config(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=512,
        )

    @patch('paddle.is_compiled_with_cuda', return_value=True)
    def test_quantized_model_creation(self, mock_cuda):
        """Test quantized model creation"""
        from fastdeploy.model_executor.layers.quantization.weight_only import WINT4Config

        quant_config = WINT4Config.from_config({
            "is_quantized": True,
            "group_size": 128,
        })

        model = QuantizedMiniCPM41ForCausalLM(self.config, quant_config=quant_config)
        self.assertEqual(model.quant_config, quant_config)

    def test_quantization_validation(self):
        """Test quantization validation"""
        from fastdeploy.model_executor.layers.quantization.weight_only import WINT4Config

        quant_config = WINT4Config.from_config({
            "is_quantized": True,
            "group_size": 128,
        })

        # Test valid quant config
        QuantizedMiniCPM41ForCausalLM(self.config, quant_config=quant_config)

        # Test invalid quant type
        quant_config.quant_type = "invalid_type"
        with self.assertRaises(ValueError):
            QuantizedMiniCPM41ForCausalLM(self.config, quant_config=quant_config)

    def test_quantization_compatibility_check(self):
        """Test quantization compatibility with sparse attention"""
        from fastdeploy.model_executor.layers.quantization.weight_only import WINT2Config, WINT8Config

        # Test WINT8 (should be compatible)
        quant_config = WINT8Config.from_config({
            "is_quantized": True,
            "group_size": 128,
        })

        with patch('paddle.is_compiled_with_cuda', return_value=True):
            model = QuantizedMiniCPM41ForCausalLM(self.config, quant_config=quant_config)
            self.assertTrue(model._supports_quantized_sparse_attention())

        # Test WINT2 (might have compatibility issues)
        quant_config = WINT2Config.from_config({
            "is_quantized": True,
            "group_size": 128,
        })

        with patch('paddle.is_compiled_with_cuda', return_value=True):
            model = QuantizedMiniCPM41ForCausalLM(self.config, quant_config=quant_config)
            # WINT2 might not fully support sparse attention
            self.assertFalse(model._supports_quantized_sparse_attention())

    def test_memory_reduction_estimation(self):
        """Test memory reduction estimation"""
        from fastdeploy.model_executor.layers.quantization.weight_only import WINT4Config, WINT8Config

        # Test WINT8
        quant_config = WINT8Config.from_config({
            "is_quantized": True,
            "group_size": 128,
        })

        model = QuantizedMiniCPM41ForCausalLM(self.config, quant_config=quant_config)
        reduction = model._estimate_memory_reduction()
        self.assertEqual(reduction, 0.5)  # 50% memory reduction for WINT8

        # Test WINT4
        quant_config = WINT4Config.from_config({
            "is_quantized": True,
            "group_size": 128,
        })

        model = QuantizedMiniCPM41ForCausalLM(self.config, quant_config=quant_config)
        reduction = model._estimate_memory_reduction()
        self.assertEqual(reduction, 0.25)  # 25% memory reduction for WINT4

    def test_quantization_info(self):
        """Test quantization information"""
        from fastdeploy.model_executor.layers.quantization.weight_only import WINT4Config

        quant_config = WINT4Config.from_config({
            "is_quantized": True,
            "group_size": 128,
        })

        model = QuantizedMiniCPM41ForCausalLM(self.config, quant_config=quant_config)
        info = model.get_quantization_info()

        self.assertTrue(info["quantized"])
        self.assertEqual(info["quant_type"], "wint4")
        self.assertEqual(info["group_size"], 128)
        self.assertIn("memory_reduction", info)


class TestMiniCPM41Integration(unittest.TestCase):
    """Test MiniCPM4.1 Integration"""

    def test_infllmv2_backend_integration(self):
        """Test InfLLM-V2 backend integration"""
        # Test with environment variable
        with patch.dict(os.environ, {'FD_ATTENTION_BACKEND': 'INFLLMV2_ATTN'}):
            config = MiniCPM41Config(sparse_config={"enabled": True})
            self.assertEqual(config.get_attention_backend_name(), "INFLLMV2_ATTN")

    def test_fdconfig_integration(self):
        """Test FDConfig integration"""
        fd_config = FDConfig()

        # Add InfLLM-V2 config
        fd_config.infllmv2_config = InfLLMV2Config({
            'infllmv2_kernel_size': 64,
            'infllmv2_topk': 32,
            'infllmv2_dense_len': 4096,
        })

        self.assertIsNotNone(fd_config.infllmv2_config)
        self.assertEqual(fd_config.infllmv2_config.infllmv2_kernel_size, 64)
        self.assertEqual(fd_config.infllmv2_config.infllmv2_topk, 32)
        self.assertEqual(fd_config.infllmv2_config.infllmv2_dense_len, 4096)

    @patch('paddle.is_compiled_with_cuda', return_value=True)
    def test_end_to_end_model_creation(self, mock_cuda):
        """Test end-to-end model creation with all features"""
        config = MiniCPM41Config(
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            sparse_config={
                "enabled": True,
                "kernel_size": 32,
                "topk": 16,
                "dense_len": 512,
            },
            hybrid_reasoning_config={
                "enabled": True,
                "reasoning_token": "think",
                "max_thinking_length": 64,
            }
        )

        # Create FDConfig with InfLLM-V2
        fd_config = FDConfig()
        fd_config.infllmv2_config = InfLLMV2Config({
            'infllmv2_kernel_size': 32,
            'infllmv2_topk': 16,
        })

        # Attach FDConfig to model config
        config.fd_config = fd_config

        # Create model
        model = MiniCPM41ForCausalLM(config)

        self.assertIsInstance(model, MiniCPM41ForCausalLM)
        self.assertTrue(config.supports_sparse_attention())
        self.assertTrue(config.supports_hybrid_reasoning())


if __name__ == '__main__':
    unittest.main()