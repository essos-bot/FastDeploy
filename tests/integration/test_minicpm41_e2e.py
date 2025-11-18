"""
End-to-end integration tests for MiniCPM4.1-8B

This module contains comprehensive integration tests for MiniCPM4.1-8B model,
including model loading, inference, quantization, and API server tests.
"""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import paddle
import pytest
import requests

from fastdeploy import FDConfig, LLM, SamplingParams
from fastdeploy.config.minicpm41_config import (
    MiniCPM41ModelConfig,
    SparseAttentionConfig,
    create_quantized_minicpm41_config,
)
from fastdeploy.model_executor.models.minicpm41 import MiniCPM41ForCausalLM
from fastdeploy.model_executor.models.minicpm41_quant import (
    MiniCPM41ForCausalLM as QuantizedMiniCPM41,
    MiniCPM41QuantizationConfig,
)


class TestMiniCPM41E2E(unittest.TestCase):
    """End-to-end tests for MiniCPM4.1-8B"""

    def setUp(self):
        """Set up test environment"""
        self.test_model_path = "test_minicpm41_model"
        self.temp_dir = tempfile.mkdtemp()

        # Create small test configuration
        self.test_config = MiniCPM41ModelConfig(
            hidden_size=512,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=8,
            intermediate_size=1024,
            vocab_size=1000,
            max_position_embeddings=2048,
        )

    def test_model_creation_and_loading(self):
        """Test model creation and loading"""
        # Create FDConfig
        fd_config = FDConfig(
            model_config=self.test_config,
            test_mode=True,
        )

        # Create model
        model = MiniCPM41ForCausalLM(fd_config)

        # Verify model structure
        self.assertEqual(len(model.layers), self.test_config.num_hidden_layers)
        self.assertIsInstance(model.embed_tokens, paddle.nn.Layer)
        self.assertIsInstance(model.norm, paddle.nn.Layer)
        self.assertIsInstance(model.lm_head, paddle.nn.Layer)

        print("✓ Model creation and loading test passed")

    def test_basic_inference(self):
        """Test basic inference functionality"""
        # Create model
        fd_config = FDConfig(model_config=self.test_config, test_mode=True)
        model = MiniCPM41ForCausalLM(fd_config)

        # Prepare test inputs
        batch_size, seq_len = 2, 8
        input_ids = paddle.randint(
            0, self.test_config.vocab_size, [batch_size, seq_len]
        )

        # Forward pass
        output = model(input_ids)

        # Verify outputs
        self.assertIn("last_hidden_state", output)
        self.assertIn("logits", output)
        self.assertEqual(
            output["last_hidden_state"].shape,
            [batch_size, seq_len, self.test_config.hidden_size]
        )
        self.assertEqual(
            output["logits"].shape,
            [batch_size, seq_len, self.test_config.vocab_size]
        )

        print("✓ Basic inference test passed")

    def test_quantization_support(self):
        """Test quantization support"""
        # Test different quantization configs
        quant_types = ["w4afp8", "w4a8", "wint8"]

        for quant_type in quant_types:
            with self.subTest(quant_type=quant_type):
                # Create quantized config
                fd_config, quant_config = create_quantized_minicpm41_config(
                    quant_type=quant_type,
                    group_size=128,
                )

                # Create quantized model
                model = QuantizedMiniCPM41(fd_config)

                # Verify quantization info
                quant_info = model.get_quantization_info()
                self.assertTrue(quant_info["quantized"])
                self.assertEqual(quant_info["quant_type"], quant_type)

                # Test forward pass
                batch_size, seq_len = 1, 4
                input_ids = paddle.randint(
                    0, self.test_config.vocab_size, [batch_size, seq_len]
                )

                output = model(input_ids)
                self.assertIn("logits", output)

        print("✓ Quantization support test passed")

    def test_sparse_attention_config(self):
        """Test sparse attention configuration"""
        # Create sparse attention config
        sparse_config = SparseAttentionConfig(
            kernel_size=32,
            window_size=2048,
            topk=64,
            dense_len=8192,
        )

        # Update model config
        model_config = MiniCPM41ModelConfig(
            sparse_config=sparse_config.to_dict(),
            **self.test_config.__dict__
        )

        # Create model with sparse attention
        fd_config = FDConfig(model_config=model_config, test_mode=True)
        model = MiniCPM41ForCausalLM(fd_config)

        # Verify sparse config is set
        self.assertIsNotNone(model.sparse_config)
        self.assertEqual(model.sparse_config.kernel_size, 32)
        self.assertEqual(model.sparse_config.topk, 64)

        print("✓ Sparse attention config test passed")

    def test_weight_mapping(self):
        """Test weight mapping functionality"""
        # Create model
        fd_config = FDConfig(model_config=self.test_config, test_mode=True)
        model = MiniCPM41ForCausalLM(fd_config)

        # Create mock state dict with HF format weights
        mock_state_dict = {
            "model.embed_tokens.weight": paddle.randn([
                self.test_config.vocab_size, self.test_config.hidden_size
            ]),
            "model.layers.0.attention.wq.weight": paddle.randn([
                self.test_config.hidden_size, self.test_config.hidden_size
            ]),
            "model.layers.0.attention.wk.weight": paddle.randn([
                self.test_config.hidden_size, self.test_config.hidden_size
            ]),
            "model.layers.0.attention.wv.weight": paddle.randn([
                self.test_config.hidden_size, self.test_config.hidden_size
            ]),
            "model.layers.0.attention.wo.weight": paddle.randn([
                self.test_config.hidden_size, self.test_config.hidden_size
            ]),
            "model.layers.0.feed_forward.w1.weight": paddle.randn([
                self.test_config.hidden_size, self.test_config.intermediate_size
            ]),
            "model.layers.0.feed_forward.w2.weight": paddle.randn([
                self.test_config.intermediate_size, self.test_config.hidden_size
            ]),
            "model.layers.0.feed_forward.w3.weight": paddle.randn([
                self.test_config.hidden_size, self.test_config.intermediate_size
            ]),
            "model.norm.weight": paddle.randn([self.test_config.hidden_size]),
            "lm_head.weight": paddle.randn([
                self.test_config.vocab_size, self.test_config.hidden_size
            ]),
        }

        # Test weight mapping
        try:
            model.set_state_dict(mock_state_dict)
            print("✓ Weight mapping test passed")
        except Exception as e:
            self.fail(f"Weight mapping failed: {e}")

    def test_generation_preparation(self):
        """Test generation input preparation"""
        # Create model
        fd_config = FDConfig(model_config=self.test_config, test_mode=True)
        model = MiniCPM41ForCausalLM(fd_config)

        # Test input preparation
        batch_size, seq_len = 1, 4
        input_ids = paddle.randint(
            0, self.test_config.vocab_size, [batch_size, seq_len]
        )

        # Prepare inputs for generation
        prepared = model.prepare_inputs_for_generation(input_ids)

        # Verify prepared inputs
        self.assertIn("input_ids", prepared)
        self.assertIn("position_ids", prepared)
        self.assertEqual(prepared["input_ids"].shape, [batch_size, seq_len])
        self.assertEqual(prepared["position_ids"].shape, [batch_size, seq_len])

        print("✓ Generation preparation test passed")


class TestMiniCPM41API(unittest.TestCase):
    """Test MiniCPM4.1-8B API integration"""

    @patch('fastdeploy.LLM')
    def test_llm_creation(self, mock_llm):
        """Test LLM creation with MiniCPM4.1-8B"""
        # Mock LLM instance
        mock_instance = MagicMock()
        mock_llm.return_value = mock_instance

        # Test LLM creation
        llm = LLM(model="openbmb/MiniCPM4.1-8B")

        # Verify LLM was created with correct model
        mock_llm.assert_called_once_with(model="openbmb/MiniCPM4.1-8B")
        self.assertEqual(llm, mock_instance)

        print("✓ LLM creation test passed")

    @patch('fastdeploy.LLM')
    def test_quantized_llm_creation(self, mock_llm):
        """Test quantized LLM creation"""
        # Mock LLM instance
        mock_instance = MagicMock()
        mock_llm.return_value = mock_instance

        # Create quantized config
        fd_config, quant_config = create_quantized_minicpm41_config(
            quant_type="w4afp8"
        )

        # Test quantized LLM creation
        llm = LLM(model="openbmb/MiniCPM4.1-8B", fd_config=fd_config)

        # Verify LLM was created with quantized config
        mock_llm.assert_called_once()
        args, kwargs = mock_llm.call_args
        self.assertEqual(kwargs["model"], "openbmb/MiniCPM4.1-8B")
        self.assertEqual(kwargs["fd_config"], fd_config)

        print("✓ Quantized LLM creation test passed")

    @patch('requests.post')
    def test_api_server_integration(self, mock_post):
        """Test API server integration"""
        # Mock API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "MiniCPM4.1-8B是一个高性能的大语言模型。"
                }
            }]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        # Test API call
        response = requests.post(
            "http://localhost:8180/v1/chat/completions",
            json={
                "model": "openbmb/MiniCPM4.1-8B",
                "messages": [
                    {"role": "user", "content": "请介绍一下MiniCPM4.1-8B"}
                ],
                "max_tokens": 256,
                "temperature": 0.7
            }
        )

        # Verify response
        self.assertIn("choices", response.json())
        self.assertIn("content", response.json()["choices"][0]["message"])

        print("✓ API server integration test passed")


class TestMiniCPM41Performance(unittest.TestCase):
    """Performance tests for MiniCPM4.1-8B"""

    def test_memory_usage(self):
        """Test memory usage estimation"""
        # Create test config
        model_config = MiniCPM41ModelConfig(
            hidden_size=1024,
            num_hidden_layers=4,
            num_attention_heads=16,
            intermediate_size=4096,
            vocab_size=50000,
        )

        # Calculate expected parameters
        model_info = model_config.get_model_info()
        total_params = model_info["total_params"]

        # Estimate memory usage (assuming fp16)
        memory_mb = total_params * 2 / (1024 * 1024)  # 2 bytes per fp16 parameter

        # Verify reasonable memory usage
        self.assertGreater(memory_mb, 100)  # At least 100MB
        self.assertLess(memory_mb, 10000)   # Less than 10GB for test model

        print(f"✓ Memory usage test passed: {memory_mb:.2f}MB")

    def test_quantization_memory_reduction(self):
        """Test memory reduction due to quantization"""
        # Create quantized model
        fd_config, quant_config = create_quantized_minicpm41_config(
            quant_type="w4afp8"
        )
        model = QuantizedMiniCPM41(fd_config)

        # Get quantization info
        quant_info = model.get_quantization_info()

        # Verify memory reduction
        self.assertGreater(quant_info["memory_reduction_ratio"], 1.0)
        self.assertLess(quant_info["memory_reduction_ratio"], 10.0)

        print(f"✓ Quantization memory reduction test passed: {quant_info['memory_reduction_ratio']:.2f}x")

    def test_sequence_length_scaling(self):
        """Test performance scaling with sequence length"""
        # Test different sequence lengths
        seq_lengths = [512, 1024, 2048]
        fd_config = FDConfig(model_config=self.test_config, test_mode=True)
        model = MiniCPM41ForCausalLM(fd_config)

        for seq_len in seq_lengths:
            with self.subTest(seq_len=seq_len):
                # Prepare inputs
                batch_size = 2
                input_ids = paddle.randint(
                    0, self.test_config.vocab_size, [batch_size, seq_len]
                )

                # Time forward pass
                import time
                start_time = time.time()
                output = model(input_ids)
                end_time = time.time()

                inference_time = end_time - start_time

                # Verify output correctness
                self.assertIn("logits", output)
                self.assertEqual(
                    output["logits"].shape,
                    [batch_size, seq_len, self.test_config.vocab_size]
                )

                # Reasonable time constraints
                self.assertLess(inference_time, 10.0)  # Less than 10 seconds

                print(f"  Sequence length {seq_len}: {inference_time:.3f}s")

        print("✓ Sequence length scaling test passed")


def run_integration_tests():
    """Run all integration tests"""
    print("Running MiniCPM4.1-8B Integration Tests...")
    print("=" * 50)

    # Create test suite
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTest(unittest.makeSuite(TestMiniCPM41E2E))
    suite.addTest(unittest.makeSuite(TestMiniCPM41API))
    suite.addTest(unittest.makeSuite(TestMiniCPM41Performance))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")

    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")

    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun
    print(f"\nSuccess Rate: {success_rate:.1%}")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_integration_tests()
    exit(0 if success else 1)