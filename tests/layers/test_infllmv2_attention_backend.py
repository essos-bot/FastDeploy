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

import paddle
import pytest

from fastdeploy.config import InfLLMV2Config, FDConfig
from fastdeploy.model_executor.layers.attention.infllmv2_attention_backend import (
    InfLLMV2AttentionBackend,
)
from fastdeploy.model_executor.layers.attention.infllmv2_attention_metadata import (
    InfLLMV2AttentionMetadata,
)
from fastdeploy.model_executor.forward_meta import ForwardMeta


class TestInfLLMV2AttentionBackend(unittest.TestCase):
    """Test InfLLM-V2 Attention Backend"""

    def setUp(self):
        """Setup test fixtures"""
        self.fd_config = Mock(spec=FDConfig)
        self.fd_config.infllmv2_kernel_size = 32
        self.fd_config.infllmv2_kernel_stride = 16
        self.fd_config.infllmv2_topk = 64
        self.fd_config.infllmv2_dense_len = 8192
        self.fd_config.infllmv2_block_size = 64
        self.fd_config.infllmv2_window_size = 2048
        self.fd_config.infllmv2_use_nope = False
        self.fd_config.infllmv2_init_blocks = 1

        self.backend = InfLLMV2AttentionBackend(
            fd_config=self.fd_config,
            kv_num_heads=32,
            num_heads=32,
            head_dim=128,
        )

    def test_initialization(self):
        """Test backend initialization"""
        self.assertEqual(self.backend.kernel_size, 32)
        self.assertEqual(self.backend.kernel_stride, 16)
        self.assertEqual(self.backend.topk, 64)
        self.assertEqual(self.backend.dense_len, 8192)
        self.assertEqual(self.backend.block_size, 64)
        self.assertEqual(self.backend.window_size, 2048)
        self.assertEqual(self.backend.use_nope, False)
        self.assertEqual(self.backend.init_blocks, 1)

    def test_parameter_validation(self):
        """Test parameter validation"""
        # Test invalid kernel_size
        self.fd_config.infllmv2_kernel_size = 0
        with self.assertRaises(ValueError):
            InfLLMV2AttentionBackend(
                fd_config=self.fd_config,
                kv_num_heads=32,
                num_heads=32,
                head_dim=128,
            )

        # Test invalid kernel_stride
        self.fd_config.infllmv2_kernel_size = 32
        self.fd_config.infllmv2_kernel_stride = 0
        with self.assertRaises(ValueError):
            InfLLMV2AttentionBackend(
                fd_config=self.fd_config,
                kv_num_heads=32,
                num_heads=32,
                head_dim=128,
            )

        # Test kernel_stride > kernel_size
        self.fd_config.infllmv2_kernel_size = 16
        self.fd_config.infllmv2_kernel_stride = 32
        with self.assertRaises(ValueError):
            InfLLMV2AttentionBackend(
                fd_config=self.fd_config,
                kv_num_heads=32,
                num_heads=32,
                head_dim=128,
            )

    def test_init_attention_metadata(self):
        """Test attention metadata initialization"""
        forward_meta = Mock(spec=ForwardMeta)
        forward_meta.seq_len = 1024

        metadata = self.backend.init_attention_metadata(forward_meta)

        self.assertIsInstance(metadata, InfLLMV2AttentionMetadata)
        self.assertEqual(metadata.kernel_size, 32)
        self.assertEqual(metadata.kernel_stride, 16)
        self.assertEqual(metadata.topk, 64)
        self.assertEqual(metadata.dense_len, 8192)
        self.assertEqual(metadata.current_seq_len, 1024)
        self.assertEqual(metadata.block_size, 64)
        self.assertEqual(metadata.window_size, 2048)
        self.assertEqual(metadata.use_nope, False)
        self.assertEqual(metadata.init_blocks, 1)

    def test_should_use_sparse_attention(self):
        """Test sparse attention decision logic"""
        metadata = InfLLMV2AttentionMetadata(dense_len=8192)

        # Test sequence shorter than dense_len
        self.assertFalse(metadata.should_use_sparse_attention(4096))

        # Test sequence longer than dense_len
        self.assertTrue(metadata.should_use_sparse_attention(16384))

        # Test always use sparse attention
        metadata_always = InfLLMV2AttentionMetadata(dense_len=-1)
        self.assertTrue(metadata_always.should_use_sparse_attention(1024))

    def test_get_num_blocks(self):
        """Test block count calculation"""
        metadata = InfLLMV2AttentionMetadata(block_size=64)

        # Test exact block multiple
        self.assertEqual(metadata.get_num_blocks(128), 2)

        # Test partial block
        self.assertEqual(metadata.get_num_blocks(150), 3)

        # Test single token
        self.assertEqual(metadata.get_num_blocks(1), 1)

    def test_get_kernel_position(self):
        """Test kernel position calculation"""
        metadata = InfLLMV2AttentionMetadata(kernel_stride=16)

        self.assertEqual(metadata.get_kernel_position(0), 0)
        self.assertEqual(metadata.get_kernel_position(15), 0)
        self.assertEqual(metadata.get_kernel_position(16), 1)
        self.assertEqual(metadata.get_kernel_position(31), 1)

    @patch('paddle.is_compiled_with_cuda', return_value=True)
    def test_cuda_availability_check(self, mock_cuda):
        """Test CUDA availability check"""
        # Should not raise when CUDA is available
        InfLLMV2AttentionBackend(
            fd_config=self.fd_config,
            kv_num_heads=32,
            num_heads=32,
            head_dim=128,
        )

    @patch('paddle.is_compiled_with_cuda', return_value=False)
    def test_cuda_unavailable_error(self, mock_cuda):
        """Test CUDA unavailability error"""
        with self.assertRaises(RuntimeError) as context:
            InfLLMV2AttentionBackend(
                fd_config=self.fd_config,
                kv_num_heads=32,
                num_heads=32,
                head_dim=128,
            )
        self.assertIn("CUDA support", str(context.exception))

    @patch('fastdeploy.custom_ops')
    def test_stage1_topk_selection_success(self, mock_custom_ops):
        """Test successful Stage 1 Top-K selection"""
        # Setup mock
        mock_topk_result = paddle.to_tensor([[1, 2, 3, 4]])
        mock_custom_ops.infllmv2_stage1_topk_selection.return_value = mock_topk_result

        # Test data
        query = paddle.randn([1, 2, 32, 128])
        key = paddle.randn([1, 1024, 32, 128])
        metadata = InfLLMV2AttentionMetadata()
        forward_meta = Mock(spec=ForwardMeta)
        forward_meta.attn_metadata = metadata

        # Execute
        result = self.backend._stage1_topk_selection(query, key, metadata, "decode")

        # Verify
        mock_custom_ops.infllmv2_stage1_topk_selection.assert_called_once()
        self.assertEqual(result, mock_topk_result)
        self.assertEqual(metadata.topk_indices, mock_topk_result)

    @patch('fastdeploy.custom_ops', side_effect=ImportError("Custom ops not available"))
    def test_stage1_topk_selection_missing_ops(self, mock_custom_ops):
        """Test Stage 1 Top-K selection with missing custom ops"""
        query = paddle.randn([1, 2, 32, 128])
        key = paddle.randn([1, 1024, 32, 128])
        metadata = InfLLMV2AttentionMetadata()

        with self.assertRaises(RuntimeError) as context:
            self.backend._stage1_topk_selection(query, key, metadata, "decode")
        self.assertIn("CUDA kernels not available", str(context.exception))

    @patch('fastdeploy.custom_ops')
    def test_stage2_sparse_attention_success(self, mock_custom_ops):
        """Test successful Stage 2 sparse attention"""
        # Setup mock
        mock_output = paddle.randn([1, 2, 32, 128])
        mock_custom_ops.infllmv2_stage2_sparse_attention.return_value = mock_output

        # Test data
        query = paddle.randn([1, 2, 32, 128])
        key_cache = paddle.randn([1, 1024, 32, 128])
        value_cache = paddle.randn([1, 1024, 32, 128])
        topk_indices = paddle.to_tensor([[1, 2, 3, 4]])
        metadata = InfLLMV2AttentionMetadata()

        # Execute
        result = self.backend._stage2_sparse_attention(
            query, key_cache, value_cache, topk_indices, metadata, "decode"
        )

        # Verify
        mock_custom_ops.infllmv2_stage2_sparse_attention.assert_called_once()
        self.assertEqual(result, mock_output)

    def test_fallback_standard_attention(self):
        """Test fallback to standard attention"""
        query = paddle.randn([1, 2, 32, 128])
        key = paddle.randn([1, 1024, 32, 128])
        value = paddle.randn([1, 1024, 32, 128])
        forward_meta = Mock(spec=ForwardMeta)

        # Execute fallback attention
        result = self.backend._fallback_standard_attention(query, key, value, forward_meta)

        # Verify output shape
        self.assertEqual(result.shape, query.shape)


class TestInfLLMV2Config(unittest.TestCase):
    """Test InfLLM-V2 Configuration"""

    def test_default_config(self):
        """Test default configuration values"""
        config = InfLLMV2Config()

        self.assertEqual(config.infllmv2_kernel_size, 32)
        self.assertEqual(config.infllmv2_kernel_stride, 16)
        self.assertEqual(config.infllmv2_topk, 64)
        self.assertEqual(config.infllmv2_block_size, 64)
        self.assertEqual(config.infllmv2_window_size, 2048)
        self.assertEqual(config.infllmv2_dense_len, 8192)
        self.assertEqual(config.infllmv2_init_blocks, 1)
        self.assertEqual(config.infllmv2_use_nope, False)

    def test_config_from_args(self):
        """Test configuration from arguments"""
        args = {
            'infllmv2_kernel_size': 64,
            'infllmv2_topk': 32,
            'infllmv2_dense_len': 4096,
        }

        config = InfLLMV2Config(args)

        self.assertEqual(config.infllmv2_kernel_size, 64)
        self.assertEqual(config.infllmv2_topk, 32)
        self.assertEqual(config.infllmv2_dense_len, 4096)
        # Other values should remain default
        self.assertEqual(config.infllmv2_kernel_stride, 16)

    def test_config_validation(self):
        """Test configuration validation"""
        # Test invalid values
        with self.assertRaises(ValueError):
            InfLLMV2Config({'infllmv2_kernel_size': 0})

        with self.assertRaises(ValueError):
            InfLLMV2Config({'infllmv2_topk': -1})

        with self.assertRaises(ValueError):
            InfLLMV2Config({'infllmv2_kernel_stride': 64, 'infllmv2_kernel_size': 32})


if __name__ == '__main__':
    unittest.main()