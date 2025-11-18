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

from fastdeploy import envs
from fastdeploy.config import FDConfig, InfLLMV2Config
from fastdeploy.model_executor.layers.attention.attention_selecter import get_attention_backend
from fastdeploy.platforms.base import _Backend


class TestInfLLMV2Integration(unittest.TestCase):
    """Test InfLLM-V2 integration with FastDeploy"""

    def setUp(self):
        """Setup test fixtures"""
        # Set environment variable for InfLLM-V2 backend
        original_env = os.getenv('FD_ATTENTION_BACKEND')
        os.environ['FD_ATTENTION_BACKEND'] = 'INFLLMV2_ATTN'

        # Create config with InfLLM-V2 parameters
        self.config = FDConfig()
        self.config.infllmv2_config = InfLLMV2Config({
            'infllmv2_kernel_size': 32,
            'infllmv2_kernel_stride': 16,
            'infllmv2_topk': 64,
            'infllmv2_dense_len': 4096,
        })

        # Store original env for cleanup
        self.original_env = original_env

    def tearDown(self):
        """Cleanup test fixtures"""
        if self.original_env is not None:
            os.environ['FD_ATTENTION_BACKEND'] = self.original_env
        else:
            os.environ.pop('FD_ATTENTION_BACKEND', None)

    def test_attention_backend_selection(self):
        """Test attention backend selection for InfLLM-V2"""
        with patch('fastdeploy.envs.FD_ATTENTION_BACKEND', 'INFLLMV2_ATTN'):
            backend = get_attention_backend()
            # Should select the InfLLM-V2 backend path
            self.assertIsInstance(backend, str)
            self.assertIn('InfLLMV2AttentionBackend', backend)

    def test_infllmv2_backend_enum(self):
        """Test InfLLM-V2 backend enumeration"""
        from fastdeploy.platforms.base import _Backend

        # Check that INFLLMV2_ATTN enum exists
        self.assertTrue(hasattr(_Backend, 'INFLLMV2_ATTN'))
        self.assertIsNotNone(_Backend.INFLLMV2_ATTN)

    @patch('paddle.is_compiled_with_cuda', return_value=True)
    def test_cuda_platform_backend_mapping(self, mock_cuda):
        """Test CUDA platform backend mapping for InfLLM-V2"""
        from fastdeploy.platforms.cuda import CUDAPlatform

        platform = CUDAPlatform()
        backend_cls = platform.get_attention_backend_cls(_Backend.INFLLMV2_ATTN)

        self.assertEqual(
            backend_cls,
            "fastdeploy.model_executor.layers.attention.infllmv2_attention_backend.InfLLMV2AttentionBackend"
        )

    def test_infllmv2_config_integration(self):
        """Test InfLLM-V2 config integration with FDConfig"""
        config = FDConfig()
        config.infllmv2_config = InfLLMV2Config({
            'infllmv2_kernel_size': 64,
            'infllmv2_topk': 32,
            'infllmv2_dense_len': 2048,
        })

        self.assertIsNotNone(config.infllmv2_config)
        self.assertEqual(config.infllmv2_config.infllmv2_kernel_size, 64)
        self.assertEqual(config.infllmv2_config.infllmv2_topk, 32)
        self.assertEqual(config.infllmv2_config.infllmv2_dense_len, 2048)

    @patch('paddle.is_compiled_with_cuda', return_value=False)
    def test_cuda_unavailable_error_handling(self, mock_cuda):
        """Test error handling when CUDA is not available"""
        from fastdeploy.model_executor.layers.attention.infllmv2_attention_backend import InfLLMV2AttentionBackend

        config = FDConfig()
        config.infllmv2_config = InfLLMV2Config()

        with self.assertRaises(RuntimeError) as context:
            InfLLMV2AttentionBackend(
                fd_config=config,
                kv_num_heads=32,
                num_heads=32,
                head_dim=128,
            )

        self.assertIn("CUDA support", str(context.exception))

    def test_infllmv2_config_parameter_validation(self):
        """Test InfLLM-V2 config parameter validation"""
        # Test valid config
        valid_config = InfLLMV2Config({
            'infllmv2_kernel_size': 32,
            'infllmv2_kernel_stride': 16,
            'infllmv2_topk': 64,
            'infllmv2_block_size': 64,
            'infllmv2_dense_len': 8192,
        })
        # Should not raise
        self.assertEqual(valid_config.infllmv2_kernel_size, 32)

        # Test invalid kernel stride > kernel size
        with self.assertRaises(ValueError):
            InfLLMV2Config({
                'infllmv2_kernel_size': 16,
                'infllmv2_kernel_stride': 32,
            })

        # Test negative dense_len
        with self.assertRaises(ValueError):
            InfLLMV2Config({'infllmv2_dense_len': -1})

    def test_infllmv2_attention_metadata_creation(self):
        """Test InfLLM-V2 attention metadata creation"""
        from fastdeploy.model_executor.layers.attention.infllmv2_attention_backend import InfLLMV2AttentionBackend
        from fastdeploy.model_executor.forward_meta import ForwardMeta

        # Create mock config
        config = Mock()
        config.infllmv2_kernel_size = 32
        config.infllmv2_kernel_stride = 16
        config.infllmv2_topk = 64
        config.infllmv2_dense_len = 8192
        config.infllmv2_block_size = 64
        config.infllmv2_window_size = 2048
        config.infllmv2_use_nope = False
        config.infllmv2_init_blocks = 1

        # Mock CUDA availability
        with patch('paddle.is_compiled_with_cuda', return_value=True):
            backend = InfLLMV2AttentionBackend(
                fd_config=config,
                kv_num_heads=32,
                num_heads=32,
                head_dim=128,
            )

            # Create mock forward meta
            forward_meta = Mock(spec=ForwardMeta)
            forward_meta.seq_len = 1024

            # Test metadata creation
            metadata = backend.init_attention_metadata(forward_meta)

            self.assertIsNotNone(metadata)
            self.assertEqual(metadata.kernel_size, 32)
            self.assertEqual(metadata.kernel_stride, 16)
            self.assertEqual(metadata.topk, 64)
            self.assertEqual(metadata.current_seq_len, 1024)


if __name__ == '__main__':
    unittest.main()