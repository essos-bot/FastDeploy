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
MiniCPM4.1 INT量化功能测试

这个模块专门测试MiniCPM4.1-8B的INT量化功能，包括：
- WINT2/WINT4/WINT8量化配置
- 量化层自动选择
- 与InfLLM-V2稀疏注意力的兼容性
- 量化性能和精度验证
"""

import unittest
from unittest.mock import Mock, patch
import warnings

import paddle
import pytest

from fastdeploy.config import FDConfig, InfLLMV2Config
from fastdeploy.model_executor.layers.quantization import parse_minicpm41_quant_config
from fastdeploy.model_executor.layers.quantization.minicpm41_quant_parser import (
    get_minicpm41_quant_recommendations,
    auto_select_optimal_quant,
)
from fastdeploy.model_executor.layers.quantization.weight_only import (
    WINT2Config,
    WINT4Config,
    WINT8Config,
)
from fastdeploy.model_executor.models.minicpm41.config_minicpm41 import (
    MiniCPM41Config,
    MiniCPM41SparseConfig,
)
from fastdeploy.model_executor.models.minicpm41.minicpm41 import MiniCPM41ForCausalLM
from fastdeploy.model_executor.models.minicpm41_quant import MiniCPM41ForCausalLM as QuantizedMiniCPM41ForCausalLM


class TestMiniCPM41IntQuantConfig(unittest.TestCase):
    """测试MiniCPM4.1 INT量化配置解析"""

    def setUp(self):
        """设置测试参数"""
        self.config = MiniCPM41Config(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
        )

        class MockArgs:
            def __init__(self, quant_type=None):
                self.quantization = quant_type
                self.group_size = 128
                self.permute = False

        self.mock_args = MockArgs()

    def test_parse_wint8_quantization(self):
        """测试WINT8量化配置解析"""
        self.mock_args.quantization = "wint8"

        quant_config = parse_minicpm41_quant_config(self.mock_args, self.config)

        self.assertIsInstance(quant_config, WINT8Config)
        self.assertTrue(quant_config.is_quantized)
        self.assertEqual(quant_config.group_size, 128)

    def test_parse_wint4_quantization(self):
        """测试WINT4量化配置解析"""
        self.mock_args.quantization = "wint4"

        quant_config = parse_minicpm41_quant_config(self.mock_args, self.config)

        self.assertIsInstance(quant_config, WINT4Config)
        self.assertTrue(quant_config.is_quantized)
        self.assertEqual(quant_config.group_size, 128)
        self.assertFalse(quant_config.permute)

    def test_parse_wint2_quantization(self):
        """测试WINT2量化配置解析"""
        self.mock_args.quantization = "wint2"

        quant_config = parse_minicpm41_quant_config(self.mock_args, self.config)

        self.assertIsInstance(quant_config, WINT2Config)
        self.assertTrue(quant_config.is_quantized)
        self.assertEqual(quant_config.group_size, 128)

    def test_parse_dict_quantization(self):
        """测试字典形式量化配置解析"""
        args = Mock()
        args.quantization = {
            "quantization": "wint8",
            "group_size": 64,
            "is_quantized": True
        }

        quant_config = parse_minicpm41_quant_config(args, self.config)

        self.assertIsInstance(quant_config, WINT8Config)
        self.assertTrue(quant_config.is_quantized)
        self.assertEqual(quant_config.group_size, 64)

    def test_invalid_quantization_type(self):
        """测试无效量化类型"""
        self.mock_args.quantization = "invalid_type"

        with self.assertRaises(ValueError):
            parse_minicpm41_quant_config(self.mock_args, self.config)

    def test_no_quantization(self):
        """测试未启用量化"""
        self.mock_args.quantization = None

        quant_config = parse_minicpm41_quant_config(self.mock_args, self.config)

        self.assertIsNone(quant_config)


class TestMiniCPM41IntQuantCompatibility(unittest.TestCase):
    """测试MiniCPM4.1 INT量化与InfLLM-V2兼容性"""

    def setUp(self):
        """设置测试参数"""
        self.config = MiniCPM41Config(
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
        )

    def test_wint8_infllmv2_compatibility(self):
        """测试WINT8与InfLLM-V2兼容性"""
        # 启用InfLLM-V2
        self.config.sparse_config = MiniCPM41SparseConfig(enabled=True)

        mock_args = Mock()
        mock_args.quantization = "wint8"
        mock_args.group_size = 128

        # 应该成功解析并给出兼容性信息
        with patch('fastdeploy.model_executor.layers.quantization.minicpm41_quant_parser.logger') as mock_logger:
            quant_config = parse_minicpm41_quant_config(mock_args, self.config)

            self.assertIsInstance(quant_config, WINT8Config)
            # 检查是否记录了兼容性信息
            mock_logger.info.assert_called()

    def test_wint4_infllmv2_compatibility(self):
        """测试WINT4与InfLLM-V2兼容性"""
        # 启用InfLLM-V2
        self.config.sparse_config = MiniCPM41SparseConfig(enabled=True)

        mock_args = Mock()
        mock_args.quantization = "wint4"
        mock_args.group_size = 128

        with patch('fastdeploy.model_executor.layers.quantization.minicpm41_quant_parser.logger') as mock_logger:
            quant_config = parse_minicpm41_quant_config(mock_args, self.config)

            self.assertIsInstance(quant_config, WINT4Config)
            # 检查是否记录了兼容性信息
            mock_logger.info.assert_called()

    def test_wint2_infllmv2_compatibility(self):
        """测试WINT2与InfLLM-V2兼容性问题"""
        # 启用InfLLM-V2
        self.config.sparse_config = MiniCPM41SparseConfig(enabled=True)

        mock_args = Mock()
        mock_args.quantization = "wint2"
        mock_args.group_size = 128

        with patch('fastdeploy.model_executor.layers.quantization.minicpm41_quant_parser.logger') as mock_logger:
            quant_config = parse_minicpm41_quant_config(mock_args, self.config)

            self.assertIsInstance(quant_config, WINT2Config)
            # 检查是否记录了警告信息
            mock_logger.warning.assert_called()

    def test_no_infllmv2_compatibility_check(self):
        """测试未启用InfLLM-V2时的兼容性检查"""
        # 禁用InfLLM-V2
        self.config.sparse_config = MiniCPM41SparseConfig(enabled=False)

        mock_args = Mock()
        mock_args.quantization = "wint2"
        mock_args.group_size = 128

        with patch('fastdeploy.model_executor.layers.quantization.minicpm41_quant_parser.logger') as mock_logger:
            quant_config = parse_minicpm41_quant_config(mock_args, self.config)

            self.assertIsInstance(quant_config, WINT2Config)
            # 不应该有兼容性警告
            mock_logger.warning.assert_not_called()


class TestMiniCPM41QuantRecommendations(unittest.TestCase):
    """测试MiniCPM4.1量化推荐系统"""

    def setUp(self):
        """设置测试参数"""
        self.config = MiniCPM41Config(
            hidden_size=4096,
            num_hidden_layers=30,
            num_attention_heads=32,
        )

    def test_low_memory_recommendations(self):
        """测试低内存约束推荐"""
        # 启用InfLLM-V2
        self.config.sparse_config = MiniCPM41SparseConfig(enabled=True)

        recommendations = get_minicpm41_quant_recommendations(
            self.config, memory_constraint="low"
        )

        self.assertIn("wint8", recommendations)
        self.assertIn("w4a8", recommendations)

        # 检查WINT8推荐
        wint8_rec = recommendations["wint8"]
        self.assertEqual(wint8_rec["memory_reduction"], "50%")
        self.assertIn("InfLLM-V2", wint8_rec["description"])

    def test_medium_memory_recommendations(self):
        """测试中等内存约束推荐"""
        self.config.sparse_config = MiniCPM41SparseConfig(enabled=True)

        recommendations = get_minicpm41_quant_recommendations(
            self.config, memory_constraint="medium"
        )

        self.assertIn("wint4", recommendations)
        self.assertIn("w4afp8", recommendations)

    def test_high_memory_recommendations(self):
        """测试高内存约束推荐"""
        recommendations = get_minicpm41_quant_recommendations(
            self.config, memory_constraint="high"
        )

        self.assertIn("fp8", recommendations)

    def test_auto_select_optimal_quant(self):
        """测试自动量化选择"""
        # 测试内存约束选择
        selected = auto_select_optimal_quant(
            self.config, available_memory_gb=10.0  # 假设10GB可用内存
        )

        self.assertIsNotNone(selected)
        self.assertIn(selected, ["wint8", "w4afp8", "w4a8"])

        # 测试速度约束选择
        selected = auto_select_optimal_quant(
            self.config, target_speedup=2.5
        )

        self.assertEqual(selected, "w4afp8")

        # 测试精度约束选择
        selected = auto_select_optimal_quant(
            self.config, max_accuracy_loss=0.5
        )

        self.assertEqual(selected, "wint8")


class TestMiniCPM41QuantizedModel(unittest.TestCase):
    """测试量化MiniCPM4.1模型"""

    def setUp(self):
        """设置测试参数"""
        self.config = MiniCPM41Config(
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
        )

    @patch('paddle.is_compiled_with_cuda', return_value=True)
    def test_wint8_quantized_model_creation(self, mock_cuda):
        """测试WINT8量化模型创建"""
        config = MiniCPM41Config(quantization="wint8")

        with patch('fastdeploy.model_executor.layers.quantization.parse_minicpm41_quant_config') as mock_parser:
            # 模拟量化配置解析
            mock_quant_config = WINT8Config.from_config({
                "is_quantized": True,
                "group_size": 128,
            })
            mock_parser.return_value = mock_quant_config

            model = MiniCPM41ForCausalLM(config)

            # 验证量化配置被正确设置
            mock_parser.assert_called_once()

    @patch('paddle.is_compiled_with_cuda', return_value=True)
    def test_wint4_quantized_model_creation(self, mock_cuda):
        """测试WINT4量化模型创建"""
        config = MiniCPM41Config(quantization="wint4")

        with patch('fastdeploy.model_executor.layers.quantization.parse_minicpm41_quant_config') as mock_parser:
            mock_quant_config = WINT4Config.from_config({
                "is_quantized": True,
                "group_size": 128,
            })
            mock_parser.return_value = mock_quant_config

            model = MiniCPM41ForCausalLM(config)

            mock_parser.assert_called_once()

    @patch('paddle.is_compiled_with_cuda', return_value=True)
    def test_wint2_quantized_model_creation(self, mock_cuda):
        """测试WINT2量化模型创建"""
        config = MiniCPM41Config(quantization="wint2")

        with patch('fastdeploy.model_executor.layers.quantization.parse_minicpm41_quant_config') as mock_parser:
            mock_quant_config = WINT2Config.from_config({
                "is_quantized": True,
                "group_size": 128,
            })
            mock_parser.return_value = mock_quant_config

            model = MiniCPM41ForCausalLM(config)

            mock_parser.assert_called_once()

    @patch('paddle.is_compiled_with_cuda', return_value=True)
    def test_quantized_model_with_infllmv2(self, mock_cuda):
        """测试量化模型与InfLLM-V2集成"""
        config = MiniCPM41Config(
            quantization="wint8",
            sparse_config={"enabled": True, "kernel_size": 32, "topk": 64}
        )

        # 创建FDConfig与InfLLM-V2配置
        fd_config = FDConfig()
        fd_config.infllmv2_config = InfLLMV2Config({
            'infllmv2_kernel_size': 32,
            'infllmv2_topk': 64
        })
        config.fd_config = fd_config

        with patch('fastdeploy.model_executor.layers.quantization.parse_minicpm41_quant_config') as mock_parser:
            mock_quant_config = WINT8Config.from_config({
                "is_quantized": True,
                "group_size": 128,
            })
            mock_parser.return_value = mock_quant_config

            model = MiniCPM41ForCausalLM(config)

            # 验证配置正确设置
            self.assertTrue(config.sparse_config.enabled)
            self.assertIsNotNone(fd_config.infllmv2_config)


class TestMiniCPM41QuantPerformance(unittest.TestCase):
    """测试MiniCPM4.1量化性能指标"""

    def test_memory_reduction_calculation(self):
        """测试内存压缩率计算"""
        from fastdeploy.model_executor.layers.quantization.minicpm41_quant_parser import _optimize_quant_params

        config = MiniCPM41Config(
            hidden_size=4096,  # 8B模型
            num_hidden_layers=30,
        )

        # 测试WINT8压缩率
        class MockQuantConfig:
            def __init__(self, quant_type, group_size):
                self.quant_type = quant_type
                self.group_size = group_size

        quant_config = MockQuantConfig("wint8", 128)

        with patch('fastdeploy.model_executor.layers.quantization.minicpm41_quant_parser.logger') as mock_logger:
            _optimize_quant_params(quant_config, config)

            # 应该给出优化建议
            mock_logger.info.assert_called()

    def test_quant_config_validation(self):
        """测试量化配置验证"""
        from fastdeploy.model_executor.models.minicpm41_quant import MiniCPM41QuantizationConfig

        # 测试有效配置
        config = MiniCPM41QuantizationConfig(
            quant_type="wint8",
            weight_bits=8,
            activation_bits=16,
            group_size=128
        )
        config._validate_config()  # 应该不抛出异常

        # 测试无效配置
        with self.assertRaises(ValueError):
            config = MiniCPM41QuantizationConfig(
                quant_type="invalid_type",
                weight_bits=8,
                activation_bits=16,
                group_size=128
            )
            config._validate_config()

        with self.assertRaises(ValueError):
            config = MiniCPM41QuantizationConfig(
                quant_type="wint8",
                weight_bits=4,  # 无效的位数
                activation_bits=16,
                group_size=128
            )
            config._validate_config()

    def test_infllmv2_compatibility_warnings(self):
        """测试InfLLM-V2兼容性警告"""
        from fastdeploy.model_executor.models.minicpm41_quant import MiniCPM41QuantizationConfig

        # 测试WINT2兼容性警告
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = MiniCPM41QuantizationConfig(
                quant_type="wint2",
                weight_bits=2,
                activation_bits=16,
                group_size=128
            )
            config._validate_infllmv2_compatibility()

            # 应该有兼容性警告
            self.assertTrue(len(w) > 0)


class TestMiniCPM41QuantIntegration(unittest.TestCase):
    """测试MiniCPM4.1量化集成功能"""

    @patch('paddle.is_compiled_with_cuda', return_value=True)
    def test_end_to_end_wint8_quantization(self, mock_cuda):
        """测试端到端WINT8量化"""
        # 创建配置
        config = MiniCPM41Config(
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            quantization="wint8",
            sparse_config={"enabled": True}
        )

        # 模拟量化配置解析
        with patch('fastdeploy.model_executor.layers.quantization.parse_minicpm41_quant_config') as mock_parser:
            mock_quant_config = WINT8Config.from_config({
                "is_quantized": True,
                "group_size": 128,
            })
            mock_parser.return_value = mock_quant_config

            # 创建模型
            model = MiniCPM41ForCausalLM(config)

            # 验证量化设置
            self.assertTrue(hasattr(config, 'quantization'))
            self.assertEqual(config.quantization, "wint8")
            self.assertTrue(config.sparse_config.enabled)

    @patch('paddle.is_compiled_with_cuda', return_value=True)
    def test_quantized_model_forward_pass(self, mock_cuda):
        """测试量化模型前向传播"""
        config = MiniCPM41Config(
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
        )

        # 创建量化模型
        quant_config = WINT8Config.from_config({
            "is_quantized": True,
            "group_size": 128,
        })

        with patch('fastdeploy.model_executor.layers.quantization.parse_minicpm41_quant_config', return_value=quant_config):
            model = MiniCPM41ForCausalLM(config)

            # 测试前向传播
            batch_size, seq_len = 2, 8
            input_ids = paddle.randint(0, config.vocab_size, [batch_size, seq_len])

            with paddle.no_grad():
                logits, _ = model(input_ids=input_ids)

            # 验证输出形状
            expected_shape = [batch_size, seq_len, config.vocab_size]
            self.assertEqual(logits.shape, expected_shape)


if __name__ == '__main__':
    unittest.main()