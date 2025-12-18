#!/usr/bin/env python3
"""
MiniCPM For CausalLM 权重加载函数
可直接集成到 FastDeploy 的 MiniCPM 模型实现中
"""

import paddle
import re
from typing import Optional, Iterator, Dict, Any

def get_pooling_config(model_path: str, revision: str) -> Optional[dict]:
    """
    获取池化模型配置
    """
    # 这里应该实现实际的池化配置获取逻辑
    # 暂时返回 None
    return None

class MiniCPMForCausalLM(paddle.nn.Layer):
    """
    MiniCPM 模型类，包含完整的权重加载实现
    这是一个参考实现，可以直接集成到 FastDeploy 中
    """

    def __init__(self, fd_config):
        super().__init__()
        self.fd_config = fd_config

        # 这里应该实现完整的 MiniCPM 模型结构
        # 包括：嵌入层、Transformer 层、最终层归一化、LM 头等

    @paddle.no_grad()
    def load_weights(self, weights_iterator: Iterator) -> None:
        """
        Load model parameters from a given weights_iterator object.
        MiniCPM 特化版本 - 简化的加载逻辑，保持原生分离投影架构。

        Args:
            weights_iterator (Iterator): An iterator yielding (name, weight) pairs.
        """

        from fastdeploy.model_executor.utils import (
            default_weight_loader,
            process_weights_after_loading,
        )

        # MiniCPM 的分片参数映射 - 只合并 MLP，保持注意力分离
        stacked_params_mapping = [
            # MLP 投影合并（可选的性能优化）
            ("up_gate_proj", "gate_proj", "gate"),
            ("up_gate_proj", "up_proj", "up"),
            # 标准层名称映射（处理框架差异）
            ("embed_tokens.embeddings", "embed_tokens", None),
            ("lm_head.linear", "lm_head", None),
        ]

        # 获取模型参数字典
        params_dict = dict(self.named_parameters())

        # 检查是否为池化模型
        is_pooling_model = hasattr(self, "is_pooling_model") and self.is_pooling_model
        model_path = self.fd_config.model_config.model
        revision = self.fd_config.model_config.revision

        if is_pooling_model and get_pooling_config(model_path, revision):
            params_dict = {
                param_name[6:] if param_name.startswith("model.") else param_name: param
                for param_name, param in params_dict.items()
            }

        # 获取权重加载后处理函数
        process_weights_after_loading_fn = process_weights_after_loading(dict(self.named_sublayers()))

        # 主要权重加载循环
        for loaded_weight_name, loaded_weight in weights_iterator:
            loaded = False
            param = None
            model_param_name = None

            # 第一轮：处理分片参数合并（主要是 MLP）
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in loaded_weight_name:
                    continue

                # 构建目标参数名
                model_param_name = loaded_weight_name.replace(weight_name, param_name)
                if model_param_name not in params_dict:
                    continue

                # 获取模型参数和加载器
                param = params_dict[model_param_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))

                # 加载分片权重
                weight_loader(param, loaded_weight, loaded_weight_name, shard_id=shard_id)
                loaded = True
                break

            # 第二轮：直接加载其他权重（包括所有注意力投影）
            if not loaded:
                # MiniCPM 的大部分权重直接映射，不需要复杂的合并逻辑
                model_param_name = loaded_weight_name

                if model_param_name in params_dict:
                    param = params_dict[model_param_name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                    weight_loader(param, loaded_weight, loaded_weight_name)
                else:
                    # 处理可能的名称映射
                    mapped_name = self._handle_name_mapping(loaded_weight_name, params_dict)
                    if mapped_name and mapped_name in params_dict:
                        param = params_dict[mapped_name]
                        model_param_name = mapped_name
                        weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                        weight_loader(param, loaded_weight, loaded_weight_name)

            # 权重加载后处理 - 每个权重加载后立即调用
            # 这对于量化模型、KVBatchLinear 等特殊层很重要
            if model_param_name:
                # 从参数名中提取层名（移除 .weight 后缀）
                model_sublayer_name = re.sub(r"\.(weight|bias)$", "", model_param_name)
                process_weights_after_loading_fn(model_sublayer_name, param)

        # 处理嵌入层权重共享
        if hasattr(self, 'tie_word_embeddings') and self.tie_word_embeddings:
            if hasattr(self.model, 'embed_tokens') and hasattr(self, 'lm_head'):
                self.lm_head.weight = self.model.embed_tokens.weight

    def _handle_name_mapping(self, weight_name: str, params_dict: dict) -> Optional[str]:
        """
        处理可能的参数名称映射

        Args:
            weight_name: SafeTensors 中的权重名称
            params_dict: 模型参数字典

        Returns:
            映射后的参数名称，如果不需要映射则返回 None
        """

        # MiniCPM 常见的名称映射规则
        name_mappings = {
            # 处理可能的框架差异
            "embed_tokens.weight": "model.embed_tokens.weight",
            "lm_head.weight": "lm_head.weight",
            "norm.weight": "model.norm.weight",

            # 处理可能的层次结构差异
            ".weight": "",  # 暂时不移除，保持精确匹配
            ".bias": "",   # 暂时不移除，保持精确匹配
        }

        original_name = weight_name

        # 尝试各种映射
        mapped_name = weight_name
        for pattern, replacement in name_mappings.items():
            if pattern in mapped_name:
                mapped_name = mapped_name.replace(pattern, replacement)

        # 如果映射后的名称存在，返回映射结果
        if mapped_name in params_dict:
            return mapped_name

        # 如果原名称存在，返回原名称
        if original_name in params_dict:
            return original_name

        return None

    def name(self):
        """返回模型名称"""
        return "MiniCPMForCausalLM"

# 辅助函数和工具类
class MiniCPMWeightLoader:
    """
    MiniCPM 权重加载工具类
    提供额外的加载和验证功能
    """

    @staticmethod
    def validate_weight_compatibility(model, weights_iterator):
        """
        验证权重兼容性

        Args:
            model: MiniCPM 模型实例
            weights_iterator: 权重迭代器

        Returns:
            兼容性报告
        """
        params_dict = dict(model.named_parameters())
        model_params = set(params_dict.keys())

        file_weights = set()
        loaded_weights = []

        # 收集文件中的权重
        temp_iterator = list(weights_iterator)
        for name, weight in temp_iterator:
            file_weights.add(name)
            loaded_weights.append((name, weight))

        # 重新创建迭代器
        weights_iterator = iter(loaded_weights)

        missing = model_params - file_weights
        extra = file_weights - model_params

        compatibility_report = {
            "total_model_params": len(model_params),
            "total_file_weights": len(file_weights),
            "missing_weights": list(missing),
            "extra_weights": list(extra),
            "compatible": len(missing) == 0 and len(extra) <= 1,  # 允许一个额外权重（如 lm_head）
        }

        return compatibility_report, weights_iterator

    @staticmethod
    def load_with_validation(model, weights_iterator):
        """
        带验证的权重加载

        Args:
            model: MiniCPM 模型实例
            weights_iterator: 权重迭代器
        """
        # 验证兼容性
        report, weights_iterator = MiniCPMWeightLoader.validate_weight_compatibility(
            model, weights_iterator
        )

        print("📊 权重兼容性报告:")
        print(f"  模型参数数量: {report['total_model_params']}")
        print(f"  文件权重数量: {report['total_file_weights']}")
        print(f"  兼容性: {'✅ 通过' if report['compatible'] else '❌ 失败'}")

        if report['missing_weights']:
            print(f"  缺失权重: {len(report['missing_weights'])} 个")
            for weight in report['missing_weights'][:5]:  # 只显示前5个
                print(f"    - {weight}")

        if report['extra_weights']:
            print(f"  额外权重: {len(report['extra_weights'])} 个")
            for weight in report['extra_weights']:
                print(f"    - {weight}")

        # 如果兼容，执行加载
        if report['compatible']:
            model.load_weights(weights_iterator)
            print("✅ 权重加载完成")
        else:
            raise ValueError("权重不兼容，无法加载")

def create_minicpm_model_with_loading(fd_config, model_path: str):
    """
    创建带有权重加载功能的 MiniCPM 模型

    Args:
        fd_config: FastDeploy 配置
        model_path: 模型路径

    Returns:
        加载了权重的 MiniCPM 模型
    """
    from fastdeploy.model_executor.load_weight_utils import get_weight_iterator

    # 创建模型
    model = MiniCPMForCausalLM(fd_config)

    # 获取权重迭代器
    weights_iterator = get_weight_iterator(model_path)

    # 带验证的加载
    MiniCPMWeightLoader.load_with_validation(model, weights_iterator)

    return model

# 使用示例
if __name__ == "__main__":
    # 这里是一个使用示例
    print("🚀 MiniCPM 权重加载实现")
    print("=" * 40)

    print("📝 主要功能:")
    print("  1. MiniCPMForCausalLM.load_weights() - 核心加载函数")
    print("  2. MiniCPMWeightLoader - 权重加载工具类")
    print("  3. create_minicpm_model_with_loading() - 便捷创建函数")

    print("\n🎯 关键特性:")
    print("  ✅ 分离的 QKV 投影 - 简化实现")
    print("  ✅ 可选的 MLP 合并 - 性能优化")
    print("  ✅ 权重兼容性验证 - 安全加载")
    print("  ✅ 嵌入层权重共享 - 内存效率")
    print("  ✅ 名称映射处理 - 框架兼容")

    print("\n📚 集成指南:")
    print("  1. 将 MiniCPMForCausalLM 类集成到 fastdeploy/model_executor/models/minicpm.py")
    print("  2. 将辅助函数集成到相应的工具模块中")
    print("  3. 在模型注册表中注册 MiniCPM 模型")
    print("  4. 添加相关的配置和测试")

    print("\n🔗 相关文档:")
    print("  - docs/_docs/minicpm_load_weights_analysis.md")
    print("  - minicpm_load_weights_example.py")
    print("  - FastDeploy 模型集成指南")