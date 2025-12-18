#!/usr/bin/env python3
"""
SafeTensors 加载示例
演示 FastDeploy 如何加载和使用 safetensors 文件
"""

import json
from pathlib import Path

def demonstrate_safetensors_loading():
    """
    演示 SafeTensors 加载流程
    """

    print("🔍 SafeTensors 加载流程演示")
    print("=" * 50)

    # 1. 模拟模型目录结构
    model_dir = Path("/path/to/your/model")

    print("\n📁 1. 检测权重文件格式:")
    print("   - 检查 model.safetensors.index.json")
    print("   - 读取 weight_map 映射关系")

    # 2. 读取索引文件（模拟你提供的结构）
    simulated_index = {
        "metadata": {
            "total_parameters": 8185253888,
            "total_size": 16370507776
        },
        "weight_map": {
            "lm_head.weight": "model-00003-of-00004.safetensors",
            "model.embed_tokens.weight": "model-00002-of-00004.safetensors",
            "model.layers.0.input_layernorm.weight": "model-00004-of-00004.safetensors",
            "model.layers.0.mlp.down_proj.weight": "model-00003-of-00004.safetensors",
            "model.layers.0.mlp.gate_proj.weight": "model-00004-of-00004.safetensors",
            "model.layers.0.mlp.up_proj.weight": "model-00001-of-00004.safetensors",
            "model.layers.0.post_attention_layernorm.weight": "model-00002-of-00004.safetensors",
            "model.layers.0.self_attn.k_proj.weight": "model-00003-of-00004.safetensors",
        }
    }

    print(f"\n📊 2. 模型信息:")
    print(f"   - 总参数量: {simulated_index['metadata']['total_parameters']:,}")
    print(f"   - 总大小: {simulated_index['metadata']['total_size'] / 1024**3:.2f} GB")

    print(f"\n📋 3. 权重文件分布:")
    # 统计每个文件包含的权重数量
    file_stats = {}
    for weight_name, file_name in simulated_index["weight_map"].items():
        if file_name not in file_stats:
            file_stats[file_name] = []
        file_stats[file_name].append(weight_name)

    for file_name, weights in file_stats.items():
        print(f"   - {file_name}: {len(weights)} 个权重")

    print(f"\n🔄 4. 加载流程:")

    # 3. 模拟权重加载流程
    print("   步骤 1: 创建权重迭代器")
    print("   ```python")
    print("   weights_iterator = get_weight_iterator(model_path)")
    print("   ```")

    print("\n   步骤 2: 模型加载权重")
    print("   ```python")
    print("   for loaded_weight_name, loaded_weight in weights_iterator:")
    print("       # 处理分片合并")
    print("       if 'q_proj' in loaded_weight_name:")
    print("           # 合并到 qkv_proj")
    print("       elif 'k_proj' in loaded_weight_name:")
    print("           # 合并到 qkv_proj")
    print("       elif 'v_proj' in loaded_weight_name:")
    print("           # 合并到 qkv_proj")
    print("       else:")
    print("           # 直接加载")
    print("           param.copy_(loaded_weight, False)")
    print("   ```")

def show_safetensors_advantages():
    """
    展示 SafeTensors 的优势
    """

    print("\n🚀 SafeTensors 优势")
    print("=" * 30)

    advantages = [
        {
            "特性": "安全性",
            "说明": "防止恶意代码执行，只包含纯张量数据"
        },
        {
            "特性": "零拷贝加载",
            "说明": "使用内存映射，直接从文件系统读取，不复制数据"
        },
        {
            "特性": "内存高效",
            "说明": "惰性加载，只加载需要的权重张量"
        },
        {
            "特性": "并行友好",
            "说明": "支持多进程并发读取同一文件"
        },
        {
            "特性": "跨平台",
            "说明": "支持 PyTorch、TensorFlow、PaddlePaddle 等框架"
        },
        {
            "特性": "压缩支持",
            "说明": "内置 LZ4/ZSTD 压缩，减少存储空间"
        }
    ]

    for advantage in advantages:
        print(f"\n✅ {advantage['特性']}:")
        print(f"   {advantage['说明']}")

def show_code_examples():
    """
    显示实用的代码示例
    """

    print("\n💡 实用代码示例")
    print("=" * 30)

    # 示例 1: 基本加载
    print("\n📝 示例 1: 基本权重加载")
    print("```python")
    print("from fastdeploy.model_executor.load_weight_utils import get_weight_iterator")
    print("")
    print("# 创建权重迭代器")
    print("weights_iterator = get_weight_iterator('/path/to/model')")
    print("")
    print("# 模型自动加载")
    print("model.load_weights(weights_iterator)")
    print("```")

    # 示例 2: 检查权重分布
    print("\n📝 示例 2: 检查权重分布")
    print("```python")
    print("import json")
    print("from pathlib import Path")
    print("")
    print("def analyze_weights(model_path):")
    print("    model_path = Path(model_path)")
    print("    index_file = model_path / 'model.safetensors.index.json'")
    print("    ")
    print("    with index_file.open('r') as f:")
    print("        index_data = json.load(f)")
    print("    ")
    print("    # 统计每个文件的权重数量")
    print("    file_stats = {}")
    print("    for weight_name, file_name in index_data['weight_map'].items():")
    print("        file_stats.setdefault(file_name, 0)")
    print("        file_stats[file_name] += 1")
    print("    ")
    print("    print('权重文件分布:')")
    print("    for file_name, count in file_stats.items():")
    print("        print(f'  {file_name}: {count} 个权重')")
    print("```")

    # 示例 3: 自定义加载
    print("\n📝 示例 3: 自定义加载逻辑")
    print("```python")
    print("from fastdeploy.model_executor.load_weight_utils import fast_weights_iterator")
    print("")
    print("def custom_load(model_path, target_weights=None):")
    print("    # 获取所有 safetensors 文件")
    print("    _, files_list, _ = get_all_weights_file(model_path)")
    print("    ")
    print("    # 创建迭代器")
    print("    weights_iter = fast_weights_iterator(files_list)")
    print("    ")
    print("    loaded_weights = {}")
    print("    for name, weight in weights_iter:")
    print("        # 只加载指定的权重")
    print("        if target_weights is None or name in target_weights:")
    print("            loaded_weights[name] = weight")
    print("            print(f'已加载: {name}, shape: {weight.shape}')")
    print("    ")
    print("    return loaded_weights")
    print("```")

def main():
    """
    主函数
    """
    demonstrate_safetensors_loading()
    show_safetensors_advantages()
    show_code_examples()

    print("\n📚 更多信息")
    print("=" * 20)
    print("- 完整文档: docs/_docs/safetensors_loading_guide.md")
    print("- 源码: fastdeploy/model_executor/load_weight_utils.py")
    print("- 模型实现: fastdeploy/model_executor/models/qwen3.py")

if __name__ == "__main__":
    main()