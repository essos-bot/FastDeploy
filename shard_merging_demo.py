#!/usr/bin/env python3
"""
Qwen3 权重分片合并演示
可视化展示 load_weights() 中的分片合并逻辑
"""

import numpy as np
import paddle

def demonstrate_qkv_merging():
    """
    演示 QKV 投影合并过程
    """

    print("🔍 QKV 投影合并演示")
    print("=" * 50)

    # 模拟原始分离的权重
    hidden_size = 4096
    print(f"📊 模型配置: hidden_size = {hidden_size}")

    # 创建模拟权重
    q_weight = np.random.randn(hidden_size, hidden_size).astype(np.float32)
    k_weight = np.random.randn(hidden_size, hidden_size).astype(np.float32)
    v_weight = np.random.randn(hidden_size, hidden_size).astype(np.float32)

    print(f"\n📦 原始分离权重:")
    print(f"  q_proj.weight: {q_weight.shape}")
    print(f"  k_proj.weight: {k_weight.shape}")
    print(f"  v_proj.weight: {v_weight.shape}")

    # 模拟分片合并过程
    print(f"\n🔄 分片合并过程:")

    # 第一步：加载 q_proj
    print(f"  步骤 1: 加载 q_proj.weight")
    print(f"    -> 目标: qkv_proj.weight 的第 0-{hidden_size-1} 行")
    qkv_proj = np.zeros((3 * hidden_size, hidden_size), dtype=np.float32)
    qkv_proj[0:hidden_size, :] = q_weight
    print(f"    -> 当前进度: {qkv_proj.shape[0]}/{3*hidden_size} 行已加载")

    # 第二步：加载 k_proj
    print(f"  步骤 2: 加载 k_proj.weight")
    print(f"    -> 目标: qkv_proj.weight 的第 {hidden_size}-{2*hidden_size-1} 行")
    qkv_proj[hidden_size:2*hidden_size, :] = k_weight
    print(f"    -> 当前进度: {qkv_proj.shape[0]}/{3*hidden_size} 行已加载")

    # 第三步：加载 v_proj
    print(f"  步骤 3: 加载 v_proj.weight")
    print(f"    -> 目标: qkv_proj.weight 的第 {2*hidden_size}-{3*hidden_size-1} 行")
    qkv_proj[2*hidden_size:3*hidden_size, :] = v_weight
    print(f"    -> 当前进度: {qkv_proj.shape[0]}/{3*hidden_size} 行已加载")

    print(f"\n✅ 合并完成:")
    print(f"  qkv_proj.weight: {qkv_proj.shape}")

    # 验证合并正确性
    expected_shape = (3 * hidden_size, hidden_size)
    assert qkv_proj.shape == expected_shape, f"形状不匹配: 期望 {expected_shape}, 实际 {qkv_proj.shape}"
    print(f"  ✅ 形状验证通过")

    return qkv_proj, q_weight, k_weight, v_weight

def demonstrate_mlp_merging():
    """
    演示 MLP 门控投影合并过程
    """

    print("\n🔍 MLP 门控投影合并演示")
    print("=" * 50)

    # 模拟配置
    hidden_size = 4096
    intermediate_size = 22016  # 通常是 hidden_size * 4 * 1.34
    print(f"📊 模型配置:")
    print(f"  hidden_size = {hidden_size}")
    print(f"  intermediate_size = {intermediate_size}")

    # 创建模拟权重
    gate_weight = np.random.randn(intermediate_size, hidden_size).astype(np.float32)
    up_weight = np.random.randn(intermediate_size, hidden_size).astype(np.float32)

    print(f"\n📦 原始分离权重:")
    print(f"  gate_proj.weight: {gate_weight.shape}")
    print(f"  up_proj.weight: {up_weight.shape}")

    # 模拟合并过程
    print(f"\n🔄 分片合并过程:")

    # 第一步：加载 gate_proj
    print(f"  步骤 1: 加载 gate_proj.weight")
    print(f"    -> 目标: up_gate_proj.weight 的第 0-{intermediate_size-1} 行")
    up_gate_proj = np.zeros((2 * intermediate_size, hidden_size), dtype=np.float32)
    up_gate_proj[0:intermediate_size, :] = gate_weight
    print(f"    -> 当前进度: {up_gate_proj.shape[0]}/{2*intermediate_size} 行已加载")

    # 第二步：加载 up_proj
    print(f"  步骤 2: 加载 up_proj.weight")
    print(f"    -> 目标: up_gate_proj.weight 的第 {intermediate_size}-{2*intermediate_size-1} 行")
    up_gate_proj[intermediate_size:2*intermediate_size, :] = up_weight
    print(f"    -> 当前进度: {up_gate_proj.shape[0]}/{2*intermediate_size} 行已加载")

    print(f"\n✅ 合并完成:")
    print(f"  up_gate_proj.weight: {up_gate_proj.shape}")

    # 验证合并正确性
    expected_shape = (2 * intermediate_size, hidden_size)
    assert up_gate_proj.shape == expected_shape, f"形状不匹配: 期望 {expected_shape}, 实际 {up_gate_proj.shape}"
    print(f"  ✅ 形状验证通过")

    return up_gate_proj, gate_weight, up_weight

def demonstrate_name_mapping():
    """
    演示参数名称映射
    """

    print("\n🔍 参数名称映射演示")
    print("=" * 30)

    # 模拟 stacked_params_mapping
    stacked_params_mapping = [
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("up_gate_proj", "gate_proj", "gate"),
        ("up_gate_proj", "up_proj", "up"),
        ("embed_tokens.embeddings", "embed_tokens", None),
        ("lm_head.linear", "lm_head", None),
    ]

    # 模拟 SafeTensors 中的权重名称
    file_weights = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.embed_tokens.weight",  # 注意：这里是 embed_tokens
        "lm_head.weight",           # 注意：这里是 lm_head
    ]

    print("📋 名称映射过程:")

    for file_weight in file_weights:
        print(f"\n  原始名称: {file_weight}")

        # 查找匹配的映射
        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name in file_weight:
                model_param_name = file_weight.replace(weight_name, param_name)
                print(f"    映射到: {model_param_name}")
                print(f"    分片ID: {shard_id}")
                break
        else:
            print(f"    直接使用: {file_weight}")

def simulate_load_weights_flow():
    """
    模拟完整的 load_weights 流程
    """

    print("\n🔍 完整加载流程模拟")
    print("=" * 30)

    # 模拟 weights_iterator 的输出
    simulated_weights = [
        ("model.layers.0.self_attn.q_proj.weight", "q_weight_tensor"),
        ("model.layers.0.self_attn.k_proj.weight", "k_weight_tensor"),
        ("model.layers.0.self_attn.v_proj.weight", "v_weight_tensor"),
        ("model.layers.0.mlp.gate_proj.weight", "gate_weight_tensor"),
        ("model.layers.0.mlp.up_proj.weight", "up_weight_tensor"),
        ("model.layers.0.input_layernorm.weight", "layernorm_weight_tensor"),
        ("model.embed_tokens.weight", "embed_weight_tensor"),
    ]

    # 模拟 stacked_params_mapping
    stacked_params_mapping = [
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("up_gate_proj", "gate_proj", "gate"),
        ("up_gate_proj", "up_proj", "up"),
    ]

    print("📦 模拟权重加载过程:")

    for loaded_weight_name, loaded_weight in simulated_weights:
        print(f"\n  处理权重: {loaded_weight_name}")
        loaded = False

        # 第一轮：检查分片映射
        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name in loaded_weight_name:
                model_param_name = loaded_weight_name.replace(weight_name, param_name)
                print(f"    ✅ 分片合并: {loaded_weight_name} → {model_param_name}")
                print(f"       分片ID: {shard_id}")
                loaded = True
                break

        # 第二轮：直接加载
        if not loaded:
            print(f"    ✅ 直接加载: {loaded_weight_name}")

def main():
    """
    主演示函数
    """

    print("🚀 Qwen3 权重加载分片合并完整演示")
    print("=" * 60)

    # 演示 QKV 合并
    qkv_weight, q, k, v = demonstrate_qkv_merging()

    # 演示 MLP 合并
    mlp_weight, gate, up = demonstrate_mlp_merging()

    # 演示名称映射
    demonstrate_name_mapping()

    # 模拟完整流程
    simulate_load_weights_flow()

    print("\n📊 性能优势总结")
    print("=" * 20)
    print("✅ 内存效率: 避免存储重复的投影矩阵")
    print("✅ 计算效率: 单次矩阵运算替代三次运算")
    print("✅ 缓存友好: 连续内存布局提高访问效率")
    print("✅ 量化友好: 更大的矩阵便于量化优化")

    print("\n🎯 关键技术点")
    print("=" * 15)
    print("1. 分片映射: stacked_params_mapping 定义合并规则")
    print("2. 名称替换: .replace() 实现参数名映射")
    print("3. 分片加载: 根据 shard_id 确定加载位置")
    print("4. 直接拷贝: copy_() 实现高效数据转移")

    print("\n📚 相关文档")
    print("=" * 15)
    print("- 详细分析: docs/_docs/qwen3_load_weights_analysis.md")
    print("- SafeTensors: docs/_docs/safetensors_loading_guide.md")
    print("- 源码: fastdeploy/model_executor/models/qwen3.py:259")

if __name__ == "__main__":
    main()