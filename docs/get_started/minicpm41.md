# MiniCPM4.1-8B 部署指南

MiniCPM4.1-8B 是OpenBMB推出的高性能大语言模型，FastDeploy现已全面支持其部署，包括多种量化选项和优化特性。

## 模型特点

- **参数规模**: 8B参数的高性能语言模型
- **长上下文**: 支持64K+上下文长度
- **稀疏注意力**: 可训练稀疏注意力机制 (InfLLM v2)
- **混合推理**: 支持深度推理和快速推理模式切换
- **量化支持**: 完整的量化方案 (W4AFP8, W4A8, W8A16等)
- **高性能**: 在端侧设备上实现3倍推理速度提升

## 快速开始

### 基础部署

```python
from fastdeploy import LLM

# 创建MiniCPM4.1-8B实例
llm = LLM(model="openbmb/MiniCPM4.1-8B")

# 生成文本
output = llm.generate("请介绍一下人工智能的发展历程")
print(output)
```

### 使用本地模型

```python
from fastdeploy import LLM

# 使用本地转换后的模型
llm = LLM(model="/path/to/minicpm4.1-8b-ppd")

# 生成对话
messages = [
    {"role": "user", "content": "你好，请介绍一下MiniCPM4.1-8B模型"}
]
output = llm.chat(messages)
print(output)
```

## 权重转换

如果需要从HuggingFace转换权重到PaddlePaddle格式：

```bash
# 转换权重
python tools/convert_minicpm41_weights.py \
    --input_dir /path/to/openbmb/MiniCPM4.1-8B \
    --output_dir /path/to/minicpm4.1-8b-ppd \
    --model_name minicpm4.1-8b-ppd \
    --dtype float16
```

## 量化部署

### W4AFP8 量化 (推荐)

```python
from fastdeploy import LLM
from fastdeploy.config.minicpm41_config import create_quantized_minicpm41_config

# 创建量化配置
fd_config, quant_config = create_quantized_minicpm41_config(
    quant_type="w4afp8",
    group_size=128,
    is_permuted=True,
)

# 创建量化模型
llm = LLM(model="openbmb/MiniCPM4.1-8B", fd_config=fd_config)

# 获取量化信息
quant_info = llm.model.get_quantization_info()
print(f"量化类型: {quant_info['quant_type']}")
print(f"内存压缩比: {quant_info['memory_reduction_ratio']:.2f}x")
```

### W8A16 量化

```python
from fastdeploy import LLM

# 权重8位，激活16位量化
llm = LLM(
    model="openbmb/MiniCPM4.1-8B",
    quantization="wint8"
)
```

### W4A8 量化

```python
from fastdeploy import LLM

# 权重4位，激活8位量化
llm = LLM(
    model="openbmb/MiniCPM4.1-8B",
    quantization="w4a8"
)
```

## 配置选项

### 基础配置

```python
from fastdeploy import FDConfig, LLM
from fastdeploy.config.minicpm41_config import MiniCPM41ModelConfig

# 创建自定义配置
model_config = MiniCPM41ModelConfig(
    max_position_embeddings=32768,  # 最大上下文长度
    enable_thinking=True,           # 启用混合推理模式
    temperature=0.7,               # 生成温度
    top_p=0.9,                    # Top-p采样
    max_new_tokens=2048,          # 最大生成长度
)

# 创建FDConfig
fd_config = FDConfig(
    model_config=model_config,
    tensor_parallel_size=4,       # 张量并行度
)

# 创建LLM实例
llm = LLM(model="openbmb/MiniCPM4.1-8B", fd_config=fd_config)
```

### 稀疏注意力配置

```python
from fastdeploy.config.minicpm41_config import SparseAttentionConfig

# 配置稀疏注意力
sparse_config = SparseAttentionConfig(
    kernel_size=32,           # 语义核大小
    kernel_stride=16,         # 核间步长
    init_blocks=1,           # 初始块数
    block_size=64,           # KV块大小
    window_size=2048,        # 局部滑窗大小
    topk=64,                 # 前k相关块
    dense_len=8192,         # 密集注意力长度阈值
)

model_config = MiniCPM41ModelConfig(
    sparse_config=sparse_config.to_dict()
)
```

## 混合推理模式

MiniCPM4.1-8B支持独特的混合推理模式：

```python
# 启用推理模式 (深度思考)
messages = [
    {"role": "system", "content": "你是一个深度思考的AI助手"},
    {"role": "user", "content": "/think 请分析人工智能的未来发展趋势"}
]
output = llm.chat(messages, enable_thinking=True)

# 启用非推理模式 (快速响应)
messages = [
    {"role": "user", "content": "/no_think 什么是机器学习？"}
]
output = llm.chat(messages, enable_thinking=False)
```

## 高级部署

### API服务器部署

```bash
# 启动OpenAI兼容的API服务器
python -m fastdeploy.entrypoints.openai.api_server \
    --model openbmb/MiniCPM4.1-8B \
    --port 8180 \
    --host 0.0.0.0 \
    --tensor-parallel-size 4 \
    --max-model-len 65536

# 启动量化API服务器
python -m fastdeploy.entrypoints.openai.api_server \
    --model openbmb/MiniCPM4.1-8B \
    --quantization w4afp8 \
    --port 8180 \
    --tensor-parallel-size 4
```

### 客户端调用

```python
import requests

# OpenAI兼容API调用
response = requests.post(
    "http://localhost:8180/v1/chat/completions",
    json={
        "model": "openbmb/MiniCPM4.1-8B",
        "messages": [
            {"role": "user", "content": "请介绍一下FastDeploy"}
        ],
        "max_tokens": 512,
        "temperature": 0.7
    }
)

result = response.json()
print(result["choices"][0]["message"]["content"])
```

### 批量推理

```python
from fastdeploy import LLM
from fastdeploy import SamplingParams

# 创建LLM实例
llm = LLM(model="openbmb/MiniCPM4.1-8B")

# 批量生成
prompts = [
    "请介绍一下机器学习",
    "什么是深度学习？",
    "解释一下神经网络的工作原理"
]

# 创建采样参数
sampling_params = SamplingParams(
    max_tokens=256,
    temperature=0.7,
    top_p=0.9,
)

# 批量推理
outputs = llm.generate(prompts, sampling_params=sampling_params)

for i, output in enumerate(outputs):
    print(f"Prompt {i+1}: {prompts[i]}")
    print(f"Output {i+1}: {output}")
    print("-" * 50)
```

## 性能优化

### 内存优化

```python
from fastdeploy import FDConfig
from fastdeploy.config.minicpm41_config import get_optimized_cache_config

# 根据序列长度优化缓存配置
seq_len = 16384  # 示例序列长度
cache_config = get_optimized_cache_config(seq_len, model_config)

fd_config = FDConfig(
    model_config=model_config,
    cache_config=cache_config,
    # 启用内存优化
    memory_efficient=True,
)
```

### 硬件优化

```bash
# GPU部署优化
export CUDA_VISIBLE_DEVICES=0,1,2,3
export FD_USE_FLASH_ATTN=1
export FD_MEMORY_OPTIMIZATION=1

# 启动优化后的服务器
python -m fastdeploy.entrypoints.openai.api_server \
    --model openbmb/MiniCPM4.1-8B \
    --quantization w4afp8 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --max-num-batched-tokens 8192
```

## 测试和验证

### 功能测试

```python
# 基础功能测试
def test_basic_functionality():
    llm = LLM(model="openbmb/MiniCPM4.1-8B")

    # 测试文本生成
    output = llm.generate("你好", max_tokens=50)
    assert len(output) > 0

    # 测试对话功能
    messages = [{"role": "user", "content": "你好"}]
    output = llm.chat(messages)
    assert len(output) > 0

    print("✓ 基础功能测试通过")

test_basic_functionality()
```

### 性能测试

```python
import time
from fastdeploy import LLM

def benchmark_performance():
    llm = LLM(model="openbmb/MiniCPM4.1-8B")

    # 准备测试数据
    test_prompts = ["请详细介绍一下人工智能的发展历程"] * 10

    # 测试延迟
    start_time = time.time()
    outputs = llm.generate(test_prompts, max_tokens=256)
    end_time = time.time()

    # 计算性能指标
    total_time = end_time - start_time
    avg_latency = total_time / len(test_prompts)
    throughput = len(test_prompts) / total_time

    print(f"平均延迟: {avg_latency:.2f}秒")
    print(f"吞吐量: {throughput:.2f} 请求/秒")

benchmark_performance()
```

## 常见问题

### Q1: 如何选择合适的量化方案？

**A**:
- **W4AFP8**: 推荐用于生产环境，平衡精度和性能
- **W8A16**: 适合精度要求高的场景
- **W4A8**: 适合内存受限的场景
- **FP8**: 适合支持FP8的硬件

### Q2: 如何处理长文本？

**A**: MiniCPM4.1-8B支持多种长文本处理策略：

```python
# 针对不同长度优化配置
def optimize_for_length(seq_len):
    if seq_len <= 4096:
        return "normal"
    elif seq_len <= 16384:
        return "prefix_cache"
    else:
        return "sparse_attention"
```

### Q3: 如何调试性能问题？

**A**:
1. 检查量化配置是否正确
2. 监控GPU内存使用情况
3. 调整batch size和序列长度
4. 使用性能分析工具

## 版本兼容性

| FastDeploy版本 | MiniCPM4.1-8B支持 | 备注 |
|---------------|------------------|------|
| 1.0.0+        | ✓ 完整支持        | 推荐版本 |
| 0.9.x         | ✓ 基础支持        | 部分功能受限 |

## 更多资源

- [MiniCPM官方仓库](https://github.com/OpenBMB/MiniCPM)
- [FastDeploy GitHub](https://github.com/PaddlePaddle/FastDeploy)
- [性能基准测试报告](../benchmarks/minicpm41_performance.md)
- [量化详细说明](../quantization/overview.md)

---

如需更多帮助，请访问 [FastDeploy文档](https://github.com/PaddlePaddle/FastDeploy) 或提交 [Issue](https://github.com/PaddlePaddle/FastDeploy/issues)。