# FastDeploy SafeTensors 权重加载完整指南

## 概述

SafeTensors 是一种安全的、快速的张量存储格式，FastDeploy 完全支持 SafeTensors 格式的模型权重加载。SafeTensors 相比传统格式具有以下优势：

- **安全性**：防止恶意代码执行
- **效率**：零拷贝加载，内存高效
- **跨平台**：支持多种深度学习框架
- **并发友好**：支持多进程并发读取

## SafeTensors 加载完整流程

### 1. 权重文件检测和识别

**文件位置**: `fastdeploy/model_executor/load_weight_utils.py:384-408`

```python
def get_all_weights_file(model_path: str):
    """
    检测模型权重文件格式
    """
    model_path = Path(model_path)
    use_safetensors = True

    # 检查是否有 .pdparams 文件（PaddlePaddle 格式）
    if any(model_path.glob("*.pdparams")):
        key_name_list = []
        files_list = [str(file) for file in model_path.glob("*.pdparams")]
        use_safetensors = False

    # 检查单个 safetensors 文件
    elif (model_path / "model.safetensors").exists():
        safe_model_path = model_path / "model.safetensors"
        files_list = [str(safe_model_path)]
        with safe_open(safe_model_path, framework="np", device="cpu") as f:
            key_name_list = f.keys()
        return key_name_list, files_list, use_safetensors

    # 检查分片 safetensors 文件
    else:
        index_file = model_path / "model.safetensors.index.json"
        with index_file.open("r") as f:
            weight_map = json.load(f)["weight_map"]  # 你提供的这个映射
        weight_files_in_index = {str(model_path / weight_map[name]) for name in weight_map}
        key_name_list = list(weight_map.keys())
        files_list = sorted(weight_files_in_index)

    return key_name_list, files_list, use_safetensors
```

### 2. 权重迭代器创建

**文件位置**: `fastdeploy/model_executor/load_weight_utils.py:70-76`

```python
def get_weight_iterator(model_path: str):
    """
    根据文件格式创建相应的权重迭代器
    """
    _, files_list, use_safetensors = get_all_weights_file(model_path)

    if use_safetensors:
        weights_iterator = fast_weights_iterator(files_list)
    else:
        weights_iterator = pdparams_weight_iterator(files_list)

    return weights_iterator
```

### 3. SafeTensors 迭代器实现

**文件位置**: `fastdeploy/model_executor/load_weight_utils.py:319-330`

```python
def fast_weights_iterator(safe_tensor_list: list[str]):
    """
    SafeTensors 文件的迭代器实现
    """
    for st_file in tqdm(
        safe_tensor_list,
        desc="Loading safetensors checkpoint shards",
    ):
        # 使用 fast_safe_open 进行高效加载
        with fast_safe_open(st_file, framework="np") as f:
            for name in f.keys():
                param_slice = f.get_slice(name)
                yield name, param_slice  # 返回 (权重名, 权重张量) 对
```

### 4. 模型权重加载

**文件位置**: `fastdeploy/model_executor/models/qwen3.py:297-338`

```python
def load_weights(self, weights_iterator) -> None:
    """
    从 weights_iterator 加载模型权重
    """

    # 1. 分片参数映射（用于合并分离的权重）
    stacked_params_mapping = [
        ("qkv_proj", "q_proj", "q"),      # 合并 QKV 投影
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("up_gate_proj", "gate_proj", "gate"),  # 合并门控投影
        ("up_gate_proj", "up_proj", "up"),
        ("embed_tokens.embeddings", "embed_tokens", None),
        ("lm_head.linear", "lm_head", None),
    ]

    # 2. 获取模型参数字典
    params_dict = dict(self.named_parameters())

    # 3. 遍历加载每个权重
    for loaded_weight_name, loaded_weight in weights_iterator:
        loaded = False

        # 4. 处理分片参数合并
        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in loaded_weight_name:
                continue

            model_param_name = loaded_weight_name.replace(weight_name, param_name)
            if model_param_name not in params_dict:
                continue

            param = params_dict[model_param_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)

            # 加载分片权重
            weight_loader(param, loaded_weight, loaded_weight_name, shard_id=shard_id)
            loaded = True
            break

        # 5. 直接加载非分片权重
        if not loaded and loaded_weight_name in params_dict:
            param = params_dict[loaded_weight_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight, loaded_weight_name)
```

## SafeTensors 文件结构分析

### 1. 单文件模式
```
model/
├── model.safetensors          # 单个 safetensors 文件
├── config.json
└── tokenizer.json
```

### 2. 分片文件模式（你提到的模式）
```
model/
├── model.safetensors.index.json    # 索引文件
├── model-00001-of-00004.safetensors # 权重分片 1
├── model-00002-of-00004.safetensors # 权重分片 2
├── model-00003-of-00004.safetensors # 权重分片 3
├── model-00004-of-00004.safetensors # 权重分片 4
├── config.json
└── tokenizer.json
```

### 3. 索引文件结构 (`model.safetensors.index.json`)
```json
{
  "metadata": {
    "total_parameters": 8185253888,
    "total_size": 16370507776,
    "format": "pt"
  },
  "weight_map": {
    "lm_head.weight": "model-00003-of-00004.safetensors",
    "model.embed_tokens.weight": "model-00002-of-00004.safetensors",
    "model.layers.0.input_layernorm.weight": "model-00004-of-00004.safetensors",
    "model.layers.0.mlp.down_proj.weight": "model-00003-of-00004.safetensors",
    // ... 更多权重映射
  }
}
```

## 关键技术特性

### 1. 零拷贝加载
```python
# SafeTensors 使用内存映射实现零拷贝
with fast_safe_open(st_file, framework="np") as f:
    param_slice = f.get_slice(name)  # 返回视图，不复制数据
```

### 2. 惰性加载
```python
# 权重按需加载，节省内存
for name, param_slice in weights_iterator:
    # 只有在需要时才加载具体的权重
    param.copy_(param_slice, False)
```

### 3. 多进程安全
```python
# SafeTensors 支持多进程并发读取
def fastsafetensors_weights_iterator(safetensor_list):
    # 支持分布式加载
    world_size = dist.get_world_size()
    # ... 分布式加载逻辑
```

## 使用示例

### 1. 基本加载
```python
from fastdeploy.model_executor.load_weight_utils import get_weight_iterator

# 创建权重迭代器
weights_iterator = get_weight_iterator("/path/to/model")

# 模型自动加载
model.load_weights(weights_iterator)
```

### 2. 自定义加载逻辑
```python
def custom_load_weights(model, model_path):
    weights_iterator = get_weight_iterator(model_path)

    for name, weight in weights_iterator:
        if name in model.state_dict():
            print(f"Loading weight: {name}, shape: {weight.shape}")
            model.state_dict()[name].copy_(weight, False)
```

## 性能优化特性

### 1. 内存优化
- **内存映射**: 直接从文件系统映射到内存
- **惰性加载**: 只加载需要的权重
- **共享内存**: 多进程可以共享同一份权重数据

### 2. 加载速度优化
- **并行读取**: 支持多线程/多进程并行加载
- **预取机制**: 提前预取可能需要的权重
- **缓存机制**: 支持权重缓存加速重复加载

### 3. 存储优化
- **压缩**: 支持 LZ4/ZSTD 压缩
- **分片**: 大模型自动分片存储
- **校验**: 内置 CRC32 校验确保数据完整性

## 调试和监控

### 1. 权重加载监控
```python
from tqdm import tqdm

for name, weight in tqdm(weights_iterator, desc="Loading weights"):
    # 加载进度监控
    pass
```

### 2. 内存使用监控
```python
import psutil

def monitor_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
```

## 常见问题和解决方案

### 1. 文件损坏
```bash
# 验证 safetensors 文件完整性
python -c "
from safetensors import safe_open
with safe_open('model.safetensors', framework='np') as f:
    print('File is valid, keys:', list(f.keys()))
"
```

### 2. 内存不足
```python
# 使用流式加载减少内存占用
def streaming_load(weights_iterator):
    for name, weight in weights_iterator:
        yield name, weight  # 逐个处理，避免全量加载
```

### 3. 权重不匹配
```python
# 检查权重名称匹配
def check_weight_compatibility(model, weights_iterator):
    model_params = set(model.state_dict().keys())
    file_weights = set()

    for name, _ in weights_iterator:
        file_weights.add(name)

    missing = model_params - file_weights
    extra = file_weights - model_params

    if missing:
        print(f"Missing weights: {missing}")
    if extra:
        print(f"Extra weights: {extra}")
```

## 总结

FastDeploy 的 SafeTensors 加载机制提供了：

1. **高性能**: 零拷贝、内存映射、并行加载
2. **安全性**: 防止恶意代码执行
3. **兼容性**: 支持单文件和分片文件格式
4. **灵活性**: 支持自定义加载逻辑和优化策略

对于你提到的分片 SafeTensors 文件，FastDeploy 会自动读取 `model.safetensors.index.json` 索引文件，然后按需加载各个分片中的权重，实现高效、安全的模型权重加载。