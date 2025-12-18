# MiniCPM 权重加载函数设计与实现

## 概述

基于对 MiniCPM 模型结构的深入分析，设计了一个简化但高效的权重加载函数，专门针对 MiniCPM 的架构特点进行优化。

## MiniCPM 与 Qwen3 的关键差异

| 特性 | MiniCPM | Qwen3 | 对加载逻辑的影响 |
|------|---------|-------|------------------|
| **QKV 投影** | **分离的** (q_proj, k_proj, v_proj) | **合并的** (qkv_proj) | MiniCPM: 简单映射，Qwen3: 复杂合并 |
| **MLP 投影** | 分离的 (gate_proj, up_proj) | 合并的 (up_gate_proj) | 两者都可以合并优化 |
| **查询/键归一化** | 无 | 有 q_norm, k_norm | MiniCPM: 参数更少 |
| **权重加载方式** | **直接映射** | **分片映射** | MiniCPM: 实现更简单 |

## MiniCPM 模型架构分析

### 1. 注意力层结构
```python
class MiniCPMAttention(nn.Module):
    def __init__(self, config):
        # MiniCPM 使用分离的投影
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=config.attention_bias)
```

### 2. MLP 层结构
```python
class MiniCPMMLP(nn.Module):
    def __init__(self, config):
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
```

### 3. 完整参数命名
```
model.embed_tokens.weight                          # 词嵌入
model.layers.{i}.self_attn.q_proj.weight           # 查询投影
model.layers.{i}.self_attn.k_proj.weight           # 键投影
model.layers.{i}.self_attn.v_proj.weight           # 值投影
model.layers.{i}.self_attn.o_proj.weight           # 输出投影
model.layers.{i}.mlp.gate_proj.weight              # MLP 门控
model.layers.{i}.mlp.up_proj.weight                 # MLP 上投影
model.layers.{i}.mlp.down_proj.weight               # MLP 下投影
model.layers.{i}.input_layernorm.weight            # 输入层归一化
model.layers.{i}.post_attention_layernorm.weight    # 注意力后归一化
model.norm.weight                                   # 最终层归一化
lm_head.weight                                     # LM 头 (与嵌入共享)
```

## MiniCPM 权重加载函数实现

### 核心加载函数

```python
@paddle.no_grad()
def load_weights(self, weights_iterator) -> None:
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
                    weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                    weight_loader(param, loaded_weight, loaded_weight_name)

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

    # 常见的名称映射规则
    name_mappings = {
        # 处理可能的框架差异
        "embed_tokens.weight": "model.embed_tokens.weight",
        "lm_head.weight": "lm_head.weight",
        "norm.weight": "model.norm.weight",

        # 处理可能的层次结构差异
        "weight": "",  # 移除 .weight 后缀
        ".bias": "",   # 移除 .bias 后缀
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
```

## 加载流程详细分析

### 1. 参数映射策略

#### MiniCPM 的简化策略
- **注意力投影**: 直接映射，不需要合并
  - `q_proj.weight` → 直接加载
  - `k_proj.weight` → 直接加载
  - `v_proj.weight` → 直接加载
  - `o_proj.weight` → 直接加载

- **MLP 投影**: 可选合并优化
  - `gate_proj.weight` + `up_proj.weight` → `up_gate_proj.weight`
  - `down_proj.weight` → 直接加载

#### 对比 Qwen3 的复杂策略
- **注意力投影**: 需要复杂合并
  - `q_proj.weight` + `k_proj.weight` + `v_proj.weight` → `qkv_proj.weight`
  - 需要分片 ID 和位置计算

### 2. 内存效率分析

```python
# MiniCPM 注意力投影内存使用
q_memory = hidden_size * num_heads * head_dim * 4  # bytes
k_memory = hidden_size * num_kv_heads * head_dim * 4  # bytes
v_memory = hidden_size * num_kv_heads * head_dim * 4  # bytes
o_memory = num_heads * head_dim * hidden_size * 4  # bytes

total_attention_memory = q_memory + k_memory + v_memory + o_memory

# 保持分离的优势：
# 1. 支持 GQA (Grouped Query Attention)
# 2. 更灵活的量化策略
# 3. 简化的加载逻辑
# 4. 更好的可维护性
```

### 3. 性能优化建议

#### 可选的 MLP 合并
```python
# 如果选择合并 MLP 投影：
"up_gate_proj", "gate_proj", "gate"),  # 加载到前半部分
"up_gate_proj", "up_proj", "up"),     # 加载到后半部分

# 优势：
# - 单次矩阵运算替代两次
# - 更好的缓存局部性
# - 量化友好

# 劣势：
# - 增加加载复杂度
# - 可能影响某些优化
```

#### 直接加载的优势
```python
# MiniCPM 主要使用直接加载：
for loaded_weight_name, loaded_weight in weights_iterator:
    if loaded_weight_name in params_dict:
        # 直接加载，无需复杂处理
        param = params_dict[loaded_weight_name]
        param.copy_(loaded_weight, False)
```

## 特殊处理

### 1. 权重共享处理
```python
# MiniCPM 可能的权重共享
if hasattr(self, 'tie_word_embeddings') and self.tie_word_embeddings:
    # lm_head.weight 与 embed_tokens.weight 共享
    self.lm_head.weight = self.model.embed_tokens.weight
```

### 2. RMSNorm 处理
```python
# MiniCPM 使用 RMSNorm
# 权重通常是学习到的缩放参数
# 直接加载即可，无需特殊处理
```

### 3. Scale 配置处理
```python
# MiniCPM 的特殊缩放参数
if hasattr(self.config, 'scale_depth'):
    # 残差连接缩放
    pass

if hasattr(self.config, 'scale_emb'):
    # 嵌入缩放
    pass
```

## 与其他模型的对比

| 模型 | QKV 处理 | MLP 复杂度 | 加载策略 | 维护难度 |
|------|---------|------------|----------|----------|
| **MiniCPM** | 分离（简单） | 中等（可选合并） | 直接映射 | 低 |
| **Qwen3** | 合并（复杂） | 中等（已合并） | 分片映射 | 高 |
| **LLaMA** | 分离（简单） | 简单（分离） | 直接映射 | 低 |
| **Mistral** | 分离（简单） | 简单（分离） | 直接映射 | 低 |

## 实际使用示例

### 1. 基本使用
```python
from fastdeploy.model_executor.models.minicpm import MiniCPMForCausalLM
from fastdeploy.model_executor.load_weight_utils import get_weight_iterator

# 创建模型
model = MiniCPMForCausalLM(config)

# 获取权重迭代器
weights_iterator = get_weight_iterator("/path/to/minicpm")

# 加载权重
model.load_weights(weights_iterator)
```

### 2. 调试加载过程
```python
def debug_load_weights(model, weights_iterator):
    """调试版本，显示加载详情"""

    params_dict = dict(model.named_parameters())
    loaded_count = 0
    skipped_count = 0

    for loaded_weight_name, loaded_weight in weights_iterator:
        if loaded_weight_name in params_dict:
            print(f"✅ 加载: {loaded_weight_name}, shape: {loaded_weight.shape}")
            param = params_dict[loaded_weight_name]
            param.copy_(loaded_weight, False)
            loaded_count += 1
        else:
            print(f"⚠️  跳过: {loaded_weight_name} (未找到匹配参数)")
            skipped_count += 1

    print(f"📊 加载统计: 成功 {loaded_count}, 跳过 {skipped_count}")
```

## 总结

MiniCPM 的权重加载函数设计遵循以下原则：

1. **简化优先**: 利用 MiniCPM 原生的分离投影架构
2. **性能平衡**: 可选的 MLP 合并优化
3. **可维护性**: 清晰直接的映射逻辑
4. **兼容性**: 支持各种权重格式和框架差异
5. **扩展性**: 易于添加新的映射规则

相比 Qwen3，MiniCPM 的加载逻辑更加简单直接，这降低了实现复杂度和维护成本，同时保持了良好的性能特征。