## **项目：在 FastDeploy 中原生支持 MiniCPM4.1-8B**

**项目状态**: ✅ **已基本完成，需要完善和优化**

**当前进度**: FastDeploy已实现MiniCPM4.1-8B的基础支持，包括：
- ✅ 完整的模型架构实现
- ✅ 基础稀疏注意力支持
- ✅ 多种量化方案 (W4AFP8, W4A8, W8A16)
- ✅ 权重转换工具
- ✅ 基础测试套件
- ✅ API服务器集成

**剩余目标**: 完善稀疏注意力、性能优化、生产验证

**核心技术路径**:
1. **完善**: 基于现有实现，完善InfLLM v2稀疏注意力机制
2. **优化**: 优化现有量化方案的性能和内存使用
3. **验证**: 进行生产环境验证和性能基准测试

**核心策略**: 快速迭代，重点完善关键功能，确保6-8周内完成生产就绪支持。
---

### **Phase 1: 稀疏注意力机制完善 (1-2 周)**

**目标**: 完善InfLLM v2稀疏注意力机制，提升长文本处理性能。

**任务 1.1: 完善现有稀疏注意力实现**
* **当前状态**: 基础框架已存在，需要完善具体实现
* **文件**: `fastdeploy/model_executor/models/minicpm41.py` - `MiniCPM41Attention` 类
* **完善内容**:
  ```python
  class MiniCPM41Attention(nn.Layer):
      def forward(self, hidden_states, attention_mask, position_ids, sparse_config=None):
          # 完善稀疏注意力逻辑
          if sparse_config and self._should_use_sparse(hidden_states.shape[1]):
              return self._sparse_attention_forward(
                  hidden_states, attention_mask, position_ids, sparse_config
              )
          else:
              return self._standard_attention_forward(
                  hidden_states, attention_mask, position_ids
              )

      def _should_use_sparse(self, seq_len):
          """动态判断是否使用稀疏注意力"""
          dense_len = self.sparse_config.dense_len if self.sparse_config else 8192
          return seq_len > dense_len

      def _sparse_attention_forward(self, hidden_states, attention_mask, position_ids, sparse_config):
          """InfLLM v2稀疏注意力实现"""
          # 1. 语义核计算
          # 2. 块选择和Top-k计算
          # 3. 局部窗口注意力
          # 4. 合并注意力结果
  ```

**任务 1.2: 优化稀疏注意力性能**
* **目标**: 确保稀疏注意力在长文本场景下的性能优势
* **优化点**:
  - 减少内存分配和拷贝
  - 优化块选择算法
  - 提升Top-k计算效率
  - 添加CUDA kernel优化（如需要）

**任务 1.3: 添加稀疏注意力测试**
* **文件**: `tests/models/test_minicpm41.py`
* **测试内容**:
  ```python
  def test_sparse_attention_correctness(self):
      """验证稀疏注意力的正确性"""

  def test_sparse_attention_performance(self):
      """验证稀疏注意力的性能优势"""

  def test_sparse_dense_switch(self):
      """验证稀疏/密集注意力切换"""
  ```

---

### **Phase 2: 量化方案优化 (1-2 周)**

**目标**: 优化现有量化方案的性能和内存使用。

**任务 2.1: 完善量化配置**
* **当前状态**: 量化框架已实现，需要优化配置和性能
* **文件**: `fastdeploy/model_executor/models/minicpm41_quant.py`
* **优化内容**:
  ```python
  class MiniCPM41QuantizationConfig:
      def get_optimized_config(self, seq_len, hardware_type="gpu"):
          """根据序列长度和硬件类型获取优化配置"""
          if seq_len <= 4096:
              return self._get_short_sequence_config(hardware_type)
          elif seq_len <= 16384:
              return self._get_medium_sequence_config(hardware_type)
          else:
              return self._get_long_sequence_config(hardware_type)
  ```

**任务 2.2: 性能调优**
* **目标**: 提升各量化方案的推理性能
* **调优方向**:
  - W4AFP8: 优化FP8计算精度
  - W4A8: 改进INT4权重量化算法
  - W8A16: 减少精度损失
  - 内存优化: 减少量化过程中的内存峰值

**任务 2.3: 添加量化基准测试**
* **文件**: `benchmarks/minicpm41_performance.py`
* **基准测试内容**:
  - 各量化方案的精度对比
  - 推理速度基准测试
  - 内存使用分析
  - 不同序列长下的性能表现

---

### **Phase 1: 核心模型架构实现 (2-3 周)**

这是整个项目的基础部分，实现 MiniCPM4.1-8B 的核心模型架构。

**任务 1.1: 实现基础组件 (`minicpm41/modeling.py`)**
* **目标**: 实现 MiniCPM4.1-8B 的基础组件，包括注意力机制、MLP、层归一化等。
* **创建文件**: `fastdeploy/model_executor/models/minicpm41/modeling.py`
* **核心组件**:
  ```python
  class MiniCPM41RMSNorm(nn.Layer):
      """MiniCPM4.1-8B RMS归一化"""

  class MiniCPM41MLP(nn.Layer):
      """MiniCPM4.1-8B MLP (SwiGLU激活)"""

  class MiniCPM41RotaryEmbedding(nn.Layer):
      """MiniCPM4.1-8B RoPE位置编码"""

  class MiniCPM41LinearAttention(nn.Layer):
      """MiniCPM4.1-8B 稀疏注意力实现"""
  ```

**任务 1.2: 实现注意力机制 (`minicpm41/attention.py`)**
* **目标**: 实现支持稀疏注意力的完整注意力机制。
* **核心功能**:
  ```python
  class MiniCPM41Attention(nn.Layer):
      """MiniCPM4.1-8B 注意力机制"""
      def __init__(self, fd_config, layer_id):
          # QKV并行投影
          self.qkv_proj = QKVParallelLinear(...)

          # 输出投影
          self.o_proj = RowParallelLinear(...)

          # 注意力计算
          self.attn = Attention(...)

          # Q/K归一化（如果启用）
          if self.use_qk_norm:
              self.q_norm = RMSNorm(...)
              self.k_norm = RMSNorm(...)

      def forward(self, hidden_states, attention_mask, sparse_config=None):
          # 标准注意力或稀疏注意力
          if sparse_config and self.should_use_sparse():
              return self._sparse_attention_forward(...)
          else:
              return self._standard_attention_forward(...)
  ```

**任务 1.3: 实现主模型 (`minicpm41.py`)**
* **目标**: 实现完整的 MiniCPM4.1-8B 模型结构。
* **创建文件**: `fastdeploy/model_executor/models/minicpm41.py`
* **核心结构**:
  ```python
  @ModelRegistry.register_model_class(
      architecture="MiniCPM41ForCausalLM",
      module_name="minicpm41",
      category=ModelCategory.TEXT_GENERATION,
      primary_use=ModelCategory.TEXT_GENERATION,
  )
  class MiniCPM41ForCausalLM(ModelForCasualLM):
      def __init__(self, fd_config):
          # 词嵌入
          self.embed_tokens = VocabParallelEmbedding(...)

          # 解码器层
          self.layers = nn.LayerList([
              MiniCPM41DecoderLayer(fd_config, layer_id=i)
              for i in range(fd_config.model_config.num_hidden_layers)
          ])

          # 最终归一化
          self.norm = RMSNorm(...)

          # LM头
          self.lm_head = ParallelLMHead(...)
  ```

---

### **Phase 2: 量化支持实现 (2-3 周)**

实现 MiniCPM4.1-8B 的完整量化支持，包括多种量化方案。

**任务 2.1: 实现量化配置 (`minicpm41_quant.py`)**
* **目标**: 实现支持多种量化方案的配置和模型。
* **创建文件**: `fastdeploy/model_executor/models/minicpm41_quant.py`
* **量化配置**:
  ```python
  class MiniCPM41QuantizationConfig(QuantConfigBase):
      """MiniCPM4.1-8B 量化配置"""
      def __init__(self, quant_type="w4afp8", weight_bits=4, activation_bits=8, ...):

  @dataclass
  class QuantizedMiniCPM41ModelConfig(MiniCPM41ModelConfig):
      """量化版MiniCPM4.1-8B配置"""
      quantization_config: Optional[MiniCPM41QuantizationConfig] = None
  ```

**任务 2.2: 实现量化模型组件**
* **目标**: 为每个组件创建量化版本。
* **量化组件**:
  ```python
  class QuantizedMiniCPM41MLP(MiniCPM41MLP):
      """量化版MLP"""
      def __init__(self, fd_config, layer_id):
          # 使用量化线性层
          self.gate_proj = QuantizedRowParallelLinear(...)
          self.up_proj = QuantizedRowParallelLinear(...)
          self.down_proj = QuantizedRowParallelLinear(...)

  class QuantizedMiniCPM41Attention(MiniCPM41Attention):
      """量化版注意力"""
      def __init__(self, fd_config, layer_id):
          # 使用量化线性层
          self.qkv_proj = QuantizedQKVParallelLinear(...)
          self.o_proj = QuantizedRowParallelLinear(...)
  ```

**任务 2.3: 实现权重处理**
* **目标**: 实现量化权重的加载和处理。
* **权重处理**:
  ```python
  def set_state_dict(self, state_dict):
      """处理量化权重映射"""
      if self.quant_config:
          state_dict = self._process_quantized_weights(state_dict)
      # 标准权重映射
      mapped_state_dict = self._map_weights(state_dict)
      self.load_dict(mapped_state_dict)

  def _process_quantized_weights(self, state_dict):
      """处理不同类型的量化权重"""
      quant_type = self.quant_config.quant_type
      if quant_type == "w4afp8":
          return self._process_w4afp8_weights(state_dict)
      elif quant_type == "w4a8":
          return self._process_w4a8_weights(state_dict)
      # ...
  ```

---

### **Phase 3: 权重转换与工具开发 (1-2 周)**

开发权重转换脚本和相关工具。

**任务 3.1: 实现权重转换脚本 (`tools/convert_minicpm41_weights.py`)**
* **目标**: 实现从 HuggingFace 到 PaddlePaddle 的权重转换。
* **核心功能**:
  ```python
  def convert_weight_name(hf_name):
      """HF权重名到PaddlePaddle权重名映射"""
      name_mapping = {
          "model.embed_tokens.weight": "model.embed_tokens.weight",
          "model.layers.{}.attention.wq.weight": "model.layers.{}.self_attn.qkv_proj.q_proj.weight",
          "model.layers.{}.attention.wk.weight": "model.layers.{}.self_attn.qkv_proj.k_proj.weight",
          # ... 完整映射表
      }

  def convert_weights(state_dict, dtype="float16"):
      """权重格式转换"""
      for name, tensor in state_dict.items():
          paddle_name = convert_weight_name(name)
          paddle_tensor = paddle.to_tensor(tensor.cpu().numpy())
          # 数据类型转换
          paddle_state_dict[paddle_name] = paddle_tensor
  ```

**任务 3.2: 实现配置转换**
* **目标**: 转换模型配置文件。
* **配置转换**:
  ```python
  def convert_config(hf_config, model_name):
      """HF配置到PaddlePaddle配置转换"""
      paddle_config = {
          "model_type": "minicpm41",
          "hidden_size": hf_config.hidden_size,
          "num_hidden_layers": hf_config.num_hidden_layers,
          # 稀疏注意力配置
          "sparse_config": hf_config.sparse_config,
          # RoPE缩放配置
          "rope_scaling": hf_config.rope_scaling,
      }
  ```

---

### **Phase 4: 混合推理模式实现 (1-2 周)**

实现 MiniCPM4.1-8B 的混合推理模式支持。

**任务 4.1: 实现混合推理逻辑**
* **目标**: 支持深度推理和快速推理模式切换。
* **核心逻辑**:
  ```python
  class MiniCPM41ForCausalLM(ModelForCasualLM):
      def prepare_inputs_for_generation(self, input_ids, enable_thinking=False, **kwargs):
          """生成输入准备，支持混合推理模式"""
          if enable_thinking:
              # 深度推理模式预处理
              input_ids = self._add_thinking_tokens(input_ids)
          else:
              # 快速推理模式预处理
              input_ids = self._remove_thinking_tokens(input_ids)
          return super().prepare_inputs_for_generation(input_ids, **kwargs)

      def _process_thinking_output(self, output):
          """处理推理模式的输出"""
          if self.enable_thinking:
              # 深度推理后处理
              return self._format_thinking_response(output)
          return output
  ```

**任务 4.2: 实现特殊Token处理**
* **目标**: 处理推理模式的特殊token。
* **特殊Token**:
  ```python
  class MiniCPM41Tokenizer:
      def apply_chat_template(self, messages, enable_thinking=False, **kwargs):
          """应用对话模板，支持混合推理模式"""
          if enable_thinking:
              # 添加/think标记启用推理模式
              messages = self._add_thinking_markers(messages)
          else:
              # 添加/no_think标记禁用推理模式
              messages = self._add_no_thinking_markers(messages)
          return self._standard_chat_template(messages, **kwargs)
  ```

---

### **Phase 5: 长上下文和稀疏注意力优化 (2-3 周)**

实现和优化长上下文支持和稀疏注意力机制。

**任务 5.1: 实现稀疏注意力**
* **目标**: 实现可训练稀疏注意力机制。
* **稀疏注意力实现**:
  ```python
  class SparseAttentionBackend(AttentionBackend):
      """稀疏注意力后端"""
      def forward(self, q, k, v, sparse_config, **kwargs):
          if sparse_config and self.should_use_sparse_attention():
              return self._sparse_attention_forward(q, k, v, sparse_config)
          else:
              return self._standard_attention_forward(q, k, v, **kwargs)

      def _sparse_attention_forward(self, q, k, v, sparse_config):
          """稀疏注意力前向传播"""
          # 实现InfLLM v2算法
          # 1. 语义核计算
          # 2. 块选择
          # 3. 局部窗口注意力
          # 4. Top-k注意力计算
  ```

**任务 5.2: 长上下文优化**
* **目标**: 优化长上下文处理的性能和内存使用。
* **长上下文优化**:
  ```python
  def get_optimized_cache_config(seq_len, model_config):
      """根据序列长度优化缓存配置"""
      if seq_len <= 4096:
          return {"cache_mode": "normal", "max_batch_size": 32}
      elif seq_len <= 16384:
          return {"cache_mode": "prefix", "max_batch_size": 16}
      else:
          return {
              "cache_mode": "sparse",
              "max_batch_size": 8,
              "block_size": model_config.sparse_config.block_size,
          }
  ```

---

### **Phase 6: 性能优化和基准测试 (2 周)**

进行性能优化和基准测试。

**任务 6.1: 性能优化**
* **目标**: 优化推理性能和内存使用。
* **优化措施**:
  ```python
  class OptimizedMiniCPM41Config:
      """性能优化配置"""
      def get_memory_optimization(self):
          """内存优化配置"""
          if self.quant_config.weight_bits <= 4:
              return {
                  "enable_weight_reuse": True,
                  "enable_activation_quantization": True,
                  "memory_efficient_cache": True,
              }

      def get_computation_optimization(self):
          """计算优化配置"""
          return {
              "use_flash_attention": True,
              "enable_gradient_checkpointing": False,
              "use_cudnn_graph": True,
          }
  ```

**任务 6.2: 基准测试工具**
* **目标**: 实现性能基准测试工具。
* **基准测试**:
  ```python
  class MiniCPM41PerformanceBenchmark:
      """MiniCPM4.1-8B 性能基准测试"""
      def benchmark_latency(self, prompts, max_tokens=256):
          """延迟基准测试"""

      def benchmark_throughput(self, prompts, max_tokens=256):
          """吞吐量基准测试"""

      def compare_quantizations(self):
          """量化方案性能对比"""

      def profile_memory_usage(self):
          """内存使用分析"""
  ```

---

### **Phase 7: 测试和文档 (1-2 周)**

完善测试用例和文档。

**任务 7.1: 单元测试**
* **目标**: 编写完整的单元测试。
* **测试内容**:
  ```python
  class TestMiniCPM41Model(unittest.TestCase):
      def test_model_creation(self):
          """测试模型创建"""

      def test_forward_pass(self):
          """测试前向传播"""

      def test_weight_loading(self):
          """测试权重加载"""

      def test_quantization(self):
          """测试量化功能"""

      def test_sparse_attention(self):
          """测试稀疏注意力"""
  ```

**任务 7.2: 集成测试**
* **目标**: 端到端集成测试。
* **集成测试**:
  ```python
  class TestMiniCPM41E2E(unittest.TestCase):
      def test_end_to_end_generation(self):
          """端到端生成测试"""

      def test_api_server_integration(self):
          """API服务器集成测试"""

      def test_performance_validation(self):
          """性能验证测试"""
  ```

**任务 7.3: 文档编写**
* **目标**: 编写完整的使用文档。
* **文档结构**:
  ```markdown
  # docs/get_started/minicpm41.md
  - 模型特点
  - 快速开始
  - 量化部署
  - 配置选项
  - 混合推理模式
  - 高级部署
  - 性能优化
  - 常见问题
  ```

---

### **Phase 8: 部署和集成 (1 周)**

完成最终的部署集成和验证。

**任务 8.1: API服务器集成**
* **目标**: 确保与FastDeploy API服务器的完整集成。
* **集成验证**:
  ```bash
  # 启动API服务器
  python -m fastdeploy.entrypoints.openai.api_server \
      --model openbmb/MiniCPM4.1-8B \
      --quantization w4afp8 \
      --tensor-parallel-size 4

  # 客户端调用验证
  curl -X POST "http://localhost:8180/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d '{"model": "openbmb/MiniCPM4.1-8B", "messages": [...]}'
  ```

**任务 8.2: 最终验证**
* **目标**: 最终功能和性能验证。
* **验证清单**:
  - [ ] 基础推理功能正常
  - [ ] 量化方案工作正常
  - [ ] 稀疏注意力机制正确
  - [ ] 混合推理模式可用
  - [ ] 长上下文处理稳定
  - [ ] 性能指标达标
  - [ ] 内存使用合理
  - [ ] API服务器集成正常
  - [ ] 文档完整准确
  - [ ] 测试覆盖充分

---

### **项目里程碑**

| 里程碑 | 时间 | 主要交付物 |
|--------|------|-----------|
| Phase 0 | 第1-2天 | 项目骨架、配置体系 |
| Phase 1 | 第2-3周 | 核心模型架构 |
| Phase 2 | 第4-6周 | 量化支持实现 |
| Phase 3 | 第7-8周 | 权重转换工具 |
| Phase 4 | 第8-10周 | 混合推理模式 |
| Phase 5 | 第10-13周 | 稀疏注意力优化 |
| Phase 6 | 第13-15周 | 性能优化和基准测试 |
| Phase 7 | 第15-17周 | 测试和文档 |
| Phase 8 | 第17-18周 | 部署集成和最终验证 |

**总项目周期**: 约18周

**核心成功指标**:
- 功能完整性: 100% 支持MiniCPM4.1-8B所有特性
- 性能指标: W4AFP8量化下达到4倍内存压缩，2倍推理加速
- 代码质量: 95%以上测试覆盖率，通过所有集成测试
- 文档完整性: 提供完整的使用文档和API说明
- 生产就绪: 可直接用于生产环境部署