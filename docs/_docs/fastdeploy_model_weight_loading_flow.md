# FastDeploy 模型权重加载完整流程图

根据对 FastDeploy 模型权重加载流程的深入分析，我为你总结了完整的 Mermaid 流程图：

## FastDeploy 模型权重加载完整流程图

```mermaid
graph TD
    A[用户调用 test_chat_py] --> B[LLM init]
    B --> C[LLMEngine from_engine_args]
    C --> D[LLMEngine init]
    D --> E[EngineService init]
    E --> J[EngineService start]

    %% Engine 初始化阶段 - 不加载模型
    E --> F[创建调度器 Scheduler]
    E --> G[创建资源管理器 ResourceManager]
    E --> H[启动队列服务 start_worker_queue_service]
    E --> I[创建 IPC 信号]

    %% Engine 启动服务 - 仍不加载模型
    J --> K[启动调度线程]
    J --> L[启动 token 处理器]
    J --> M[注册到路由器]
    J --> N[LLMEngine start_worker_service]

    %% 启动 Worker 服务
    N --> O[subprocess Popen]
    O --> P[paddle distributed launch]
    P --> Q[worker_process_py]

    %% Worker 进程启动 - 开始模型加载
    Q --> R[run_worker_proc]
    R --> S[parse_args]
    R --> T[init_distributed_environment]
    R --> U[initialize_fd_config]
    R --> V[PaddleDisWorkerProc]
    V --> W[init_device]

    %% 模型权重加载阶段
    W --> X[worker_proc load_model]
    X --> Y[PaddleDisWorkerProc load_model]
    Y --> Z[worker load_model]
    Z --> AA[GpuWorker load_model]
    AA --> BB[model_runner load_model]

    %% 实际的权重加载实现
    BB --> CC[GPUModelRunner load_model]
    CC --> DD[get_model_loader]
    DD --> EE[model_loader load_model]
    EE --> FF[DefaultModelLoader load_model]

    %% 具体的模型类和权重加载
    FF --> GG[ModelRegistry get_class]
    GG --> HH[Qwen3ForCausalLM fd_config]
    HH --> II[Qwen3ForCausalLM init]
    II --> JJ[self load_weights]

    %% 权重加载详细流程
    JJ --> KK[stacked_params_mapping]
    JJ --> LL[params_dict dict self_named_parameters]
    JJ --> MM[遍历 weights_iterator]
    MM --> NN[处理分片参数合并]
    MM --> OO[直接加载权重]
    JJ --> PP[weight_loader]

    %% 模型加载完成后的初始化
    PP --> QQ[设置 loaded_model_signal]
    QQ --> RR[分布式同步 paddle_distributed_barrier]
    RR --> SS[initialize_kv_cache]
    SS --> TT[graph_optimize_and_warm_up_model]
    TT --> UU[start_task_queue_service]
    UU --> VV[event_loop_normal]

    %% 最终模型准备就绪
    VV --> WW[Worker 准备接收请求]
    WW --> XX[模型可以处理推理请求]

    %% 样式设置
    classDef engineBox fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef workerBox fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef loadBox fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef modelBox fill:#fff3e0,stroke:#e65100,stroke-width:2px

    class E,F,G,H,I,J,K,L,M,N engineBox
    class Q,R,S,T,U,V,W,X,Y,Z,AA workerBox
    class BB,CC,DD,EE,FF,GG,HH,II,JJ,KK,LL,MM,NN,OO,PP,QQ,RR,SS,TT,UU,VV loadBox
    class WW,XX modelBox


```

## 关键时间节点总结

```mermaid
timeline
    title FastDeploy 模型权重加载时序

    section Engine 层面
        用户调用 test_chat.py : 创建 LLM 实例
        EngineService.__init__() : 初始化调度器、资源管理器
        EngineService.start() : 启动服务线程，但不加载模型

    section Worker 进程启动
        _start_worker_service() : 通过 subprocess 启动 worker_process.py
        run_worker_proc() : Worker 进程主函数
        init_device() : 初始化 GPU 设备

    section 模型权重加载
        worker_proc.load_model() : 开始模型加载
        GPUModelRunner.load_model() : 模型运行器加载
        DefaultModelLoader.load_model() : **真正的权重加载**
        Qwen3ForCausalLM.__init__() : 模型类实例化
        Qwen3ForCausalLM.load_weights() : 具体的权重加载逻辑

    section 模型准备就绪
        initialize_kv_cache() : 初始化 KV 缓存
        graph_optimize_and_warm_up_model() : 图优化和预热
        event_loop_normal() : Worker 进入事件循环
        模型可以处理推理请求 : 完整的推理服务就绪
```

## 详细步骤说明

### 1. Engine 层面（不加载模型权重）

**文件位置**:

- `fastdeploy/entrypoints/llm.py:73-107`
- `fastdeploy/engine/engine.py:78-95`
- `fastdeploy/engine/common_engine.py:73-166`

**核心功能**:

```python
class EngineService:
    def __init__(self, cfg, start_queue=True):
        # 创建各种管理组件，但不加载模型
        self.scheduler = cfg.scheduler_config.scheduler()
        self.resource_manager = ResourceManager(...)
        self._init_worker_monitor_signals()

    def start(self):
        # 启动服务线程，但仍不加载模型
        self.insert_task_to_worker_thread = threading.Thread(...)
        self.token_processor.run()
```

### 2. Worker 进程启动阶段

**文件位置**: `fastdeploy/engine/engine.py:490-596`

**关键代码**:

```python
def _start_worker_service(self):
    # 使用 paddle.distributed.launch 启动 worker_process.py
    pd_cmd = f"{sys.executable} -m paddle.distributed.launch"
    pd_cmd += " ../worker/worker_process.py"

    # 启动子进程
    p = subprocess.Popen(pd_cmd, shell=True, preexec_fn=os.setsid)
    return p
```

### 3. 模型权重加载阶段

**文件位置**: `fastdeploy/worker/worker_process.py:994`

**执行流程**:

```python
def run_worker_proc() -> None:
    # 1. 分布式环境初始化
    ranks, local_rank = init_distributed_environment()

    # 2. 配置初始化
    fd_config = initialize_fd_config(args, ranks, local_rank)

    # 3. 创建 Worker 进程
    worker_proc = PaddleDisWorkerProc(fd_config, ranks, local_rank)

    # 4. 设备初始化
    worker_proc.init_device()

    # 5. 【关键】模型权重加载
    worker_proc.load_model()  # ← 这里开始加载

    # 6. KV 缓存初始化
    worker_proc.initialize_kv_cache()

    # 7. 图优化和预热
    worker_proc.graph_optimize_and_warm_up_model()
```

### 4. 具体的权重加载实现

**调用链**:

1. `worker_process.py:575` - `PaddleDisWorkerProc.load_model()`
2. `gpu_worker.py:174` - `GpuWorker.load_model()`
3. `gpu_model_runner.py:1312` - `GPUModelRunner.load_model()`
4. `default_loader.py` - `DefaultModelLoader.load_model()`
5. `qwen3.py:257` - `Qwen3ForCausalLM.load_weights()`

**核心权重加载逻辑**:

```python
@paddle.no_grad()
def load_weights(self, weights_iterator) -> None:
    # 1. 分片参数映射（用于合并分离的权重）
    stacked_params_mapping = [
        ("qkv_proj", "q_proj", "q"),      # QKV 投影合并
        ("up_gate_proj", "gate_proj", "gate"),  # 门控投影合并
        # ...
    ]

    # 2. 获取模型参数字典
    params_dict = dict(self.named_parameters())

    # 3. 遍历加载每个权重
    for loaded_weight_name, loaded_weight in weights_iterator:
        # 4. 处理分片合并或直接加载
        weight_loader(param, loaded_weight, loaded_weight_name)
```

## 调试建议

### 1. Worker 进程调试

由于模型权重加载在 Worker 进程中进行，外部调试需要：

```python
# 在 worker_process.py 中添加断点
def run_worker_proc():
    import pdb; pdb.set_trace()  # Worker 启动时会停在这里
    # ... 其他代码
```

### 2. 使用远程调试

```python
# 在 worker_process.py 开头添加
import debugpy
debugpy.listen(5678)
debugpy.wait_for_client()
```

### 3. 关键调试位置

- **Worker 启动**: `worker_process.py:1010`
- **模型加载**: `worker_process.py:994`
- **权重加载**: `gpu_model_runner.py:1317`
- **具体模型**: `qwen3.py:257`

## 架构设计特点

### 1. 多进程分离

- **Engine 进程**: 负责调度、资源管理
- **Worker 进程**: 负责模型加载和推理计算
- **缓存管理进程**: 独立的 KV 缓存管理

### 2. 分布式支持

- 通过 `paddle.distributed.launch` 实现多卡并行
- 支持张量并行、数据并行、专家并行

### 3. 异步处理

- Engine 和 Worker 通过队列和信号通信
- 支持动态权重加载和模型更新

## 总结

FastDeploy 的模型权重加载是一个高度复杂的多进程分布式系统：

1. **Engine 不加载模型**：只负责调度和管理
2. **Worker 加载模型**：通过独立的进程加载权重
3. **分布式启动**：使用 PaddlePaddle 的分布式框架
4. **异步通信**：通过 IPC 信号和队列进行进程间通信

这种架构确保了高性能和可扩展性，但也增加了调试的复杂性。要调试模型权重加载，需要在 Worker 进程内部设置断点。
