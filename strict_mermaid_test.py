#!/usr/bin/env python3
"""
更严格的 Mermaid 语法验证
"""

def test_mermaid_online_compatibility():
    """测试与在线 Mermaid 编辑器的兼容性"""

    mermaid_code = '''
    A[用户调用 test_chat_py] --> B[LLM init]
    B --> C[LLMEngine from_engine_args]
    C --> D[LLMEngine init]
    D --> E[EngineService init]
    E --> J[EngineService start]
    E --> F[创建调度器 Scheduler]
    E --> G[创建资源管理器 ResourceManager]
    E --> H[启动队列服务 start_worker_queue_service]
    E --> I[创建 IPC 信号]
    J --> K[启动调度线程]
    J --> L[启动 token 处理器]
    J --> M[注册到路由器]
    J --> N[LLMEngine start_worker_service]
    N --> O[subprocess Popen]
    O --> P[paddle distributed launch]
    P --> Q[worker_process_py]
    Q --> R[run_worker_proc]
    R --> S[parse_args]
    R --> T[init_distributed_environment]
    R --> U[initialize_fd_config]
    R --> V[PaddleDisWorkerProc]
    V --> W[init_device]
    W --> X[worker_proc load_model]
    X --> Y[PaddleDisWorkerProc load_model]
    Y --> Z[worker load_model]
    Z --> AA[GpuWorker load_model]
    AA --> BB[model_runner load_model]
    BB --> CC[GPUModelRunner load_model]
    CC --> DD[get_model_loader]
    DD --> EE[model_loader load_model]
    EE --> FF[DefaultModelLoader load_model]
    FF --> GG[ModelRegistry get_class]
    GG --> HH[Qwen3ForCausalLM fd_config]
    HH --> II[Qwen3ForCausalLM init]
    II --> JJ[self load_weights]
    JJ --> KK[stacked_params_mapping]
    JJ --> LL[params_dict dict self_named_parameters]
    JJ --> MM[遍历 weights_iterator]
    MM --> NN[处理分片参数合并]
    MM --> OO[直接加载权重]
    JJ --> PP[weight_loader]
    PP --> QQ[设置 loaded_model_signal]
    QQ --> RR[分布式同步 paddle_distributed_barrier]
    RR --> SS[initialize_kv_cache]
    SS --> TT[graph_optimize_and_warm_up_model]
    TT --> UU[start_task_queue_service]
    UU --> VV[event_loop_normal]
    VV --> WW[Worker 准备接收请求]
    WW --> XX[模型可以处理推理请求]
    '''

    print("🧪 严格语法测试...")

    # 检查每一行
    lines = [line.strip() for line in mermaid_code.strip().split('\n') if line.strip()]

    issues = []

    for i, line in enumerate(lines, 1):
        print(f"测试第 {i} 行: {line[:50]}...")

        # 检查节点ID格式（应该是字母）
        if '-->' in line:
            parts = line.split('-->')
            for part in parts:
                part = part.strip()
                if '[' in part:
                    node_id = part.split('[')[0]
                    if not node_id.isalpha():
                        issues.append(f"第 {i} 行: 节点ID '{node_id}' 不是纯字母")

                # 检查节点标签
                if '[' in part and ']' in part:
                    label = part.split('[')[1].split(']')[0]
                    # 检查是否有特殊字符
                    forbidden_chars = ['.', '(', ')', '{', '}', '|', '"', "'"]
                    for char in forbidden_chars:
                        if char in label:
                            issues.append(f"第 {i} 行: 标签包含禁止字符 '{char}'")

    if issues:
        print("❌ 发现问题:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✅ 通过严格语法测试")

    # 输出最小化的测试版本
    minimal_test = '''
graph TD
    A[Start] --> B[End]
    classDef default fill:#f9f9f9,stroke:#333,stroke-width:2px
    class A,B default
    '''

    print("\n🔬 最小化测试版本:")
    print(minimal_test)
    print("如果这个版本能工作，说明基础语法没问题")

if __name__ == "__main__":
    test_mermaid_online_compatibility()