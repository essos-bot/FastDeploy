from fastdeploy import LLM, SamplingParams

msg1 = [
    {"role": "system", "content": "I'm a helpful AI assistant."},
    {"role": "user", "content": "把李白的静夜思改写为现代诗"},
]
msg2 = [
    {"role": "system", "content": "I'm a helpful AI assistant."},
    {"role": "user", "content": "Write me a poem about large language model."},
]
messages = [msg1]

# 采样参数
sampling_params = SamplingParams(top_p=0.95, max_tokens=100)  # Reduced tokens for faster testing

# 加载模型
# llm = LLM(model="baidu/ERNIE-4.5-0.3B-Paddle", tensor_parallel_size=1, max_model_len=8192)
llm = LLM(
    model="/home/aistudio/data/models/PaddlePaddle/ERNIE-4.5-0.3B-Paddle", tensor_parallel_size=1, max_model_len=8192
)
print("Starting to load model...")
# llm = LLM(model="/home/aistudio/data/models/39695/minicpm4.1-8b", tensor_parallel_size=1, max_model_len=8192)
print("Model loaded successfully!")
# 批量进行推理（llm内部基于资源情况进行请求排队、动态插入处理）
outputs = llm.chat(messages, sampling_params)
print(outputs)
# 输出结果
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs.text
