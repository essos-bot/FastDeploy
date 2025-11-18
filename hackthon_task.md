NO.74 为 FastDeploy 新增 MiniCPM4.1-8B 模型
详细描述：

为 FastDeploy 提供部署高性能的 openbmb/MiniCPM4.1-8B 系列模型的能力.

提交内容

MiniCPM4.1-8B相关模型的组网代码, 提交至 FastDeploy/fastdeploy/model_executor/models/ 目录下. 同时提交模型使用说明文档.
如需开发自定义算子, 提交至 FastDeploy/custom_ops/gpu_ops/ 目录下.
为 MiniCPM4.1-8B系列模型适配FastDeploy现有的各种低bit量化推理的能力.