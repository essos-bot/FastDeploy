"""
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# InfLLM-V2 Attention Backend Implementation
# Reference: https://github.com/OpenBMB/infllmv2_cuda_impl
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import paddle

from fastdeploy.model_executor.layers.attention.base_attention_backend import (
    AttentionBackend,
    AttentionMetadata,
)
from fastdeploy.model_executor.layers.attention.infllmv2_attention_metadata import (
    InfLLMV2AttentionMetadata,
)

if TYPE_CHECKING:
    from fastdeploy.model_executor.forward_meta import ForwardMeta


class InfLLMV2AttentionBackend(AttentionBackend):
    """
    InfLLM-V2稀疏注意力后端实现

    实现InfLLM-V2的两阶段稀疏注意力机制:
    - Stage 1: Top-K上下文选择，使用语义核计算相关性分数并聚合
    - Stage 2: 稀疏注意力计算，仅在选中的块上进行注意力计算
    """

    def __init__(self, fd_config, kv_num_heads: int, num_heads: int, head_dim: int, **kwargs):
        """
        初始化InfLLM-V2注意力后端

        Args:
            fd_config: FastDeploy配置对象
            kv_num_heads: KV头的数量
            num_heads: 查询头的数量
            head_dim: 头维度
            **kwargs: 其他参数
        """
        super().__init__()
        self.fd_config = fd_config
        self.kv_num_heads = kv_num_heads
        self.num_heads = num_heads
        self.head_dim = head_dim

        # InfLLM-V2特有参数
        self.kernel_size = getattr(fd_config, 'infllmv2_kernel_size', 32)
        self.kernel_stride = getattr(fd_config, 'infllmv2_kernel_stride', 16)
        self.topk = getattr(fd_config, 'infllmv2_topk', 64)
        self.dense_len = getattr(fd_config, 'infllmv2_dense_len', 8192)
        self.block_size = getattr(fd_config, 'infllmv2_block_size', 64)
        self.window_size = getattr(fd_config, 'infllmv2_window_size', 2048)
        self.use_nope = getattr(fd_config, 'infllmv2_use_nope', False)
        self.init_blocks = getattr(fd_config, 'infllmv2_init_blocks', 1)

        # 验证参数合理性
        self._validate_parameters()

        # 检查CUDA可用性
        self._check_cuda_availability()

    def _validate_parameters(self):
        """验证参数合理性"""
        if self.kernel_size <= 0:
            raise ValueError("infllmv2_kernel_size must be positive")
        if self.kernel_stride <= 0:
            raise ValueError("infllmv2_kernel_stride must be positive")
        if self.topk <= 0:
            raise ValueError("infllmv2_topk must be positive")
        if self.block_size <= 0:
            raise ValueError("infllmv2_block_size must be positive")
        if self.kernel_stride > self.kernel_size:
            raise ValueError("infllmv2_kernel_stride should not exceed infllmv2_kernel_size")

    def _check_cuda_availability(self):
        """检查CUDA可用性"""
        if not paddle.is_compiled_with_cuda():
            raise RuntimeError("InfLLM-V2 requires CUDA support but PaddlePaddle was not compiled with CUDA")

    def init_attention_metadata(self, forward_meta: ForwardMeta) -> AttentionMetadata:
        """
        初始化InfLLM-V2特有的元数据

        Args:
            forward_meta: 前向传播元数据

        Returns:
            InfLLMV2AttentionMetadata: InfLLM-V2注意力元数据
        """
        # 获取当前序列长度
        current_seq_len = forward_meta.seq_len

        # 创建InfLLM-V2注意力元数据
        attn_metadata = InfLLMV2AttentionMetadata(
            kernel_size=self.kernel_size,
            kernel_stride=self.kernel_stride,
            init_blocks=self.init_blocks,
            block_size=self.block_size,
            window_size=self.window_size,
            topk=self.topk,
            use_nope=self.use_nope,
            dense_len=self.dense_len,
            max_seq_len=getattr(forward_meta, 'max_seq_len', 0),
            current_seq_len=current_seq_len,
            device=str(paddle.device.get_device()),
            dtype=paddle.get_default_dtype(),
        )

        forward_meta.attn_metadata = attn_metadata
        return attn_metadata

    def forward_decode(
        self,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        qkv: paddle.Tensor,
        compressed_kv: paddle.Tensor,
        k_pe: paddle.Tensor,
        layer: paddle.nn.Layer,
        forward_meta: ForwardMeta,
    ) -> paddle.Tensor:
        """
        Decode模式前向传播 - 适用于逐token生成

        Args:
            q: 查询张量 [batch_size, num_heads, 1, head_dim]
            k: 键张量 [batch_size, kv_num_heads, seq_len, head_dim]
            v: 值张量 [batch_size, kv_num_heads, seq_len, head_dim]
            qkv: QKV张量 (未使用)
            compressed_kv: 压缩KV缓存 (未使用)
            k_pe: 键位置编码 (未使用)
            layer: 注意力层
            forward_meta: 前向传播元数据

        Returns:
            paddle.Tensor: 注意力输出
        """
        return self._infllmv2_forward(q, k, v, forward_meta, mode="decode")

    def forward_extend(
        self,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        qkv: paddle.Tensor,
        compressed_kv: paddle.Tensor,
        k_pe: paddle.Tensor,
        layer: paddle.nn.Layer,
        forward_meta: ForwardMeta,
    ) -> paddle.Tensor:
        """
        Extend模式前向传播 - 适用于预填充阶段

        Args:
            q: 查询张量 [batch_size, seq_len_q, num_heads, head_dim]
            k: 键张量 [batch_size, seq_len_k, kv_num_heads, head_dim]
            v: 值张量 [batch_size, seq_len_k, kv_num_heads, head_dim]
            qkv: QKV张量 (未使用)
            compressed_kv: 压缩KV缓存 (未使用)
            k_pe: 键位置编码 (未使用)
            layer: 注意力层
            forward_meta: 前向传播元数据

        Returns:
            paddle.Tensor: 注意力输出
        """
        return self._infllmv2_forward(q, k, v, forward_meta, mode="extend")

    def _infllmv2_forward(
        self,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        forward_meta: ForwardMeta,
        mode: str,
    ) -> paddle.Tensor:
        """
        核心InfLLM-V2前向传播逻辑

        Args:
            q: 查询张量
            k: 键张量
            v: 值张量
            forward_meta: 前向传播元数据
            mode: 前向传播模式 ("decode" 或 "extend")

        Returns:
            paddle.Tensor: 注意力输出
        """
        attn_metadata: InfLLMV2AttentionMetadata = forward_meta.attn_metadata

        # 检查是否需要使用稀疏注意力
        seq_len = k.shape[-2]
        if not attn_metadata.should_use_sparse_attention(seq_len):
            # 对于短序列，使用标准注意力
            return self._fallback_standard_attention(q, k, v, forward_meta)

        # Stage 1: Top-K 上下文选择
        topk_indices = self._stage1_topk_selection(q, k, attn_metadata, mode)

        # Stage 2: 稀疏注意力计算
        output = self._stage2_sparse_attention(q, k, v, topk_indices, attn_metadata, mode)

        return output

    def _stage1_topk_selection(
        self,
        q: paddle.Tensor,
        k: paddle.Tensor,
        attn_metadata: InfLLMV2AttentionMetadata,
        mode: str,
    ) -> paddle.Tensor:
        """
        Stage 1: Top-K 上下文选择

        Args:
            q: 查询张量
            k: 键张量
            attn_metadata: 注意力元数据
            mode: 前向传播模式

        Returns:
            paddle.Tensor: TopK选中的块索引
        """
        # 调用底层CUDA算子实现InfLLM-V2 Stage 1
        try:
            # 尝试导入InfLLM-V2 CUDA算子
            from fastdeploy import custom_ops

            # 调用CUDA算子进行Top-K选择
            topk_indices = custom_ops.infllmv2_stage1_topk_selection(
                q, k,
                kernel_size=attn_metadata.kernel_size,
                kernel_stride=attn_metadata.kernel_stride,
                topk=attn_metadata.topk,
                mode=mode,
                block_size=attn_metadata.block_size,
            )

            # 保存到元数据中供Stage 2使用
            attn_metadata.topk_indices = topk_indices

            return topk_indices

        except (ImportError, AttributeError) as e:
            # 如果CUDA算子不可用，回退到CPU实现或抛出错误
            raise RuntimeError(
                f"InfLLM-V2 CUDA kernels not available: {e}. "
                "Please install infllmv2_cuda_impl package."
            )

    def _stage2_sparse_attention(
        self,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        topk_indices: paddle.Tensor,
        attn_metadata: InfLLMV2AttentionMetadata,
        mode: str,
    ) -> paddle.Tensor:
        """
        Stage 2: 稀疏注意力计算

        Args:
            q: 查询张量
            k: 键张量
            v: 值张量
            topk_indices: TopK选中的块索引
            attn_metadata: 注意力元数据
            mode: 前向传播模式

        Returns:
            paddle.Tensor: 稀疏注意力输出
        """
        try:
            # 尝试导入InfLLM-V2 CUDA算子
            from fastdeploy import custom_ops

            # 调用CUDA算子进行稀疏注意力计算
            output = custom_ops.infllmv2_stage2_sparse_attention(
                q, k, v, topk_indices,
                kernel_size=attn_metadata.kernel_size,
                kernel_stride=attn_metadata.kernel_stride,
                topk=attn_metadata.topk,
                mode=mode,
                block_size=attn_metadata.block_size,
            )

            return output

        except (ImportError, AttributeError) as e:
            # 如果CUDA算子不可用，回退到CPU实现或抛出错误
            raise RuntimeError(
                f"InfLLM-V2 CUDA kernels not available: {e}. "
                "Please install infllmv2_cuda_impl package."
            )

    def _fallback_standard_attention(
        self,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        forward_meta: ForwardMeta,
    ) -> paddle.Tensor:
        """
        回退到标准注意力实现

        Args:
            q: 查询张量
            k: 键张量
            v: 值张量
            forward_meta: 前向传播元数据

        Returns:
            paddle.Tensor: 标准注意力输出
        """
        # 使用PaddlePaddle内置的多头注意力
        # 这里简化实现，实际应该调用标准的attention实现
        scale = 1.0 / (self.head_dim ** 0.5)

        # 计算注意力分数
        attn_scores = paddle.matmul(q, k.transpose([0, 1, 3, 2])) * scale

        # 应用softmax
        attn_weights = paddle.nn.functional.softmax(attn_scores, axis=-1)

        # 计算输出
        output = paddle.matmul(attn_weights, v)

        return output