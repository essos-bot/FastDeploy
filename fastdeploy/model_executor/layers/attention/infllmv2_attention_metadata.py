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

# InfLLM-V2 Attention Metadata Implementation
# Reference: https://github.com/OpenBMB/infllmv2_cuda_impl
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import paddle

if TYPE_CHECKING:
    pass


@dataclass
class InfLLMV2AttentionMetadata:
    """
    InfLLM-V2特有的注意力元数据，用于两阶段稀疏注意力机制

    This metadata class contains the necessary information for InfLLM-V2's
    two-stage sparse attention mechanism:
    - Stage 1: Top-K context selection using semantic kernels
    - Stage 2: Sparse attention computation on selected blocks
    """

    # InfLLM-V2 核心参数
    kernel_size: int = 32                    # 语义核大小
    kernel_stride: int = 16                  # 相邻核的步长
    init_blocks: int = 1                     # 初始注意力块数
    block_size: int = 64                     # KV缓存块大小
    window_size: int = 2048                  # 局部滑动窗口大小
    topk: int = 64                           # 每个token计算的相关块数
    use_nope: bool = False                   # 是否使用NOPE技术
    dense_len: int = 8192                    # 使用密集注意力的序列长度阈值

    # Stage 1: Top-K 上下文选择元数据
    kernel_select_mask: Optional[paddle.Tensor] = None    # 动态核选择掩码
    aggregated_scores: Optional[paddle.Tensor] = None    # 聚合后的注意力分数

    # Stage 2: 稀疏注意力计算元数据
    topk_indices: Optional[paddle.Tensor] = None         # TopK相关块索引
    block_mask: Optional[paddle.Tensor] = None           # 块级注意力掩码

    # 序列长度信息
    max_seq_len: int = 0                      # 最大序列长度
    current_seq_len: int = 0                  # 当前序列长度

    # 设备和精度信息
    device: Optional[str] = None              # 计算设备
    dtype: Optional[paddle.dtype] = None      # 数据类型

    def __post_init__(self):
        """初始化后处理，确保参数合理性"""
        if self.kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if self.kernel_stride <= 0:
            raise ValueError("kernel_stride must be positive")
        if self.topk <= 0:
            raise ValueError("topk must be positive")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.dense_len < 0:
            raise ValueError("dense_len cannot be negative")

        # 确保kernel_size和kernel_stride的合理性
        if self.kernel_stride > self.kernel_size:
            raise ValueError("kernel_stride should not exceed kernel_size")

    def should_use_sparse_attention(self, seq_len: int) -> bool:
        """
        判断是否应该使用稀疏注意力

        Args:
            seq_len: 序列长度

        Returns:
            bool: 是否使用稀疏注意力
        """
        if self.dense_len == -1:
            return True  # 总是使用稀疏注意力
        return seq_len > self.dense_len

    def get_num_blocks(self, seq_len: int) -> int:
        """
        计算给定序列长度需要的块数

        Args:
            seq_len: 序列长度

        Returns:
            int: 块数量
        """
        return (seq_len + self.block_size - 1) // self.block_size

    def get_kernel_position(self, block_idx: int) -> int:
        """
        计算给定块索引对应的语义核位置

        Args:
            block_idx: 块索引

        Returns:
            int: 语义核位置
        """
        return block_idx // self.kernel_stride