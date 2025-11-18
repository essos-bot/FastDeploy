/*
 * Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include <vector>

#include "infllmv2_impl.cuh"

namespace fastdeploy {
namespace custom_ops {

#define DISPATCH_CASE_BOTH_TYPES(...) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define DISPATCH_BOTH_TYPES(CONTAINER, ...) \
  AT_DISPATCH_TYPE(CONTAINER, DISPATCH_CASE_BOTH_TYPES, __VA_ARGS__)

/**
 * Stage 1: Top-K context selection for InfLLM-V2
 *
 * Computes relevance scores between query tokens and semantic kernels,
 * performs softmax normalization and aggregation across query group dimension.
 *
 * Args:
 *   query: Query tensor [batch_size, seq_len_q, num_heads, head_dim]
 *   compressed_key: Compressed key tensor representing semantic kernels
 *   cu_seqlens_q: Cumulative sequence lengths for queries
 *   cu_seqlens_k: Cumulative sequence lengths for keys
 *   kernel_size: Size of semantic kernels
 *   kernel_stride: Stride between adjacent kernels
 *   topk: Number of top-k blocks to select
 *   block_size: Size of KV cache blocks
 *   causal: Whether to apply causal masking
 *   return_attn_probs: Whether to return attention probabilities
 *
 * Returns:
 *   aggregated_scores: Aggregated attention scores for subsequent Top-K selection
 */
torch::Tensor infllmv2_stage1_topk_selection(
    const torch::Tensor& query,
    const torch::Tensor& compressed_key,
    const int kernel_size,
    const int kernel_stride,
    const int topk,
    const int block_size,
    const bool causal = true,
    const bool return_attn_probs = true
) {
    TORCH_CHECK(query.dim() == 4, "Expected query to be 4D tensor");
    TORCH_CHECK(compressed_key.dim() == 4, "Expected compressed_key to be 4D tensor");
    TORCH_CHECK(query.size(0) == compressed_key.size(0), "Batch sizes must match");
    TORCH_CHECK(query.size(2) == compressed_key.size(2), "Number of heads must match");
    TORCH_CHECK(query.size(3) == compressed_key.size(3), "Head dimensions must match");

    const int batch_size = query.size(0);
    const int seq_len_q = query.size(1);
    const int num_heads = query.size(2);
    const int head_dim = query.size(3);
    const int seq_len_k = compressed_key.size(1);

    // 计算语义核数量
    const int num_kernels = (seq_len_k - kernel_size) / kernel_stride + 1;

    // 创建输出张量
    auto options = torch::TensorOptions()
        .dtype(query.dtype())
        .device(query.device());
    auto aggregated_scores = torch::empty({batch_size, seq_len_q, num_heads, num_kernels}, options);

    // 计算scale
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // 分配累积序列长度张量（简化实现，实际应该根据输入参数生成）
    auto cu_seqlens_q = torch::arange(0, (batch_size + 1) * seq_len_q, seq_len_q,
                                     torch::TensorOptions().dtype(torch::kInt32).device(query.device()));
    auto cu_seqlens_k = torch::arange(0, (batch_size + 1) * seq_len_k, seq_len_k,
                                     torch::TensorOptions().dtype(torch::kInt32).device(query.device()));

    // 根据数据类型分派CUDA kernel
    DISPATCH_BOTH_TYPES(query.scalar_type(), "infllmv2_stage1_topk_selection", ([&] {
        using scalar_t = torch::kFloat16 == query.scalar_type() ? half : nv_bfloat16;

        // 设置CUDA kernel配置
        const int threads_per_block = 256;
        const dim3 grid_dim(seq_len_q, num_heads);
        const dim3 block_dim(threads_per_block);

        // 计算共享内存大小
        const size_t shared_mem_size = num_kernels * sizeof(scalar_t);

        // 启动kernel
        infllmv2_stage1_kernel<scalar_t><<<grid_dim, block_dim, shared_mem_size, query.device().index()>>>(
            // 输入
            reinterpret_cast<const scalar_t*>(query.data_ptr()),
            reinterpret_cast<const scalar_t*>(compressed_key.data_ptr()),
            nullptr, // compressed_value not used in stage 1

            // 累积序列长度
            reinterpret_cast<const int*>(cu_seqlens_q.data_ptr()),
            reinterpret_cast<const int*>(cu_seqlens_k.data_ptr()),

            // 输出
            reinterpret_cast<scalar_t*>(aggregated_scores.data_ptr()),

            // 参数
            batch_size,
            num_heads,
            head_dim,
            seq_len_q,
            seq_len_k,
            kernel_size,
            kernel_stride,
            topk,
            block_size,
            scale,
            causal,
            return_attn_probs
        );
    }));

    // 同步CUDA流
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return aggregated_scores;
}

/**
 * Stage 2: Sparse attention computation for InfLLM-V2
 *
 * Performs standard attention computation only on blocks selected in Stage 1.
 *
 * Args:
 *   query: Query tensor [batch_size, seq_len_q, num_heads, head_dim]
 *   key_cache: Key cache tensor [batch_size, seq_len_k, kv_num_heads, head_dim]
 *   value_cache: Value cache tensor [batch_size, seq_len_k, kv_num_heads, head_dim]
 *   topk_indices: Selected block indices from Stage 1
 *   kernel_size: Size of semantic kernels
 *   kernel_stride: Stride between adjacent kernels
 *   topk: Number of selected blocks
 *   block_size: Size of KV cache blocks
 *   causal: Whether to apply causal masking
 *
 * Returns:
 *   output: Sparse attention output tensor
 */
torch::Tensor infllmv2_stage2_sparse_attention(
    const torch::Tensor& query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& topk_indices,
    const int kernel_size,
    const int kernel_stride,
    const int topk,
    const int block_size,
    const bool causal = true
) {
    TORCH_CHECK(query.dim() == 4, "Expected query to be 4D tensor");
    TORCH_CHECK(key_cache.dim() == 4, "Expected key_cache to be 4D tensor");
    TORCH_CHECK(value_cache.dim() == 4, "Expected value_cache to be 4D tensor");
    TORCH_CHECK(topk_indices.dim() == 4, "Expected topk_indices to be 4D tensor");
    TORCH_CHECK(query.size(0) == key_cache.size(0), "Batch sizes must match");
    TORCH_CHECK(key_cache.size(0) == value_cache.size(0), "Batch sizes must match");

    const int batch_size = query.size(0);
    const int seq_len_q = query.size(1);
    const int num_heads = query.size(2);
    const int head_dim = query.size(3);
    const int seq_len_k = key_cache.size(1);
    const int kv_num_heads = key_cache.size(2);

    // 创建输出张量
    auto options = torch::TensorOptions()
        .dtype(query.dtype())
        .device(query.device());
    auto output = torch::empty({batch_size, seq_len_q, num_heads, head_dim}, options);

    // 计算scale
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // 分配累积序列长度张量（简化实现）
    auto cu_seqlens_q = torch::arange(0, (batch_size + 1) * seq_len_q, seq_len_q,
                                     torch::TensorOptions().dtype(torch::kInt32).device(query.device()));
    auto cu_seqlens_k = torch::arange(0, (batch_size + 1) * seq_len_k, seq_len_k,
                                     torch::TensorOptions().dtype(torch::kInt32).device(query.device()));

    // 根据数据类型分派CUDA kernel
    DISPATCH_BOTH_TYPES(query.scalar_type(), "infllmv2_stage2_sparse_attention", ([&] {
        using scalar_t = torch::kFloat16 == query.scalar_type() ? half : nv_bfloat16;

        // 设置CUDA kernel配置
        const int threads_per_block = 256;
        const dim3 grid_dim(seq_len_q, num_heads);
        const dim3 block_dim(threads_per_block);

        // 计算共享内存大小
        const size_t shared_mem_size = head_dim * 3 * sizeof(scalar_t) +  // query, key, value
                                       block_size * head_dim * 2 * sizeof(scalar_t) +  // key, value tiles
                                       head_dim * sizeof(scalar_t);  // output

        // 启动kernel
        infllmv2_sparse_attention_kernel<scalar_t><<<grid_dim, block_dim, shared_mem_size, query.device().index()>>>(
            // 输入
            reinterpret_cast<const scalar_t*>(query.data_ptr()),
            reinterpret_cast<const scalar_t*>(key_cache.data_ptr()),
            reinterpret_cast<const scalar_t*>(value_cache.data_ptr()),
            reinterpret_cast<const int*>(topk_indices.data_ptr()),

            // 累积序列长度
            reinterpret_cast<const int*>(cu_seqlens_q.data_ptr()),
            reinterpret_cast<const int*>(cu_seqlens_k.data_ptr()),

            // 输出
            reinterpret_cast<scalar_t*>(output.data_ptr()),

            // 参数
            batch_size,
            num_heads,
            kv_num_heads,
            head_dim,
            seq_len_q,
            seq_len_k,
            kernel_size,
            kernel_stride,
            topk,
            block_size,
            scale,
            causal
        );
    }));

    // 同步CUDA流
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}

/**
 * Python bindings for InfLLM-V2 attention operators
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("infllmv2_stage1_topk_selection", &infllmv2_stage1_topk_selection,
          "InfLLM-V2 Stage 1: Top-K context selection",
          py::arg("query"),
          py::arg("compressed_key"),
          py::arg("kernel_size"),
          py::arg("kernel_stride"),
          py::arg("topk"),
          py::arg("block_size"),
          py::arg("causal") = true,
          py::arg("return_attn_probs") = true);

    m.def("infllmv2_stage2_sparse_attention", &infllmv2_stage2_sparse_attention,
          "InfLLM-V2 Stage 2: Sparse attention computation",
          py::arg("query"),
          py::arg("key_cache"),
          py::arg("value_cache"),
          py::arg("topk_indices"),
          py::arg("kernel_size"),
          py::arg("kernel_stride"),
          py::arg("topk"),
          py::arg("block_size"),
          py::arg("causal") = true);
}

} // namespace custom_ops
} // namespace fastdeploy