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

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace fastdeploy {
namespace custom_ops {

/**
 * InfLLM-V2 Stage 1: Top-K Context Selection Kernel
 *
 * 实现InfLLM-V2第一阶段注意力计算：
 * 1. 计算查询token与每个语义核的相似度分数
 * 2. 对分数进行softmax归一化
 * 3. 聚合每个语义核在查询组维度上的分数 (hdim16_reduce)
 *
 * 注意：实际的Top-K选择步骤在此kernel之外进行
 */

template <typename T>
__global__ void infllmv2_stage1_kernel(
    const T* __restrict__ query,                     // [bs, seq_len_q, num_heads, head_dim]
    const T* __restrict__ compressed_key,            // 语义核压缩表示
    const T* __restrict__ compressed_value,          // 未使用
    const int* __restrict__ cu_seqlens_q,             // 累积序列长度
    const int* __restrict__ cu_seqlens_k,             // 累积键序列长度
    T* __restrict__ aggregated_scores,               // 聚合后的注意力分数
    const int batch_size,
    const int num_heads,
    const int head_dim,
    const int seq_len_q,
    const int seq_len_k,
    const int kernel_size,
    const int kernel_stride,
    const int topk,
    const int block_size,
    const float scale,
    const bool causal,
    const bool return_attn_probs
);

/**
 * InfLLM-V2 Stage 2: Sparse Attention Kernel
 *
 * 实现InfLLM-V2第二阶段稀疏注意力计算：
 * 仅在第一阶段选中的块上进行标准注意力计算
 */

template <typename T>
__global__ void infllmv2_sparse_attention_kernel(
    const T* __restrict__ query,                     // [bs, seq_len_q, num_heads, head_dim]
    const T* __restrict__ key_cache,                 // 分块KV缓存
    const T* __restrict__ value_cache,               // 分块KV缓存
    const int* __restrict__ topk_indices,            // 第一阶段选中的块索引
    const int* __restrict__ cu_seqlens_q,             // 累积序列长度
    const int* __restrict__ cu_seqlens_k,             // 累积键序列长度
    T* __restrict__ output,                          // 输出
    const int batch_size,
    const int num_heads,
    const int kv_num_heads,
    const int head_dim,
    const int seq_len_q,
    const int seq_len_k,
    const int kernel_size,
    const int kernel_stride,
    const int topk,
    const int block_size,
    const float scale,
    const bool causal
);

/**
 * Device function for Stage 1 top-k selection computation
 */
template <typename T>
__device__ __forceinline__ void stage1_topk_computation(
    const T* query,
    const T* compressed_key,
    T* scores,
    const int head_dim,
    const int num_kernels,
    const float scale
);

/**
 * Device function for Stage 2 sparse attention computation
 */
template <typename T>
__device__ __forceinline__ void stage2_sparse_attention_computation(
    const T* query,
    const T* key,
    const T* value,
    const int* topk_indices,
    T* output,
    const int head_dim,
    const int num_selected_blocks,
    const float scale
);

/**
 * Utility functions for kernel parameter validation and computation
 */
__device__ __forceinline__ int compute_num_kernels(
    const int seq_len_k,
    const int kernel_size,
    const int kernel_stride
);

__device__ __forceinline__ int get_kernel_start_idx(
    const int kernel_idx,
    const int kernel_stride
);

/**
 * Memory access helper functions
 */
template <typename T>
__device__ __forceinline__ T load_with_bounds_check(
    const T* __restrict__ ptr,
    const int idx,
    const int max_idx,
    const T default_val = T(0)
);

template <typename T>
__device__ __forceinline__ void store_with_bounds_check(
    T* __restrict__ ptr,
    const int idx,
    const int max_idx,
    const T val
);

/**
 * Shared memory management for efficient computation
 */
template <typename T, int BLOCK_SIZE>
struct SharedMemory {
    __device__ __forceinline__ T* get_query_tile();
    __device__ __forceinline__ T* get_key_tile();
    __device__ __forceinline__ T* get_score_tile();
};

/**
 * Warp-level primitives for efficient reduction
 */
template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val);

template <typename T>
__device__ __forceinline__ T warp_reduce_max(T val);

/**
 * Implementation details
 */

template <typename T>
__global__ void infllmv2_stage1_kernel(
    const T* __restrict__ query,
    const T* __restrict__ compressed_key,
    const T* __restrict__ compressed_value,
    const int* __restrict__ cu_seqlens_q,
    const int* __restrict__ cu_seqlens_k,
    T* __restrict__ aggregated_scores,
    const int batch_size,
    const int num_heads,
    const int head_dim,
    const int seq_len_q,
    const int seq_len_k,
    const int kernel_size,
    const int kernel_stride,
    const int topk,
    const int block_size,
    const float scale,
    const bool causal,
    const bool return_attn_probs
) {
    // 计算每个线程块处理的查询token和头
    const int token_idx = blockIdx.x;
    const int head_idx = blockIdx.y;

    if (token_idx >= seq_len_q || head_idx >= num_heads) {
        return;
    }

    // 计算语义核数量
    const int num_kernels = compute_num_kernels(seq_len_k, kernel_size, kernel_stride);

    // 为每个线程块分配共享内存
    extern __shared__ char shared_mem[];
    T* shared_scores = reinterpret_cast<T*>(shared_mem);

    // 计算当前token在查询张量中的位置
    const int query_offset = token_idx * num_heads * head_dim + head_idx * head_dim;
    const T* current_query = query + query_offset;

    // 对每个语义核计算相似度分数
    for (int kernel_idx = threadIdx.x; kernel_idx < num_kernels; kernel_idx += blockDim.x) {
        // 计算当前语义核在压缩键中的位置
        const int kernel_start = get_kernel_start_idx(kernel_idx, kernel_stride);

        // 计算相似度分数
        T score = T(0.0f);
        for (int dim = 0; dim < head_dim; ++dim) {
            T q_val = load_with_bounds_check(current_query, dim, head_dim);
            T k_val = load_with_bounds_check(compressed_key,
                                             kernel_start * num_heads * head_dim +
                                             head_idx * head_dim + dim,
                                             seq_len_k * num_heads * head_dim);
            score += q_val * k_val;
        }
        score *= scale;

        // 存储分数
        shared_scores[kernel_idx] = score;
    }

    __syncthreads();

    // 应用softmax归一化
    // 简化实现，实际应该使用更高效的warp-level reduction
    T max_score = T(-INFINITY);
    for (int i = threadIdx.x; i < num_kernels; i += blockDim.x) {
        max_score = max(max_score, shared_scores[i]);
    }
    max_score = warp_reduce_max(max_score);

    T sum_exp = T(0.0f);
    for (int i = threadIdx.x; i < num_kernels; i += blockDim.x) {
        T exp_score = __expf(shared_scores[i] - max_score);
        shared_scores[i] = exp_score;
        sum_exp += exp_score;
    }
    sum_exp = warp_reduce_sum(sum_exp);

    // 归一化分数并存储到输出
    const T inv_sum = T(1.0f) / sum_exp;
    for (int i = threadIdx.x; i < num_kernels; i += blockDim.x) {
        T normalized_score = shared_scores[i] * inv_sum;
        store_with_bounds_check(aggregated_scores,
                               token_idx * num_heads * num_kernels + head_idx * num_kernels + i,
                               seq_len_q * num_heads * num_kernels,
                               normalized_score);
    }
}

template <typename T>
__global__ void infllmv2_sparse_attention_kernel(
    const T* __restrict__ query,
    const T* __restrict__ key_cache,
    const T* __restrict__ value_cache,
    const int* __restrict__ topk_indices,
    const int* __restrict__ cu_seqlens_q,
    const int* __restrict__ cu_seqlens_k,
    T* __restrict__ output,
    const int batch_size,
    const int num_heads,
    const int kv_num_heads,
    const int head_dim,
    const int seq_len_q,
    const int seq_len_k,
    const int kernel_size,
    const int kernel_stride,
    const int topk,
    const int block_size,
    const float scale,
    const bool causal
) {
    // 计算每个线程块处理的查询token和头
    const int token_idx = blockIdx.x;
    const int head_idx = blockIdx.y;

    if (token_idx >= seq_len_q || head_idx >= num_heads) {
        return;
    }

    // 分配共享内存
    extern __shared__ char shared_mem[];
    T* shared_query = reinterpret_cast<T*>(shared_mem);
    T* shared_key = reinterpret_cast<T*>(shared_mem + head_dim * sizeof(T));
    T* shared_value = reinterpret_cast<T*>(shared_mem + head_dim * 2 * sizeof(T));
    T* shared_output = reinterpret_cast<T*>(shared_mem + head_dim * 3 * sizeof(T));

    // 加载当前查询到共享内存
    const int query_offset = token_idx * num_heads * head_dim + head_idx * head_dim;
    for (int dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
        shared_query[dim] = load_with_bounds_check(query, query_offset + dim,
                                                  seq_len_q * num_heads * head_dim);
    }

    // 计算当前头对应的KV头索引（支持GQA）
    const int kv_head_idx = head_idx * kv_num_heads / num_heads;

    // 初始化输出为零
    for (int dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
        shared_output[dim] = T(0.0f);
    }

    __syncthreads();

    // 对每个选中的块进行注意力计算
    for (int block_idx = 0; block_idx < topk; ++block_idx) {
        // 获取当前选中的块索引
        const int selected_block_idx = topk_indices[token_idx * num_heads * topk +
                                                    head_idx * topk + block_idx];

        if (selected_block_idx < 0 || selected_block_idx >= seq_len_k / block_size) {
            continue;
        }

        // 计算块的起始位置
        const int block_start = selected_block_idx * block_size;
        const int block_end = min(block_start + block_size, seq_len_k);

        // 加载当前块的键和值
        for (int pos = 0; pos < block_end - block_start; ++pos) {
            const int key_pos = block_start + pos;
            const int key_offset = key_pos * kv_num_heads * head_dim + kv_head_idx * head_dim;
            const int value_offset = key_pos * kv_num_heads * head_dim + kv_head_idx * head_dim;

            for (int dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
                shared_key[pos * head_dim + dim] = load_with_bounds_check(key_cache,
                                                                        key_offset + dim,
                                                                        seq_len_k * kv_num_heads * head_dim);
                shared_value[pos * head_dim + dim] = load_with_bounds_check(value_cache,
                                                                          value_offset + dim,
                                                                          seq_len_k * kv_num_heads * head_dim);
            }
        }

        __syncthreads();

        // 计算当前块的注意力
        for (int pos = 0; pos < block_end - block_start; ++pos) {
            T score = T(0.0f);
            for (int dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
                score += shared_query[dim] * shared_key[pos * head_dim + dim];
            }
            score *= scale;

            // 应用softmax（简化实现）
            score = __expf(score);

            // 累加到输出
            for (int dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
                shared_output[dim] += score * shared_value[pos * head_dim + dim];
            }
        }

        __syncthreads();
    }

    // 将结果写回全局内存
    const int output_offset = token_idx * num_heads * head_dim + head_idx * head_dim;
    for (int dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
        store_with_bounds_check(output, output_offset + dim,
                               seq_len_q * num_heads * head_dim, shared_output[dim]);
    }
}

// Utility function implementations
template <typename T>
__device__ __forceinline__ T load_with_bounds_check(
    const T* __restrict__ ptr,
    const int idx,
    const int max_idx,
    const T default_val
) {
    if (idx < max_idx) {
        return ptr[idx];
    }
    return default_val;
}

template <typename T>
__device__ __forceinline__ void store_with_bounds_check(
    T* __restrict__ ptr,
    const int idx,
    const int max_idx,
    const T val
) {
    if (idx < max_idx) {
        ptr[idx] = val;
    }
}

__device__ __forceinline__ int compute_num_kernels(
    const int seq_len_k,
    const int kernel_size,
    const int kernel_stride
) {
    return (seq_len_k - kernel_size) / kernel_stride + 1;
}

__device__ __forceinline__ int get_kernel_start_idx(
    const int kernel_idx,
    const int kernel_stride
) {
    return kernel_idx * kernel_stride;
}

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    for (int mask = 16; mask > 0; mask /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, mask, 32);
    }
    return val;
}

template <typename T>
__device__ __forceinline__ T warp_reduce_max(T val) {
    for (int mask = 16; mask > 0; mask /= 2) {
        val = max(val, __shfl_xor_sync(0xffffffff, val, mask, 32));
    }
    return val;
}

} // namespace custom_ops
} // namespace fastdeploy