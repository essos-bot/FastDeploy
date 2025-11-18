"""
MiniCPM4.1-8B Performance Benchmark

This module provides comprehensive performance benchmarking for MiniCPM4.1-8B model
in FastDeploy, including latency, throughput, memory usage, and accuracy tests.
"""

import gc
import json
import time
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import paddle

from fastdeploy import FDConfig, LLM, SamplingParams
from fastdeploy.config.minicpm41_config import (
    MiniCPM41ModelConfig,
    create_quantized_minicpm41_config,
)


@dataclass
class BenchmarkConfig:
    """Configuration for performance benchmark"""
    model_path: str = "openbmb/MiniCPM4.1-8B"
    quant_type: Optional[str] = None
    tensor_parallel_size: int = 1
    max_batch_size: int = 8
    max_sequence_length: int = 4096
    warmup_steps: int = 3
    benchmark_steps: int = 10
    output_file: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Results from performance benchmark"""
    model_name: str
    quant_type: str
    tensor_parallel_size: int

    # Latency metrics (ms)
    latency_mean: float
    latency_p50: float
    latency_p90: float
    latency_p95: float
    latency_p99: float

    # Throughput metrics
    tokens_per_second: float
    requests_per_second: float

    # Memory metrics
    peak_memory_gb: float
    model_memory_gb: float

    # Accuracy metrics (if available)
    accuracy_score: Optional[float] = None

    # Additional info
    batch_size: int
    sequence_length: int
    total_time: float


class MiniCPM41PerformanceBenchmark:
    """Performance benchmark for MiniCPM4.1-8B"""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = []

    def setup_model(self) -> LLM:
        """Set up the model for benchmarking"""
        print(f"Setting up model: {self.config.model_path}")

        # Create base model config
        model_config = MiniCPM41ModelConfig(
            max_position_embeddings=self.config.max_sequence_length,
            use_cache=True,
        )

        # Create FDConfig with quantization if specified
        if self.config.quant_type:
            fd_config, quant_config = create_quantized_minicpm41_config(
                quant_type=self.config.quant_type
            )
            fd_config.model_config = model_config
            fd_config.parallel_config.tensor_parallel_size = self.config.tensor_parallel_size
        else:
            fd_config = FDConfig(
                model_config=model_config,
                parallel_config=FDConfig.ParallelConfig(
                    tensor_parallel_size=self.config.tensor_parallel_size
                )
            )

        # Create LLM instance
        llm = LLM(model=self.config.model_path, fd_config=fd_config)

        return llm

    def prepare_test_data(self, batch_size: int, seq_len: int) -> List[str]:
        """Prepare test prompts for benchmarking"""
        # Create prompts of varying lengths
        base_prompt = "请详细介绍人工智能在以下领域的应用："
        prompts = []

        for i in range(batch_size):
            # Create prompts with different lengths
            length_factor = 0.5 + (i / batch_size) * 1.5  # 0.5x to 2x length
            target_length = int(seq_len * length_factor)

            # Generate prompt by repeating the base phrase
            prompt = base_prompt
            while len(prompt) < target_length:
                prompt += " 机器学习、深度学习、自然语言处理、计算机视觉、强化学习"

            prompts.append(prompt[:target_length])

        return prompts

    def measure_memory_usage(self) -> Tuple[float, float]:
        """Measure current memory usage (model memory and peak memory)"""
        paddle.device.cuda.empty_cache()

        # Get current memory usage
        memory_info = paddle.device.cuda.memory_info()
        current_memory_gb = memory_info['Current'] / (1024**3)
        max_memory_gb = memory_info['Max'] / (1024**3)

        return current_memory_gb, max_memory_gb

    def benchmark_latency(
        self,
        llm: LLM,
        prompts: List[str],
        max_tokens: int = 256
    ) -> Dict[str, float]:
        """Benchmark inference latency"""
        print(f"Benchmarking latency for {len(prompts)} prompts...")

        latencies = []

        # Warmup
        print("Running warmup...")
        for i in range(self.config.warmup_steps):
            try:
                _ = llm.generate(prompts[:1], max_tokens=max_tokens)
            except Exception as e:
                print(f"Warmup step {i} failed: {e}")

        # Benchmark
        print("Running latency benchmark...")
        for i in range(self.config.benchmark_steps):
            try:
                start_time = time.time()
                _ = llm.generate(prompts, max_tokens=max_tokens)
                end_time = time.time()

                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)

                print(f"  Step {i+1}: {latency_ms:.2f}ms")

            except Exception as e:
                print(f"Benchmark step {i} failed: {e}")
                continue

        if not latencies:
            raise RuntimeError("All latency benchmark steps failed")

        # Calculate statistics
        latencies = np.array(latencies)
        return {
            "mean": float(np.mean(latencies)),
            "p50": float(np.percentile(latencies, 50)),
            "p90": float(np.percentile(latencies, 90)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
            "std": float(np.std(latencies)),
        }

    def benchmark_throughput(
        self,
        llm: LLM,
        prompts: List[str],
        max_tokens: int = 256
    ) -> Dict[str, float]:
        """Benchmark inference throughput"""
        print(f"Benchmarking throughput...")

        # Measure total tokens and time
        total_input_tokens = sum(len(p.split()) for p in prompts)  # Approximate
        total_output_tokens = len(prompts) * max_tokens
        total_tokens = total_input_tokens + total_output_tokens

        start_time = time.time()

        try:
            outputs = llm.generate(prompts, max_tokens=max_tokens)
            end_time = time.time()

            # Calculate throughput
            total_time = end_time - start_time
            tokens_per_second = total_tokens / total_time
            requests_per_second = len(prompts) / total_time

            return {
                "tokens_per_second": tokens_per_second,
                "requests_per_second": requests_per_second,
                "total_tokens": total_tokens,
                "total_time": total_time,
            }

        except Exception as e:
            print(f"Throughput benchmark failed: {e}")
            return {
                "tokens_per_second": 0.0,
                "requests_per_second": 0.0,
                "total_tokens": 0,
                "total_time": 0.0,
            }

    def run_single_benchmark(
        self,
        batch_size: int,
        sequence_length: int
    ) -> BenchmarkResult:
        """Run a single benchmark with specified batch size and sequence length"""
        print(f"\nRunning benchmark: batch_size={batch_size}, seq_len={sequence_length}")

        # Setup model
        llm = self.setup_model()

        # Prepare test data
        prompts = self.prepare_test_data(batch_size, sequence_length)

        # Measure initial memory
        initial_memory, _ = self.measure_memory_usage()

        # Benchmark latency
        latency_stats = self.benchmark_latency(llm, prompts)

        # Benchmark throughput
        throughput_stats = self.benchmark_throughput(llm, prompts)

        # Measure peak memory
        _, peak_memory = self.measure_memory_usage()

        # Calculate model memory (peak - initial)
        model_memory = peak_memory - initial_memory

        # Create result
        result = BenchmarkResult(
            model_name=self.config.model_path,
            quant_type=self.config.quant_type or "none",
            tensor_parallel_size=self.config.tensor_parallel_size,
            latency_mean=latency_stats["mean"],
            latency_p50=latency_stats["p50"],
            latency_p90=latency_stats["p90"],
            latency_p95=latency_stats["p95"],
            latency_p99=latency_stats["p99"],
            tokens_per_second=throughput_stats["tokens_per_second"],
            requests_per_second=throughput_stats["requests_per_second"],
            peak_memory_gb=peak_memory,
            model_memory_gb=model_memory,
            batch_size=batch_size,
            sequence_length=sequence_length,
            total_time=throughput_stats["total_time"],
        )

        # Cleanup
        del llm
        paddle.device.cuda.empty_cache()
        gc.collect()

        return result

    def run_full_benchmark(self) -> List[BenchmarkResult]:
        """Run comprehensive benchmark with multiple configurations"""
        print("Starting comprehensive MiniCPM4.1-8B performance benchmark")
        print("=" * 60)

        # Test configurations
        test_configs = [
            (1, 512),    # Single request, short
            (1, 2048),   # Single request, medium
            (1, 4096),   # Single request, long
            (4, 512),    # Small batch, short
            (4, 2048),   # Small batch, medium
            (8, 512),    # Larger batch, short
        ]

        results = []

        for batch_size, seq_len in test_configs:
            try:
                result = self.run_single_benchmark(batch_size, seq_len)
                results.append(result)

                # Print summary
                print(f"✓ Completed: batch={batch_size}, seq_len={seq_len}")
                print(f"  Latency: {result.latency_mean:.2f}ms")
                print(f"  Throughput: {result.tokens_per_second:.1f} tokens/s")
                print(f"  Memory: {result.model_memory_gb:.2f}GB")

            except Exception as e:
                print(f"✗ Failed benchmark for batch={batch_size}, seq_len={seq_len}")
                print(f"  Error: {e}")
                traceback.print_exc()
                continue

        self.results = results
        return results

    def compare_quantizations(self) -> Dict[str, List[BenchmarkResult]]:
        """Compare performance across different quantization types"""
        print("Running quantization comparison benchmark...")

        quant_types = [None, "wint8", "w4a8", "w4afp8"]  # None = FP16
        comparison_results = {}

        original_quant_type = self.config.quant_type

        for quant_type in quant_types:
            print(f"\nTesting quantization: {quant_type or 'FP16'}")

            self.config.quant_type = quant_type

            try:
                # Run a standard benchmark
                results = self.run_single_benchmark(batch_size=4, sequence_length=1024)
                comparison_results[quant_type or "fp16"] = [results]

                print(f"  ✓ {quant_type or 'FP16'}: {results.latency_mean:.2f}ms, {results.tokens_per_second:.1f} tokens/s")

            except Exception as e:
                print(f"  ✗ Failed: {e}")
                comparison_results[quant_type or "fp16"] = []

        # Restore original quant type
        self.config.quant_type = original_quant_type

        return comparison_results

    def save_results(self, results: List[BenchmarkResult], filename: str):
        """Save benchmark results to file"""
        # Convert results to dict for JSON serialization
        results_dict = []
        for result in results:
            results_dict.append({
                "model_name": result.model_name,
                "quant_type": result.quant_type,
                "tensor_parallel_size": result.tensor_parallel_size,
                "latency_mean": result.latency_mean,
                "latency_p50": result.latency_p50,
                "latency_p90": result.latency_p90,
                "latency_p95": result.latency_p95,
                "latency_p99": result.latency_p99,
                "tokens_per_second": result.tokens_per_second,
                "requests_per_second": result.requests_per_second,
                "peak_memory_gb": result.peak_memory_gb,
                "model_memory_gb": result.model_memory_gb,
                "batch_size": result.batch_size,
                "sequence_length": result.sequence_length,
                "total_time": result.total_time,
                "accuracy_score": result.accuracy_score,
            })

        # Save to file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)

        print(f"Results saved to {filename}")

    def print_summary(self, results: List[BenchmarkResult]):
        """Print benchmark summary"""
        if not results:
            print("No results to display")
            return

        print("\n" + "=" * 80)
        print("MINICPM4.1-8B PERFORMANCE BENCHMARK SUMMARY")
        print("=" * 80)

        # Group by quantization type
        quant_groups = {}
        for result in results:
            key = result.quant_type
            if key not in quant_groups:
                quant_groups[key] = []
            quant_groups[key].append(result)

        for quant_type, group_results in quant_groups.items():
            print(f"\n{quant_type.upper()} QUANTIZATION:")
            print("-" * 40)

            # Calculate averages
            avg_latency = np.mean([r.latency_mean for r in group_results])
            avg_throughput = np.mean([r.tokens_per_second for r in group_results])
            avg_memory = np.mean([r.model_memory_gb for r in group_results])

            print(f"  Average Latency:     {avg_latency:.2f}ms")
            print(f"  Average Throughput:  {avg_throughput:.1f} tokens/s")
            print(f"  Average Memory:      {avg_memory:.2f}GB")

            # Find best performing configuration
            best_result = min(group_results, key=lambda x: x.latency_mean)
            print(f"  Best Latency:        {best_result.latency_mean:.2f}ms "
                  f"(batch={best_result.batch_size}, seq_len={best_result.sequence_length})")

        # Overall comparison
        print(f"\nQUANTIZATION COMPARISON:")
        print("-" * 40)

        quant_summary = []
        for quant_type, group_results in quant_groups.items():
            if group_results:
                avg_latency = np.mean([r.latency_mean for r in group_results])
                avg_throughput = np.mean([r.tokens_per_second for r in group_results])
                avg_memory = np.mean([r.model_memory_gb for r in group_results])

                quant_summary.append({
                    "type": quant_type,
                    "latency": avg_latency,
                    "throughput": avg_throughput,
                    "memory": avg_memory,
                })

        # Sort by latency
        quant_summary.sort(key=lambda x: x["latency"])

        print(f"{'Quantization':<12} {'Latency':<12} {'Throughput':<15} {'Memory':<10}")
        print(f"{'-'*12:<12} {'-'*12:<12} {'-'*15:<15} {'-'*10:<10}")

        for summary in quant_summary:
            print(f"{summary['type']:<12} {summary['latency']:<12.2f} "
                  f"{summary['throughput']:<15.1f} {summary['memory']:<10.2f}")


def run_default_benchmark():
    """Run default performance benchmark"""
    config = BenchmarkConfig(
        model_path="openbmb/MiniCPM4.1-8B",
        max_batch_size=4,
        max_sequence_length=2048,
        warmup_steps=2,
        benchmark_steps=5,
        output_file="minicpm41_benchmark_results.json",
    )

    benchmark = MiniCPM41PerformanceBenchmark(config)

    try:
        # Run full benchmark
        results = benchmark.run_full_benchmark()

        # Print summary
        benchmark.print_summary(results)

        # Save results
        if config.output_file:
            benchmark.save_results(results, config.output_file)

        # Run quantization comparison
        print("\n" + "=" * 80)
        print("RUNNING QUANTIZATION COMPARISON...")
        print("=" * 80)

        comparison_results = benchmark.compare_quantizations()

        return results

    except Exception as e:
        print(f"Benchmark failed: {e}")
        traceback.print_exc()
        return []


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MiniCPM4.1-8B Performance Benchmark")
    parser.add_argument("--model", type=str, default="openbmb/MiniCPM4.1-8B",
                       help="Model path or identifier")
    parser.add_argument("--quant", type=str, choices=["wint8", "w4a8", "w4afp8"],
                       help="Quantization type")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Maximum batch size")
    parser.add_argument("--seq-len", type=int, default=2048,
                       help="Maximum sequence length")
    parser.add_argument("--tp-size", type=int, default=1,
                       help="Tensor parallel size")
    parser.add_argument("--output", type=str,
                       help="Output file for results")
    parser.add_argument("--compare-quant", action="store_true",
                       help="Run quantization comparison")

    args = parser.parse_args()

    # Create benchmark config
    config = BenchmarkConfig(
        model_path=args.model,
        quant_type=args.quant,
        tensor_parallel_size=args.tp_size,
        max_batch_size=args.batch_size,
        max_sequence_length=args.seq_len,
        output_file=args.output,
    )

    # Run benchmark
    benchmark = MiniCPM41PerformanceBenchmark(config)

    if args.compare_quant:
        # Run quantization comparison
        results = benchmark.compare_quantizations()

        # Print results
        for quant_type, result_list in results.items():
            if result_list:
                result = result_list[0]
                print(f"{quant_type}: {result.latency_mean:.2f}ms, "
                      f"{result.tokens_per_second:.1f} tokens/s")
    else:
        # Run standard benchmark
        results = benchmark.run_full_benchmark()
        benchmark.print_summary(results)

        if config.output_file:
            benchmark.save_results(results, config.output_file)