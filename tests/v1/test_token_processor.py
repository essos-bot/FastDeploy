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

import os
import sys
import time
import unittest
from collections import Counter
from unittest.mock import Mock, patch

import numpy as np

# Mock paddle module before any imports that might use it
try:
    import paddle
except ImportError:
    # Create a minimal mock paddle module for testing
    class MockPaddle:
        class device:
            @staticmethod
            def set_device(device):
                pass

        class Tensor:
            def __init__(self, *args, **kwargs):
                pass

            def numpy(self):
                return np.array([])

            def item(self):
                return 0

        @staticmethod
        def full(shape, fill_value, dtype="float32"):
            return MockPaddle.Tensor()

    paddle = MockPaddle()

# Determine import method based on environment
# Use environment variable FD_TEST_MODE=standalone for local testing
TEST_MODE = os.environ.get("FD_TEST_MODE", "normal")

if TEST_MODE == "standalone":
    # Local testing mode - use dynamic import
    # Mock the logger to avoid import issues
    mock_logger = Mock()

    # Create a mock module structure
    class MockUtils:
        llm_logger = mock_logger
        spec_logger = mock_logger

    class MockEnvs:
        FD_USE_GET_SAVE_OUTPUT_V1 = False
        ENABLE_V1_KVCACHE_SCHEDULER = False
        FD_DEBUG = False
        FD_ENABLE_INTERNAL_ADAPTER = False

    class MockMetrics:
        def observe(self, *args, **kwargs):
            pass

        def set(self, *args, **kwargs):
            pass

        def inc(self, *args, **kwargs):
            pass

        def dec(self, *args, **kwargs):
            pass

    class MockMainMetrics:
        request_prefill_time = MockMetrics()
        time_per_output_token = MockMetrics()
        generation_tokens_total = MockMetrics()
        first_token_latency = MockMetrics()
        time_to_first_token = MockMetrics()
        request_queue_time = MockMetrics()
        request_decode_time = MockMetrics()
        num_requests_running = MockMetrics()
        request_success_total = MockMetrics()
        infer_latency = MockMetrics()
        request_inference_time = MockMetrics()
        request_generation_tokens = MockMetrics()
        available_gpu_block_num = MockMetrics()
        batch_size = MockMetrics()
        available_batch_size = MockMetrics()

    sys.modules["fastdeploy"] = Mock()
    sys.modules["fastdeploy.utils"] = MockUtils()
    sys.modules["fastdeploy.envs"] = MockEnvs()
    sys.modules["fastdeploy.metrics"] = Mock()
    sys.modules["fastdeploy.metrics.metrics"] = Mock()
    sys.modules["fastdeploy.metrics.metrics"].main_process_metrics = MockMainMetrics()
    sys.modules["fastdeploy.platforms"] = Mock()
    sys.modules["fastdeploy.inter_communicator"] = Mock()
    sys.modules["fastdeploy.worker"] = Mock()
    sys.modules["fastdeploy.engine"] = Mock()

    # Mock paddle device setting
    with patch.object(paddle.device, "set_device"):
        # Import the token processor module directly
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "token_processor", os.path.join(os.path.dirname(__file__), "../../fastdeploy/output/token_processor.py")
        )
        token_processor_module = importlib.util.module_from_spec(spec)
        token_processor_module.llm_logger = mock_logger
        token_processor_module.spec_logger = mock_logger
        spec.loader.exec_module(token_processor_module)

        # Extract the classes we want to test
        TokenProcessor = token_processor_module.TokenProcessor
        WarmUpTokenProcessor = token_processor_module.WarmUpTokenProcessor
        RECOVERY_STOP_SIGNAL = token_processor_module.RECOVERY_STOP_SIGNAL
else:
    # Normal mode - direct import (for CI/CD and production)
    try:
        from fastdeploy.output.token_processor import (
            RECOVERY_STOP_SIGNAL,
            TokenProcessor,
            WarmUpTokenProcessor,
        )

        # If we can import directly, we don't need mocking
        mock_logger = None
    except ImportError:
        # Fallback to standalone mode if direct import fails
        print("Warning: Direct import failed, falling back to standalone mode")
        TEST_MODE = "standalone"
        # Re-run the standalone setup
        mock_logger = Mock()

        class MockUtils:
            llm_logger = mock_logger
            spec_logger = mock_logger

        class MockEnvs:
            FD_USE_GET_SAVE_OUTPUT_V1 = False
            ENABLE_V1_KVCACHE_SCHEDULER = False
            FD_DEBUG = False
            FD_ENABLE_INTERNAL_ADAPTER = False

        class MockMetrics:
            def observe(self, *args, **kwargs):
                pass

            def set(self, *args, **kwargs):
                pass

            def inc(self, *args, **kwargs):
                pass

            def dec(self, *args, **kwargs):
                pass

        class MockMainMetrics:
            request_prefill_time = MockMetrics()
            time_per_output_token = MockMetrics()
            generation_tokens_total = MockMetrics()
            first_token_latency = MockMetrics()
            time_to_first_token = MockMetrics()
            request_queue_time = MockMetrics()
            request_decode_time = MockMetrics()
            num_requests_running = MockMetrics()
            request_success_total = MockMetrics()
            infer_latency = MockMetrics()
            request_inference_time = MockMetrics()
            request_generation_tokens = MockMetrics()
            available_gpu_block_num = MockMetrics()
            batch_size = MockMetrics()
            available_batch_size = MockMetrics()

        sys.modules["fastdeploy"] = Mock()
        sys.modules["fastdeploy.utils"] = MockUtils()
        sys.modules["fastdeploy.envs"] = MockEnvs()
        sys.modules["fastdeploy.metrics"] = Mock()
        sys.modules["fastdeploy.metrics.metrics"] = Mock()
        sys.modules["fastdeploy.metrics.metrics"].main_process_metrics = MockMainMetrics()
        sys.modules["fastdeploy.platforms"] = Mock()
        sys.modules["fastdeploy.inter_communicator"] = Mock()
        sys.modules["fastdeploy.worker"] = Mock()
        sys.modules["fastdeploy.engine"] = Mock()

        with patch.object(paddle.device, "set_device"):
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "token_processor",
                os.path.join(os.path.dirname(__file__), "../../fastdeploy/output/token_processor.py"),
            )
            token_processor_module = importlib.util.module_from_spec(spec)
            token_processor_module.llm_logger = mock_logger
            token_processor_module.spec_logger = mock_logger
            spec.loader.exec_module(token_processor_module)

            TokenProcessor = token_processor_module.TokenProcessor
            WarmUpTokenProcessor = token_processor_module.WarmUpTokenProcessor
            RECOVERY_STOP_SIGNAL = token_processor_module.RECOVERY_STOP_SIGNAL


class TestTokenProcessor(unittest.TestCase):
    """Test cases for TokenProcessor class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock configuration
        self.mock_cfg = Mock()
        self.mock_cfg.parallel_config = Mock()
        self.mock_cfg.parallel_config.local_data_parallel_id = 0
        self.mock_cfg.parallel_config.enable_expert_parallel = False
        self.mock_cfg.parallel_config.data_parallel_size = 1
        self.mock_cfg.speculative_config = Mock()
        self.mock_cfg.speculative_config.method = None
        self.mock_cfg.speculative_config.num_speculative_tokens = 5
        self.mock_cfg.model_config = Mock()
        self.mock_cfg.model_config.enable_logprob = False
        self.mock_cfg.scheduler_config = Mock()
        self.mock_cfg.scheduler_config.name = "splitwise"
        self.mock_cfg.max_num_seqs = 100
        self.mock_cfg.splitwise_version = "v1"

        # Create mock dependencies
        self.mock_cached_tokens = Mock()
        self.mock_engine_worker_queue = Mock()
        self.mock_split_connector = Mock()

        # Mock platform detection
        self.mock_platform = Mock()
        self.mock_platform.is_xpu.return_value = False
        self.mock_platform.is_iluvatar.return_value = False
        self.mock_platform.is_gcu.return_value = False
        self.mock_platform.is_intel_hpu.return_value = False

    @patch.object(paddle.device, "set_device")
    def test_token_processor_initialization(self, mock_set_device):
        """Test TokenProcessor initialization."""
        with patch("fastdeploy.output.token_processor.envs") as mock_envs:
            mock_envs.FD_USE_GET_SAVE_OUTPUT_V1 = False

            processor = TokenProcessor(
                self.mock_cfg, self.mock_cached_tokens, self.mock_engine_worker_queue, self.mock_split_connector
            )

            # Check basic attributes
            self.assertEqual(processor.cfg, self.mock_cfg)
            self.assertEqual(processor.cached_generated_tokens, self.mock_cached_tokens)
            self.assertEqual(processor.engine_worker_queue, self.mock_engine_worker_queue)
            self.assertEqual(processor.split_connector, self.mock_split_connector)
            self.assertIsInstance(processor.tokens_counter, Counter)

            # Check paddle device was set to CPU
            mock_set_device.assert_called_once_with("cpu")

            # Check output tensors are created
            self.assertIsInstance(processor.output_tokens, paddle.Tensor)

            # Check statistics attributes
            self.assertEqual(processor.number_of_tasks, 0)
            self.assertEqual(processor.number_of_input_tokens, 0)
            self.assertEqual(processor.number_of_output_tokens, 0)

    def test_token_processor_with_zmq_initialization(self):
        """Test TokenProcessor initialization with ZMQ enabled."""
        with (
            patch("fastdeploy.output.token_processor.envs") as mock_envs,
            patch("fastdeploy.output.token_processor.ZmqIpcServer") as mock_zmq,
            patch.object(paddle.device, "set_device"),
        ):

            mock_envs.FD_USE_GET_SAVE_OUTPUT_V1 = True

            processor = TokenProcessor(
                self.mock_cfg, self.mock_cached_tokens, self.mock_engine_worker_queue, self.mock_split_connector
            )

            # Check ZMQ server is created
            mock_zmq.assert_called_once()
            del processor

    def test_token_processor_with_speculative_decoding(self):
        """Test TokenProcessor initialization with speculative decoding."""
        self.mock_cfg.speculative_config.method = "mtp"

        with patch("fastdeploy.output.token_processor.envs") as mock_envs, patch.object(paddle.device, "set_device"):

            mock_envs.FD_USE_GET_SAVE_OUTPUT_V1 = False

            processor = TokenProcessor(
                self.mock_cfg, self.mock_cached_tokens, self.mock_engine_worker_queue, self.mock_split_connector
            )

            # Check speculative decoding is enabled
            self.assertTrue(processor.speculative_decoding)

    def test_token_processor_with_logprobs(self):
        """Test TokenProcessor initialization with logprobs enabled."""
        self.mock_cfg.model_config.enable_logprob = True

        with patch("fastdeploy.output.token_processor.envs") as mock_envs, patch.object(paddle.device, "set_device"):

            mock_envs.FD_USE_GET_SAVE_OUTPUT_V1 = False

            processor = TokenProcessor(
                self.mock_cfg, self.mock_cached_tokens, self.mock_engine_worker_queue, self.mock_split_connector
            )

            # Check logprobs is enabled
            self.assertTrue(processor.use_logprobs)

    def test_set_resource_manager(self):
        """Test setting resource manager."""
        with patch("fastdeploy.output.token_processor.envs") as mock_envs, patch.object(paddle.device, "set_device"):

            mock_envs.FD_USE_GET_SAVE_OUTPUT_V1 = False

            processor = TokenProcessor(
                self.mock_cfg, self.mock_cached_tokens, self.mock_engine_worker_queue, self.mock_split_connector
            )

            mock_resource_manager = Mock()
            processor.set_resource_manager(mock_resource_manager)

            self.assertEqual(processor.resource_manager, mock_resource_manager)

            # Test that setting again raises an assertion error
            with self.assertRaises(AssertionError):
                processor.set_resource_manager(mock_resource_manager)

    def test_set_resource_manager_none_error(self):
        """Test that resource manager must not be None."""
        with patch("fastdeploy.output.token_processor.envs") as mock_envs, patch.object(paddle.device, "set_device"):

            mock_envs.FD_USE_GET_SAVE_OUTPUT_V1 = False

            processor = TokenProcessor(
                self.mock_cfg, self.mock_cached_tokens, self.mock_engine_worker_queue, self.mock_split_connector
            )

            with self.assertRaises(AssertionError):
                processor.set_resource_manager(None)

    def test_run_without_resource_manager(self):
        """Test that run fails without resource manager."""
        with patch("fastdeploy.output.token_processor.envs") as mock_envs, patch.object(paddle.device, "set_device"):

            mock_envs.FD_USE_GET_SAVE_OUTPUT_V1 = False

            processor = TokenProcessor(
                self.mock_cfg, self.mock_cached_tokens, self.mock_engine_worker_queue, self.mock_split_connector
            )

            with self.assertRaises(AssertionError):
                processor.run()

    def test_run_worker_already_running(self):
        """Test that run fails when worker is already running."""
        with patch("fastdeploy.output.token_processor.envs") as mock_envs, patch.object(paddle.device, "set_device"):

            mock_envs.FD_USE_GET_SAVE_OUTPUT_V1 = False

            processor = TokenProcessor(
                self.mock_cfg, self.mock_cached_tokens, self.mock_engine_worker_queue, self.mock_split_connector
            )

            # Set resource manager
            mock_resource_manager = Mock()
            processor.set_resource_manager(mock_resource_manager)

            # Mock existing worker
            processor.worker = Mock()

            with self.assertRaises(Exception):
                processor.run()

    def test_cleanup_resources(self):
        """Test resource cleanup."""
        with patch("fastdeploy.output.token_processor.envs") as mock_envs, patch.object(paddle.device, "set_device"):

            mock_envs.FD_USE_GET_SAVE_OUTPUT_V1 = False

            processor = TokenProcessor(
                self.mock_cfg, self.mock_cached_tokens, self.mock_engine_worker_queue, self.mock_split_connector
            )

            # Mock prefill_time_signal and executor
            processor.prefill_time_signal = Mock()
            processor.executor = Mock()

            # Call cleanup
            processor._cleanup_resources()

            # Verify cleanup calls
            processor.prefill_time_signal.clear.assert_called_once()
            processor.executor.shutdown.assert_called_once_with(wait=False)

    def test_record_metrics(self):
        """Test metrics recording."""
        with patch("fastdeploy.output.token_processor.envs") as mock_envs, patch.object(paddle.device, "set_device"):

            mock_envs.FD_USE_GET_SAVE_OUTPUT_V1 = False

            processor = TokenProcessor(
                self.mock_cfg, self.mock_cached_tokens, self.mock_engine_worker_queue, self.mock_split_connector
            )

            # Create mock task
            mock_task = Mock()
            mock_task.last_token_time = time.time() - 0.1

            current_time = time.time()
            token_ids = [1, 2, 3]

            # Record metrics
            processor._record_metrics(mock_task, current_time, token_ids)

            # Check that task's last_token_time was updated
            self.assertEqual(mock_task.last_token_time, current_time)

    def test_record_metrics_first_time(self):
        """Test metrics recording for first time."""
        with patch("fastdeploy.output.token_processor.envs") as mock_envs, patch.object(paddle.device, "set_device"):

            mock_envs.FD_USE_GET_SAVE_OUTPUT_V1 = False

            processor = TokenProcessor(
                self.mock_cfg, self.mock_cached_tokens, self.mock_engine_worker_queue, self.mock_split_connector
            )

            # Create mock task without last_token_time
            mock_task = Mock()
            del mock_task.last_token_time  # Remove attribute if exists

            current_time = time.time()
            token_ids = [1, 2, 3]

            # Record metrics should not raise error
            processor._record_metrics(mock_task, current_time, token_ids)

    def test_record_first_token_metrics(self):
        """Test first token metrics recording."""
        with patch("fastdeploy.output.token_processor.envs") as mock_envs, patch.object(paddle.device, "set_device"):

            mock_envs.FD_USE_GET_SAVE_OUTPUT_V1 = False

            processor = TokenProcessor(
                self.mock_cfg, self.mock_cached_tokens, self.mock_engine_worker_queue, self.mock_split_connector
            )

            # Create mock task
            mock_task = Mock()
            mock_task.inference_start_time = time.time() - 0.5
            mock_task.schedule_start_time = time.time() - 0.4
            mock_task.preprocess_end_time = time.time() - 0.3
            mock_task.preprocess_start_time = time.time() - 0.2

            current_time = time.time()

            # Record metrics should not raise error
            processor._record_first_token_metrics(mock_task, current_time)

            # Check that task's first_token_time was set
            self.assertEqual(mock_task.first_token_time, current_time)

    def test_record_completion_metrics(self):
        """Test completion metrics recording."""
        with patch("fastdeploy.output.token_processor.envs") as mock_envs, patch.object(paddle.device, "set_device"):

            mock_envs.FD_USE_GET_SAVE_OUTPUT_V1 = False

            processor = TokenProcessor(
                self.mock_cfg, self.mock_cached_tokens, self.mock_engine_worker_queue, self.mock_split_connector
            )

            # Create mock task
            mock_task = Mock()
            mock_task.request_id = "test_request"
            mock_task.first_token_time = time.time() - 0.3
            mock_task.inference_start_time = time.time() - 0.5

            current_time = time.time()
            processor.tokens_counter[mock_task.request_id] = 5

            # Record metrics should not raise error
            processor._record_completion_metrics(mock_task, current_time)

    def test_compute_speculative_status(self):
        """Test speculative status computation."""
        self.mock_cfg.speculative_config.method = "mtp"

        with patch("fastdeploy.output.token_processor.envs") as mock_envs, patch.object(paddle.device, "set_device"):

            mock_envs.FD_USE_GET_SAVE_OUTPUT_V1 = False

            processor = TokenProcessor(
                self.mock_cfg, self.mock_cached_tokens, self.mock_engine_worker_queue, self.mock_split_connector
            )

            # Set some stats
            processor.total_step = 10
            processor.number_of_output_tokens = 100
            processor.speculative_stats_step = 0

            # Compute speculative status should not raise error
            processor._compute_speculative_status()

            # Check that step was incremented
            self.assertEqual(processor.speculative_stats_step, 1)

    def test_postprocess(self):
        """Test postprocessing of batch results."""
        with patch("fastdeploy.output.token_processor.envs") as mock_envs, patch.object(paddle.device, "set_device"):

            mock_envs.FD_USE_GET_SAVE_OUTPUT_V1 = False

            processor = TokenProcessor(
                self.mock_cfg, self.mock_cached_tokens, self.mock_engine_worker_queue, self.mock_split_connector
            )

            # Create mock batch results
            mock_result = Mock()
            batch_result = [mock_result]

            # Postprocess should not raise error
            processor.postprocess(batch_result)

            # Verify cached_generated_tokens.put_results was called
            processor.cached_generated_tokens.put_results.assert_called_once_with(batch_result)

    def test_postprocess_with_speculative_decoding(self):
        """Test postprocessing with speculative decoding."""
        self.mock_cfg.speculative_config.method = "mtp"
        self.mock_cfg.model_config.enable_logprob = True

        with patch("fastdeploy.output.token_processor.envs") as mock_envs, patch.object(paddle.device, "set_device"):

            mock_envs.FD_USE_GET_SAVE_OUTPUT_V1 = False

            processor = TokenProcessor(
                self.mock_cfg, self.mock_cached_tokens, self.mock_engine_worker_queue, self.mock_split_connector
            )

            # Create mock batch results
            mock_finished_result = Mock()
            mock_finished_result.finished = True
            mock_unfinished_result = Mock()
            mock_unfinished_result.finished = False
            batch_result = [mock_finished_result, mock_unfinished_result]

            # Postprocess should not raise error
            processor.postprocess(batch_result, mtype=3)

            # Verify cached_generated_tokens.put_results was called
            processor.cached_generated_tokens.put_results.assert_called_once()

    def test_process_per_token(self):
        """Test per-token processing."""
        with patch("fastdeploy.output.token_processor.envs") as mock_envs, patch.object(paddle.device, "set_device"):

            mock_envs.FD_USE_GET_SAVE_OUTPUT_V1 = False

            processor = TokenProcessor(
                self.mock_cfg, self.mock_cached_tokens, self.mock_engine_worker_queue, self.mock_split_connector
            )

            # Create mock task and result
            mock_task = Mock()
            mock_task.request_id = "test_request"
            mock_task.eos_token_ids = [2]
            mock_task.output_token_ids = []
            mock_task.inference_start_time = time.time() - 1.0

            processor.tokens_counter[mock_task.request_id] = 0

            mock_result = Mock()
            mock_result.outputs = Mock()
            mock_result.outputs.token_ids = []

            # Create token array
            token_ids = np.array([1, 2])

            # Process tokens
            result = processor._process_per_token(mock_task, 0, token_ids, mock_result, False)

            # Verify result was updated
            self.assertIsNotNone(result)

    def test_process_per_token_with_eos(self):
        """Test per-token processing with EOS token."""
        with patch("fastdeploy.output.token_processor.envs") as mock_envs, patch.object(paddle.device, "set_device"):

            mock_envs.FD_USE_GET_SAVE_OUTPUT_V1 = False

            processor = TokenProcessor(
                self.mock_cfg, self.mock_cached_tokens, self.mock_engine_worker_queue, self.mock_split_connector
            )

            # Create mock task and result
            mock_task = Mock()
            mock_task.request_id = "test_request"
            mock_task.eos_token_ids = [2]
            mock_task.output_token_ids = []
            mock_task.inference_start_time = time.time() - 1.0

            processor.tokens_counter[mock_task.request_id] = 0

            mock_result = Mock()
            mock_result.outputs = Mock()
            mock_result.outputs.token_ids = []

            # Create token array with EOS
            token_ids = np.array([2])

            # Process tokens
            result = processor._process_per_token(mock_task, 0, token_ids, mock_result, False)

            # Verify result is marked as finished
            self.assertTrue(result.finished)

    def test_process_per_token_with_recovery_stop(self):
        """Test per-token processing with recovery stop signal."""
        with patch("fastdeploy.output.token_processor.envs") as mock_envs, patch.object(paddle.device, "set_device"):

            mock_envs.FD_USE_GET_SAVE_OUTPUT_V1 = False

            processor = TokenProcessor(
                self.mock_cfg, self.mock_cached_tokens, self.mock_engine_worker_queue, self.mock_split_connector
            )

            # Create mock task and result
            mock_task = Mock()
            mock_task.request_id = "test_request"
            mock_task.eos_token_ids = [2]
            mock_task.output_token_ids = []
            mock_task.inference_start_time = time.time() - 1.0

            processor.tokens_counter[mock_task.request_id] = 0

            mock_result = Mock()
            mock_result.outputs = Mock()
            mock_result.outputs.token_ids = []

            # Create token array with recovery stop signal
            token_ids = np.array([RECOVERY_STOP_SIGNAL])

            # Process tokens
            result = processor._process_per_token(mock_task, 0, token_ids, mock_result, False)

            # Verify result is marked as finished
            self.assertTrue(result.finished)
            self.assertEqual(result.error_msg, "Recover is not supported, the result is incomplete!")

    def test_clear_data(self):
        """Test clearing data."""
        with patch("fastdeploy.output.token_processor.envs") as mock_envs, patch.object(paddle.device, "set_device"):

            mock_envs.FD_USE_GET_SAVE_OUTPUT_V1 = False
            mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = False

            processor = TokenProcessor(
                self.mock_cfg, self.mock_cached_tokens, self.mock_engine_worker_queue, self.mock_split_connector
            )

            # Create mock resource manager
            mock_resource_manager = Mock()
            mock_resource_manager.stop_flags = [True] * self.mock_cfg.max_num_seqs
            mock_resource_manager.tasks_list = [None] * self.mock_cfg.max_num_seqs
            processor.set_resource_manager(mock_resource_manager)

            # Clear data should not raise error
            processor.clear_data()


class TestWarmUpTokenProcessor(unittest.TestCase):
    """Test cases for WarmUpTokenProcessor class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock configuration
        self.mock_cfg = Mock()
        self.mock_cfg.parallel_config = Mock()
        self.mock_cfg.parallel_config.local_data_parallel_id = 0
        self.mock_cfg.speculative_config = Mock()
        self.mock_cfg.speculative_config.method = None
        self.mock_cfg.model_config = Mock()
        self.mock_cfg.model_config.enable_logprob = False

    def test_warmup_token_processor_initialization(self):
        """Test WarmUpTokenProcessor initialization."""
        with patch("fastdeploy.output.token_processor.envs") as mock_envs, patch.object(paddle.device, "set_device"):

            mock_envs.FD_USE_GET_SAVE_OUTPUT_V1 = False

            processor = WarmUpTokenProcessor(self.mock_cfg)

            # Check initialization attributes
            self.assertTrue(processor._is_running)
            self.assertTrue(processor._is_blocking)

    def test_warmup_postprocess(self):
        """Test that warmup postprocess does nothing."""
        with patch("fastdeploy.output.token_processor.envs") as mock_envs, patch.object(paddle.device, "set_device"):

            mock_envs.FD_USE_GET_SAVE_OUTPUT_V1 = False

            processor = WarmUpTokenProcessor(self.mock_cfg)

            # Postprocess should do nothing
            batch_result = [Mock()]
            processor.postprocess(batch_result)  # Should not raise error

    def test_warmup_stop(self):
        """Test stopping warmup processor."""
        with patch("fastdeploy.output.token_processor.envs") as mock_envs, patch.object(paddle.device, "set_device"):

            mock_envs.FD_USE_GET_SAVE_OUTPUT_V1 = False

            processor = WarmUpTokenProcessor(self.mock_cfg)

            # Mock worker
            processor.worker = Mock()
            processor.worker.join = Mock()

            # Stop should not raise error
            processor.stop()

            # Check that _is_running is set to False
            self.assertFalse(processor._is_running)

            # Check that worker.join was called
            processor.worker.join.assert_called_once()


class TestTokenProcessorConstants(unittest.TestCase):
    """Test cases for TokenProcessor constants."""

    def test_recovery_stop_signal(self):
        """Test RECOVERY_STOP_SIGNAL constant."""
        # The constant should be defined and be -3
        self.assertEqual(RECOVERY_STOP_SIGNAL, -3)


if __name__ == "__main__":
    # Print current test mode for clarity
    print(f"Running tests in {TEST_MODE} mode")
    if TEST_MODE == "standalone":
        print("To run in normal mode, ensure fastdeploy is properly installed")
        print("Or set FD_TEST_MODE=normal environment variable")
    unittest.main(verbosity=2)
