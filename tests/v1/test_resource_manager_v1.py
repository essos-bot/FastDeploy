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
import unittest
from unittest.mock import Mock, patch

# Determine import method based on environment
TEST_MODE = os.environ.get("FD_TEST_MODE", "normal")

if TEST_MODE == "standalone":
    # Local testing mode - use dynamic import
    mock_logger = Mock()

    # Create mock modules
    sys.modules["fastdeploy"] = Mock()
    sys.modules["fastdeploy.cache_manager"] = Mock()
    sys.modules["fastdeploy.cache_manager.multimodal_cache_manager"] = Mock()
    sys.modules["fastdeploy.engine"] = Mock()
    sys.modules["fastdeploy.engine.request"] = Mock()
    sys.modules["fastdeploy.engine.resource_manager"] = Mock()
    sys.modules["fastdeploy.inter_communicator"] = Mock()
    sys.modules["fastdeploy.metrics"] = Mock()
    sys.modules["fastdeploy.metrics.metrics"] = Mock()
    sys.modules["fastdeploy.multimodal"] = Mock()
    sys.modules["fastdeploy.multimodal.hasher"] = Mock()
    sys.modules["fastdeploy.platforms"] = Mock()
    sys.modules["fastdeploy.utils"] = Mock()
    sys.modules["fastdeploy.envs"] = Mock()

    # Mock paddle
    sys.modules["paddle"] = Mock()

    # Import the module directly
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "resource_manager_v1",
        os.path.join(os.path.dirname(__file__), "../../fastdeploy/engine/sched/resource_manager_v1.py"),
    )
    resource_manager_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(resource_manager_module)

    # Extract classes we want to test
    SignalConsumer = resource_manager_module.SignalConsumer
    ScheduledDecodeTask = resource_manager_module.ScheduledDecodeTask
    ScheduledPreemptTask = resource_manager_module.ScheduledPreemptTask
    ScheduledExtendBlocksTask = resource_manager_module.ScheduledExtendBlocksTask
    ResourceManagerV1 = resource_manager_module.ResourceManagerV1

else:
    # Normal mode - direct import
    try:
        from fastdeploy.engine.sched.resource_manager_v1 import (
            ResourceManagerV1,
            ScheduledDecodeTask,
            ScheduledExtendBlocksTask,
            ScheduledPreemptTask,
            SignalConsumer,
        )
    except ImportError as e:
        print(f"Warning: Direct import failed: {e}")
        print("Falling back to standalone mode")
        TEST_MODE = "standalone"

        # Re-run standalone setup
        mock_logger = Mock()

        sys.modules["fastdeploy"] = Mock()
        sys.modules["fastdeploy.cache_manager"] = Mock()
        sys.modules["fastdeploy.cache_manager.multimodal_cache_manager"] = Mock()
        sys.modules["fastdeploy.engine"] = Mock()
        sys.modules["fastdeploy.engine.request"] = Mock()
        sys.modules["fastdeploy.engine.resource_manager"] = Mock()
        sys.modules["fastdeploy.inter_communicator"] = Mock()
        sys.modules["fastdeploy.metrics"] = Mock()
        sys.modules["fastdeploy.metrics.metrics"] = Mock()
        sys.modules["fastdeploy.multimodal"] = Mock()
        sys.modules["fastdeploy.multimodal.hasher"] = Mock()
        sys.modules["fastdeploy.platforms"] = Mock()
        sys.modules["fastdeploy.utils"] = Mock()
        sys.modules["fastdeploy.envs"] = Mock()
        sys.modules["paddle"] = Mock()

        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "resource_manager_v1",
            os.path.join(os.path.dirname(__file__), "../../fastdeploy/engine/sched/resource_manager_v1.py"),
        )
        resource_manager_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(resource_manager_module)

        SignalConsumer = resource_manager_module.SignalConsumer
        ScheduledDecodeTask = resource_manager_module.ScheduledDecodeTask
        ScheduledPreemptTask = resource_manager_module.ScheduledPreemptTask
        ScheduledExtendBlocksTask = resource_manager_module.ScheduledExtendBlocksTask
        ResourceManagerV1 = resource_manager_module.ResourceManagerV1


class TestSignalConsumer(unittest.TestCase):
    """Test cases for SignalConsumer class."""

    def test_signal_consumer_initialization(self):
        """Test SignalConsumer initialization with valid parameters."""
        signal = 42
        consume_limit = 5

        consumer = SignalConsumer(signal, consume_limit)

        self.assertEqual(consumer._signal, signal)
        self.assertEqual(consumer._consume_limit, consume_limit)

    def test_signal_consumer_invalid_limit(self):
        """Test SignalConsumer initialization with invalid consume_limit."""
        with self.assertRaises(AssertionError):
            SignalConsumer(42, 0)

        with self.assertRaises(AssertionError):
            SignalConsumer(42, -1)

    def test_signal_consumer_watch(self):
        """Test watch method returns current signal without consuming."""
        signal = 42
        consume_limit = 5
        consumer = SignalConsumer(signal, consume_limit)

        # Multiple calls should return same signal
        self.assertEqual(consumer.watch(), 42)
        self.assertEqual(consumer.watch(), 42)
        self.assertEqual(consumer._consume_limit, 5)  # Should not change

    def test_signal_consumer_consume(self):
        """Test consume method returns signal and decrements limit."""
        signal = 42
        consume_limit = 3
        consumer = SignalConsumer(signal, consume_limit)

        # First consumption
        result1 = consumer.consume()
        self.assertEqual(result1, 42)
        self.assertEqual(consumer._consume_limit, 2)

        # Second consumption
        result2 = consumer.consume()
        self.assertEqual(result2, 42)
        self.assertEqual(consumer._consume_limit, 1)

        # Third consumption - should reset signal to 0
        result3 = consumer.consume()
        self.assertEqual(result3, 42)  # Still returns original signal
        self.assertEqual(consumer._consume_limit, 0)
        self.assertEqual(consumer._signal, 0)  # Signal reset to 0

        # Further consumptions should return 0
        result4 = consumer.consume()
        self.assertEqual(result4, 0)

    def test_signal_consumer_consume_complete_cycle(self):
        """Test complete consumption cycle until signal is reset."""
        signal = 42
        consume_limit = 3
        consumer = SignalConsumer(signal, consume_limit)

        # Consume until limit is reached
        result1 = consumer.consume()
        self.assertEqual(result1, 42)
        self.assertEqual(consumer._consume_limit, 2)

        result2 = consumer.consume()
        self.assertEqual(result2, 42)
        self.assertEqual(consumer._consume_limit, 1)

        result3 = consumer.consume()
        self.assertEqual(result3, 42)
        self.assertEqual(consumer._consume_limit, 0)
        self.assertEqual(consumer._signal, 0)  # Signal should be reset

        # Further consumption should return 0
        result4 = consumer.consume()
        self.assertEqual(result4, 0)


class TestScheduledTasks(unittest.TestCase):
    """Test cases for scheduled task dataclasses."""

    def test_scheduled_decode_task(self):
        """Test ScheduledDecodeTask creation and attributes."""
        task = ScheduledDecodeTask(idx=1, request_id="test_req_1", block_tables=[1, 2, 3])

        self.assertEqual(task.idx, 1)
        self.assertEqual(task.request_id, "test_req_1")
        self.assertEqual(task.block_tables, [1, 2, 3])
        self.assertEqual(task.task_type.name, "DECODE")  # RequestType enum

    def test_scheduled_preempt_task(self):
        """Test ScheduledPreemptTask creation and attributes."""
        task = ScheduledPreemptTask(idx=2, request_id="test_req_2")

        self.assertEqual(task.idx, 2)
        self.assertEqual(task.request_id, "test_req_2")
        self.assertEqual(task.task_type.name, "PREEMPTED")  # RequestType enum

    def test_scheduled_extend_blocks_task(self):
        """Test ScheduledExtendBlocksTask creation and attributes."""
        task = ScheduledExtendBlocksTask(idx=3, request_id="test_req_3", extend_block_tables=[4, 5, 6])

        self.assertEqual(task.idx, 3)
        self.assertEqual(task.request_id, "test_req_3")
        self.assertEqual(task.extend_block_tables, [4, 5, 6])
        self.assertEqual(task.task_type.name, "EXTEND")  # RequestType enum


class TestResourceManagerV1(unittest.TestCase):
    """Test cases for ResourceManagerV1 class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock config
        self.mock_config = Mock()
        self.mock_config.cache_config.block_size = 16
        self.mock_config.cache_config.max_block_num_per_seq = 1024
        self.mock_config.cache_config.enc_dec_block_num = 4
        self.mock_config.cache_config.prealloc_dec_block_slot_num_threshold = 64
        self.mock_config.cache_config.enable_prefix_caching = False
        self.mock_config.cache_config.enable_hierarchical_cache = False
        self.mock_config.cache_config.max_encoder_cache = 0
        self.mock_config.cache_config.max_processor_cache = 0
        self.mock_config.model_config.enable_mm = False
        self.mock_config.scheduler_config.max_num_batched_tokens = 8192
        self.mock_config.scheduler_config.splitwise_role = "mixed"
        self.mock_config.speculative_config.method = None

        # Create mock request
        self.mock_request = Mock()
        self.mock_request.request_id = "test_req_1"
        self.mock_request.prompt_token_ids = [1, 2, 3, 4, 5]
        self.mock_request.output_token_ids = []
        self.mock_request.num_computed_tokens = 0
        self.mock_request.num_total_tokens = 5
        self.mock_request.need_prefill_tokens = 5
        self.mock_request.block_tables = []
        self.mock_request.extend_block_tables = []
        self.mock_request.use_extend_tables = False
        self.mock_request.multimodal_inputs = None
        self.mock_request.with_image = False
        self.mock_request.status = Mock()
        self.mock_request.status.name = "WAITING"
        self.mock_request.cached_block_num = 0
        self.mock_request.multimodal_img_boundaries = None
        self.mock_request.get = Mock(return_value=False)

    def create_resource_manager(self, max_num_seqs=4):
        """Helper to create ResourceManagerV1 with mocked dependencies."""
        with patch("fastdeploy.engine.sched.resource_manager_v1.envs"):
            with patch("fastdeploy.engine.sched.resource_manager_v1.main_process_metrics"):
                with patch("fastdeploy.engine.sched.resource_manager_v1.IPCSignal"):
                    with patch("fastdeploy.engine.sched.resource_manager_v1.threading.ThreadPoolExecutor"):
                        manager = ResourceManagerV1(
                            max_num_seqs=max_num_seqs,
                            config=self.mock_config,
                            tensor_parallel_size=1,
                            splitwise_role="mixed",
                            local_data_parallel_id=0,
                        )
        # Mock the cache manager
        manager.cache_manager = Mock()
        manager.cache_manager.can_allocate_gpu_blocks.return_value = True
        manager.cache_manager.allocate_gpu_blocks.return_value = [1, 2, 3]
        manager.cache_manager.recycle_gpu_blocks.return_value = None
        manager.total_block_number.return_value = 1000
        manager.get_gpu_cache_usage_perc.return_value = 0.5

        return manager

    def test_resource_manager_initialization(self):
        """Test ResourceManagerV1 initialization."""
        manager = self.create_resource_manager()

        # Test that the manager was created successfully
        self.assertIsNotNone(manager)
        self.assertIsNotNone(manager.config)
        self.assertIsNotNone(manager.lock)
        self.assertIsNotNone(manager.cache_manager)

    def test_allocated_slots(self):
        """Test allocated_slots calculation."""
        manager = self.create_resource_manager()

        # Mock request with block tables
        self.mock_request.block_tables = [1, 2, 3, 4]  # 4 blocks

        slots = manager.allocated_slots(self.mock_request)

        expected_slots = 4 * 16  # 4 blocks * 16 block_size
        self.assertEqual(slots, expected_slots)

    def test_get_new_block_nums(self):
        """Test get_new_block_nums calculation."""
        manager = self.create_resource_manager()

        # Setup request
        self.mock_request.num_computed_tokens = 10
        self.mock_request.block_tables = [1, 2]  # 2 blocks already allocated

        # Test basic case
        num_new_blocks = manager.get_new_block_nums(self.mock_request, 20)
        # (10 + 20 + 16 - 1) // 16 - 2 = (45) // 16 - 2 = 2 - 2 = 0
        self.assertEqual(num_new_blocks, 0)

        # Test with more tokens
        num_new_blocks = manager.get_new_block_nums(self.mock_request, 50)
        # (10 + 50 + 16 - 1) // 16 - 2 = (75) // 16 - 2 = 4 - 2 = 2
        self.assertEqual(num_new_blocks, 2)

    def test_get_new_block_nums_with_speculative_config(self):
        """Test get_new_block_nums with speculative config."""
        self.mock_config.speculative_config.method = "mtp"
        manager = self.create_resource_manager()

        self.mock_request.num_computed_tokens = 10
        self.mock_request.block_tables = [1, 2]

        num_new_blocks = manager.get_new_block_nums(self.mock_request, 20)
        # Should be limited by max_block_num_per_seq
        self.assertLessEqual(num_new_blocks, self.mock_config.cache_config.max_block_num_per_seq)

    def test_prepare_prefill_task(self):
        """Test _prepare_prefill_task method."""
        manager = self.create_resource_manager()

        self.mock_request.num_computed_tokens = 5
        new_token_num = 10

        result = manager._prepare_prefill_task(self.mock_request, new_token_num)

        self.assertEqual(result, self.mock_request)
        self.assertEqual(self.mock_request.prefill_start_index, 5)
        self.assertEqual(self.mock_request.prefill_end_index, 15)
        self.assertEqual(self.mock_request.task_type.name, "PREFILL")

    def test_prepare_decode_task(self):
        """Test _prepare_decode_task method."""
        manager = self.create_resource_manager()

        self.mock_request.idx = 2
        self.mock_request.request_id = "test_req"
        self.mock_request.block_tables = [1, 2, 3]

        result = manager._prepare_decode_task(self.mock_request)

        self.assertIsInstance(result, ScheduledDecodeTask)
        self.assertEqual(result.idx, 2)
        self.assertEqual(result.request_id, "test_req")
        self.assertEqual(result.block_tables, [1, 2, 3])
        self.assertEqual(result.task_type.name, "DECODE")

    def test_prepare_preempt_task(self):
        """Test _prepare_preempt_task method."""
        manager = self.create_resource_manager()

        self.mock_request.idx = 3
        self.mock_request.request_id = "test_req"

        result = manager._prepare_preempt_task(self.mock_request)

        self.assertIsInstance(result, ScheduledPreemptTask)
        self.assertEqual(result.idx, 3)
        self.assertEqual(result.request_id, "test_req")
        self.assertEqual(result.task_type.name, "PREEMPTED")

    def test_reschedule_preempt_task(self):
        """Test reschedule_preempt_task method."""
        manager = self.create_resource_manager()

        # Add request to to_be_rescheduled set and requests dict
        manager.to_be_rescheduled_request_id_set.add("test_req")
        manager.requests["test_req"] = self.mock_request

        manager.reschedule_preempt_task("test_req")

        self.assertNotIn("test_req", manager.to_be_rescheduled_request_id_set)
        self.assertEqual(len(manager.waiting), 1)
        self.assertEqual(manager.waiting[0], self.mock_request)

    def test_can_preempt(self):
        """Test _can_preempt method."""
        manager = self.create_resource_manager()

        # No running requests
        self.assertFalse(manager._can_preempt())

        # Add running request with extend tables
        self.mock_request.use_extend_tables = True
        manager.running.append(self.mock_request)
        self.assertFalse(manager._can_preempt())

        # Add running request without extend tables
        self.mock_request.use_extend_tables = False
        self.assertTrue(manager._can_preempt())

    def test_get_available_position(self):
        """Test get_available_position method."""
        manager = self.create_resource_manager()

        # All positions should be available initially
        position = manager.get_available_position()
        self.assertEqual(position, 0)

        # Mark first position as occupied
        manager.stop_flags[0] = False
        position = manager.get_available_position()
        self.assertEqual(position, 1)

    def test_get_available_position_no_available(self):
        """Test get_available_position when no positions available."""
        manager = self.create_resource_manager()

        # Mark all positions as occupied
        for i in range(manager.max_num_seqs):
            manager.stop_flags[i] = False

        with self.assertRaises(RuntimeError):
            manager.get_available_position()

    def test_get_real_bsz(self):
        """Test get_real_bsz method."""
        manager = self.create_resource_manager()

        # All positions should be free initially
        real_bsz = manager.get_real_bsz()
        self.assertEqual(real_bsz, 0)

        # Mark some positions as occupied
        manager.stop_flags[0] = False
        manager.stop_flags[1] = False
        real_bsz = manager.get_real_bsz()
        self.assertEqual(real_bsz, 2)

    def test_add_request(self):
        """Test add_request method."""
        manager = self.create_resource_manager()

        manager.add_request(self.mock_request)

        self.assertEqual(len(manager.waiting), 1)
        self.assertEqual(manager.waiting[0], self.mock_request)
        self.assertIn(self.mock_request.request_id, manager.requests)
        self.assertEqual(manager.requests[self.mock_request.request_id], self.mock_request)

    def test_exist_mm_prefill(self):
        """Test exist_mm_prefill method."""
        manager = self.create_resource_manager()

        # Create mock multimodal request
        mm_request = Mock()
        mm_request.task_type.name = "PREFILL"

        # Mock _is_mm_request to return True
        manager._is_mm_request = Mock(return_value=True)

        scheduled_reqs = [mm_request]

        result = manager.exist_mm_prefill(scheduled_reqs)
        self.assertTrue(result)

        # Mock _is_mm_request to return False
        manager._is_mm_request = Mock(return_value=False)
        result = manager.exist_mm_prefill(scheduled_reqs)
        self.assertFalse(result)

    def test_exist_prefill(self):
        """Test exist_prefill method."""
        manager = self.create_resource_manager()

        # Create mock prefill request
        prefill_request = Mock()
        prefill_request.task_type.name = "PREFILL"

        # Create mock decode request
        decode_request = Mock()
        decode_request.task_type.name = "DECODE"

        # Test with prefill request
        scheduled_reqs = [prefill_request]
        result = manager.exist_prefill(scheduled_reqs)
        self.assertTrue(result)

        # Test with decode request
        scheduled_reqs = [decode_request]
        result = manager.exist_prefill(scheduled_reqs)
        self.assertFalse(result)

    def test_is_mm_request_with_none_inputs(self):
        """Test _is_mm_request with None multimodal inputs."""
        manager = self.create_resource_manager()

        self.mock_request.multimodal_inputs = None
        result = manager._is_mm_request(self.mock_request)
        self.assertFalse(result)

    def test_is_mm_request_with_empty_inputs(self):
        """Test _is_mm_request with empty multimodal inputs."""
        manager = self.create_resource_manager()

        self.mock_request.multimodal_inputs = {}
        result = manager._is_mm_request(self.mock_request)
        self.assertFalse(result)

    def test_is_mm_request_with_image_urls(self):
        """Test _is_mm_request with image feature URLs."""
        manager = self.create_resource_manager()

        self.mock_request.multimodal_inputs = {"image_feature_urls": ["http://example.com/image.jpg"]}
        result = manager._is_mm_request(self.mock_request)
        self.assertTrue(result)

    def test_is_mm_request_with_images_and_grid_thw(self):
        """Test _is_mm_request with images and grid_thw."""
        manager = self.create_resource_manager()

        self.mock_request.multimodal_inputs = {"images": [Mock()], "image_patch_id": 1, "grid_thw": [[1, 224, 224]]}
        result = manager._is_mm_request(self.mock_request)
        self.assertTrue(result)

    def test_update_mm_hashes_with_none_inputs(self):
        """Test _update_mm_hashes with None multimodal inputs."""
        manager = self.create_resource_manager()

        self.mock_request.multimodal_inputs = None
        # Should not raise any exception
        manager._update_mm_hashes(self.mock_request)

    def test_get_num_new_tokens_without_multimodal(self):
        """Test _get_num_new_tokens without multimodal enabled."""
        manager = self.create_resource_manager()
        self.mock_config.model_config.enable_mm = False

        token_budget = 100
        self.mock_request.need_prefill_tokens = 50
        self.mock_request.num_computed_tokens = 10

        result = manager._get_num_new_tokens(self.mock_request, token_budget)
        expected = min(50 - 10, 100)  # 40
        self.assertEqual(result, expected)

    def test_finish_requests_with_single_request(self):
        """Test finish_requests with single request ID."""
        manager = self.create_resource_manager()

        # Add request to running and requests
        manager.running.append(self.mock_request)
        manager.requests["test_req_1"] = self.mock_request
        manager.req_dict["test_req_1"] = 0

        # Mock status
        from fastdeploy.engine.request import RequestStatus

        self.mock_request.status = RequestStatus.RUNNING

        manager.finish_requests("test_req_1")

        self.assertNotIn(self.mock_request, manager.running)
        self.assertNotIn("test_req_1", manager.requests)
        self.assertEqual(manager.stop_flags[0], True)

    def test_finish_requests_with_multiple_requests(self):
        """Test finish_requests with multiple request IDs."""
        manager = self.create_resource_manager()

        # Create another request
        mock_request2 = Mock()
        mock_request2.request_id = "test_req_2"
        mock_request2.status = Mock()
        mock_request2.status.name = "RUNNING"

        # Add requests to running and requests
        manager.running.extend([self.mock_request, mock_request2])
        manager.requests["test_req_1"] = self.mock_request
        manager.requests["test_req_2"] = mock_request2
        manager.req_dict["test_req_1"] = 0
        manager.req_dict["test_req_2"] = 1

        manager.finish_requests(["test_req_1", "test_req_2"])

        self.assertEqual(len(manager.running), 0)
        self.assertEqual(len(manager.requests), 0)
        self.assertTrue(manager.stop_flags[0])
        self.assertTrue(manager.stop_flags[1])

    def test_clear_data(self):
        """Test clear_data method."""
        manager = self.create_resource_manager()

        # Add some data
        manager.waiting.append(self.mock_request)
        manager.to_be_rescheduled_request_id_set.add("test_req")

        manager.clear_data()

        self.assertEqual(len(manager.waiting), 0)
        self.assertEqual(len(manager.to_be_rescheduled_request_id_set), 0)

    @patch("fastdeploy.engine.sched.resource_manager_v1.time.time")
    def test_schedule_with_no_requests(self, mock_time):
        """Test schedule method with no requests."""
        manager = self.create_resource_manager()
        mock_time.return_value = 123456789

        scheduled_reqs = manager.schedule()

        self.assertEqual(len(scheduled_reqs), 0)

    def test_prerelease_resource(self):
        """Test prerelease_resource method."""
        manager = self.create_resource_manager()

        # Setup request
        self.mock_request.idx = 1

        manager.prerelease_resource(self.mock_request)

        self.assertTrue(manager.stop_flags[1])
        self.assertIsNone(manager.tasks_list[1])

    def test_available_batch(self):
        """Test available_batch method (inherited from parent)."""
        manager = self.create_resource_manager()

        # Initially all positions should be available
        available = manager.available_batch()
        self.assertGreaterEqual(available, 0)

        # Mark one position as occupied
        manager.stop_flags[0] = False
        available = manager.available_batch()
        self.assertGreaterEqual(available, 0)

    @patch("fastdeploy.engine.sched.resource_manager_v1.llm_logger")
    def test_info_each_block(self, mock_logger):
        """Test _info_each_block method logs correctly."""
        manager = self.create_resource_manager()

        # Add mock request to running
        self.mock_request.idx = 1
        self.mock_request.block_tables = [1, 2, 3]
        self.mock_request.extend_block_tables = [4, 5]
        manager.running.append(self.mock_request)

        manager._info_each_block()

        # Verify logger was called
        mock_logger.debug.assert_called()

        # Check the call contains expected information
        call_args = mock_logger.debug.call_args[0][0]
        self.assertIn("req idx 1", call_args)
        self.assertIn("occupy 3 block_tables", call_args)
        self.assertIn("2 extend_block_tables", call_args)


if __name__ == "__main__":
    print(f"Running tests in {TEST_MODE} mode")
    if TEST_MODE == "standalone":
        print("To run in normal mode, ensure fastdeploy is properly installed")
        print("Or set FD_TEST_MODE=normal environment variable")
    unittest.main(verbosity=2)
