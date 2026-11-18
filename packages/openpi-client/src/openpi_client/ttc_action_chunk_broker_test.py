from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
import threading

import numpy as np
import pytest

from openpi_client.ttc_action_chunk_broker import TTCActionChunkBroker


class Policy:
    def __init__(self):
        self.requests = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.resets = 0

    def get_server_metadata(self):
        return {
            "ttc": {
                "version": 1,
                "space": "model_normalized",
                "action_horizon": 8,
                "max_delay": 3,
                "blocks": {"actions": 4, "tolerance": 2},
            }
        }

    def infer(self, obs):
        self.requests.append(obs)
        request = obs["_ttc"]
        if request["prefixes"]:
            self.started.set()
            assert self.release.wait(5), "test did not release the pending request"
        start = obs["tick"]
        blocks = {
            name: np.broadcast_to(np.arange(start, start + 8)[:, None], (8, dim)).astype(np.float32).copy()
            for name, dim in self.get_server_metadata()["ttc"]["blocks"].items()
        }
        for name, prefix in request["prefixes"].items():
            blocks[name][: request["delay"]] = prefix[: request["delay"]]
        return {
            "actions": blocks["actions"][:, :2].copy(),
            "ttc": {"version": 1, "space": "model_normalized", "blocks": blocks},
        }

    def reset(self):
        self.resets += 1


def test_async_prefix_alignment_and_multiple_prior_blocks():
    policy = Policy()
    broker = TTCActionChunkBroker(policy, replan_steps=3, delay_steps=2)
    try:
        outputs = [broker.infer({"tick": tick})["actions"] for tick in range(18)]
        np.testing.assert_array_equal(outputs, np.repeat(np.arange(18)[:, None], 2, axis=1))
        assert len(policy.requests) >= 5
        for request in policy.requests[1:]:
            payload = request["_ttc"]
            for prefix in payload["prefixes"].values():
                np.testing.assert_array_equal(prefix[:2, 0], [request["tick"], request["tick"] + 1])
        assert broker.get_server_metadata() == policy.get_server_metadata()
    finally:
        broker.close()


def test_overrun_blocks_before_any_uncommitted_action():
    policy = Policy()
    broker = TTCActionChunkBroker(policy, replan_steps=2, delay_steps=2)
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        for tick in range(2):
            broker.infer({"tick": tick})
        policy.release.clear()
        broker.infer({"tick": 2})
        assert policy.started.wait(2)
        broker.infer({"tick": 3})
        pending = executor.submit(broker.infer, {"tick": 4})
        with pytest.raises(FutureTimeout):
            pending.result(timeout=0.05)
        policy.release.set()
        np.testing.assert_array_equal(pending.result(timeout=2)["actions"], [4, 4])
    finally:
        policy.release.set()
        executor.shutdown(wait=True)
        broker.close()


def test_reset_drains_stale_reply_and_starts_new_episode():
    policy = Policy()
    broker = TTCActionChunkBroker(policy, replan_steps=2, delay_steps=1)
    try:
        for tick in range(3):
            broker.infer({"tick": tick})
        broker.reset()
        assert policy.resets == 1
        np.testing.assert_array_equal(broker.infer({"tick": 100})["actions"], [100, 100])
        assert policy.requests[-1]["_ttc"]["delay"] == 0
    finally:
        broker.close()
    with pytest.raises(RuntimeError, match="closed"):
        broker.infer({"tick": 0})


@pytest.mark.parametrize("replan,delay", [(0, 0), (2, 3), (6, 3), (3, 4), (3, -1), (3, True)])
def test_invalid_budgets(replan, delay):
    with pytest.raises(ValueError, match="TTC requires"):
        TTCActionChunkBroker(Policy(), replan, delay)


def test_zero_delay_remains_synchronous_and_old_server_rejected():
    policy = Policy()
    broker = TTCActionChunkBroker(policy, 2, 0)
    try:
        for tick in range(7):
            np.testing.assert_array_equal(broker.infer({"tick": tick})["actions"], [tick, tick])
    finally:
        broker.close()
    policy.get_server_metadata = lambda: {}
    with pytest.raises(ValueError, match="advertising"):
        TTCActionChunkBroker(policy)


def test_worker_failure_propagates_without_retry_or_reusing_old_postfix():
    policy = Policy()
    broker = TTCActionChunkBroker(policy, 2, 1)
    try:
        broker.infer({"tick": 0})
        broker.infer({"tick": 1})

        def fail(obs):
            raise RuntimeError("inference failed")

        policy.infer = fail
        broker.infer({"tick": 2})
        with pytest.raises(RuntimeError, match="inference failed"):
            broker.infer({"tick": 3})
    finally:
        broker.close()


def test_dual_block_response_requires_control_space_tolerance():
    class MissingTolerance(Policy):
        def get_server_metadata(self):
            metadata = super().get_server_metadata()
            metadata["ttc"]["blocks"] = {"actions": 7, "rotation_tolerance": 6}
            return metadata

    broker = TTCActionChunkBroker(MissingTolerance(), replan_steps=2, delay_steps=1)
    try:
        with pytest.raises(ValueError, match="control rotation_tolerance"):
            broker.infer({"tick": 0})
    finally:
        broker.close()
