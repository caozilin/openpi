"""Asynchronous trained RTC client, with a bounded committed action prefix."""

import copy
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from openpi_client import base_policy


class TTCActionChunkBroker(base_policy.BasePolicy):
    """Return one control action per call while one chunk request runs in the background.

    Calls must correspond to consecutive control ticks. A request conditions on
    the next ``delay_steps`` actions. If it overruns that budget, wait rather
    than execute actions outside the committed prefix. The first request blocks.
    Only the worker owns an in-flight connection; reset drains stale responses.
    """

    def __init__(self, policy, replan_steps=5, delay_steps=2):
        protocol = policy.get_server_metadata().get("ttc", {})
        if protocol.get("version") != 1 or protocol.get("space") != "model_normalized":
            raise ValueError("TTC requires a server advertising the trained TTC v1 protocol")
        horizon = protocol["action_horizon"]
        if (
            type(delay_steps) is not int
            or not 0 <= delay_steps <= protocol["max_delay"]
            or type(replan_steps) is not int
            or not max(delay_steps, 1) <= replan_steps
            or replan_steps + delay_steps > horizon
        ):
            raise ValueError("TTC requires delay <= replan_steps and replan_steps + delay <= action_horizon")
        self._policy = policy
        self._protocol = protocol
        self._horizon = horizon
        self._replan_steps = replan_steps
        self._delay = delay_steps
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ttc")
        self._future = None
        self._results = None
        self._step = 0
        self._chunk_start = 0
        self._request_start = 0
        self._next_request_at = replan_steps
        self._closed = False

    def _request(self, obs, delay, prefixes):
        request = copy.deepcopy(obs)
        if "_ttc" in request:
            raise ValueError("TTC broker manages the reserved _ttc request field")
        request["_ttc"] = {"version": 1, "space": "model_normalized", "delay": delay, "prefixes": prefixes}
        return request

    def _validate_result(self, result):
        payload = result.get("ttc", {})
        blocks = payload.get("blocks", {})
        if (
            payload.get("version") != 1
            or payload.get("space") != "model_normalized"
            or set(blocks) != set(self._protocol["blocks"])
        ):
            raise ValueError("TTC response is missing the advertised model-space prior blocks")
        for name, dim in self._protocol["blocks"].items():
            values = np.asarray(blocks[name])
            if (
                values.shape != (self._horizon, dim)
                or not np.issubdtype(values.dtype, np.floating)
                or not np.isfinite(values).all()
            ):
                raise ValueError("Invalid TTC response prior block: " + name)
        actions = np.asarray(result["actions"])
        if actions.ndim != 2 or len(actions) != self._horizon or not np.isfinite(actions).all():
            raise ValueError("Invalid TTC response control actions")
        if "rotation_tolerance" in blocks:
            tolerance = np.asarray(result.get("rotation_tolerance"))
            if (
                tolerance.shape != (self._horizon, 6)
                or not np.issubdtype(tolerance.dtype, np.floating)
                or not np.isfinite(tolerance).all()
            ):
                raise ValueError("Invalid TTC response control rotation_tolerance")
        return result

    def infer(self, obs):
        if self._closed:
            raise RuntimeError("TTC broker is closed")
        if self._results is None:
            self._results = self._validate_result(self._policy.infer(self._request(obs, 0, {})))

        if self._future is not None and self._step >= self._request_start + self._delay:
            # Wait here on an overrun; no uncommitted action is consumed.
            result = self._validate_result(self._future.result())
            old_offset = self._request_start - self._chunk_start
            for name, block in result["ttc"]["blocks"].items():
                expected = self._results["ttc"]["blocks"][name][old_offset : old_offset + self._delay]
                if not np.array_equal(block[: self._delay], expected):
                    raise ValueError("TTC server changed the committed prior: " + name)
            self._results = result
            self._chunk_start = self._request_start
            self._future = None

        if self._future is None and self._step >= self._next_request_at:
            offset = self._step - self._chunk_start
            prefixes = {}
            for name, values in self._results["ttc"]["blocks"].items():
                prefix = np.zeros_like(values)
                prefix[: self._delay] = values[offset : offset + self._delay]
                prefixes[name] = prefix
            self._request_start = self._step
            request = self._request(obs, self._delay, prefixes)
            self._future = self._executor.submit(self._policy.infer, request)
            self._next_request_at = self._step + self._replan_steps
            if self._delay == 0:
                # No prior is committed; wait before consuming the new chunk.
                self._results = self._validate_result(self._future.result())
                self._chunk_start = self._request_start
                self._future = None

        offset = self._step - self._chunk_start
        if offset >= self._horizon:
            raise RuntimeError("TTC action chunk exhausted")
        result = {
            name: value[offset].copy()
            if isinstance(value, np.ndarray) and value.ndim >= 2 and value.shape[0] == self._horizon
            else value
            for name, value in self._results.items()
            if name != "ttc"
        }
        self._step += 1
        return result

    def reset(self):
        try:
            if self._future is not None:
                self._future.result()
        finally:
            self._future = None
            self._results = None
            self._step = self._chunk_start = self._request_start = 0
            self._next_request_at = self._replan_steps
            self._policy.reset()

    def close(self):
        self._closed = True
        self._executor.shutdown(wait=True)

    def get_server_metadata(self):
        return self._policy.get_server_metadata()
