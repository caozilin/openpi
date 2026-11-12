"""Request-local TaskTol noise is independent of replica/global RNG history."""
import asyncio

import numpy as np
from openpi_client import msgpack_numpy
import pytest
from websockets.exceptions import ConnectionClosedOK

from openpi.serving.websocket_policy_server import WebsocketPolicyServer


class Connection:
    remote_address = ("test", 1)

    def __init__(self, requests):
        self.requests = iter(requests)
        self.sent = []
        self.closed = False

    async def send(self, data):
        self.sent.append(data)

    async def recv(self):
        try:
            return msgpack_numpy.packb(next(self.requests))
        except StopIteration:
            raise ConnectionClosedOK(None, None) from None

    async def close(self, **kwargs):
        self.closed = True


class Policy:
    def __init__(self):
        self.noises = []

    def infer(self, obs, *, noise=None):
        assert "_tasktol_sampling_seed" not in obs
        self.noises.append(noise)
        return {"actions": np.zeros((16, 7))}


def test_request_noise_repeats_across_interleaved_connections():
    policy = Policy()
    server = WebsocketPolicyServer(policy, metadata={
        "tasktol_sampling": {"version": 1, "action_horizon": 16, "action_dim": 13},
    })
    asyncio.run(server._handler(Connection([  # noqa: SLF001
        {"_tasktol_sampling_seed": 42}, {"_tasktol_sampling_seed": 9}, {"_tasktol_sampling_seed": 42},
    ])))
    asyncio.run(server._handler(Connection([{"_tasktol_sampling_seed": 42}])))  # noqa: SLF001
    np.testing.assert_array_equal(policy.noises[0], policy.noises[2])
    np.testing.assert_array_equal(policy.noises[0], policy.noises[3])
    assert not np.array_equal(policy.noises[0], policy.noises[1])
    assert policy.noises[0].shape == (16, 13)
    assert policy.noises[0].dtype == np.float32


@pytest.mark.parametrize("seed", [-1, 2**32, 1.5, True])
def test_bad_request_seeds_fail_before_policy_inference(seed):
    policy = Policy()
    server = WebsocketPolicyServer(policy, metadata={"tasktol_sampling": {"version": 1}})
    connection = Connection([{"_tasktol_sampling_seed": seed}])
    with pytest.raises(ValueError, match="sampling seed"):
        asyncio.run(server._handler(connection))  # noqa: SLF001
    assert connection.closed
    assert not policy.noises


def test_legacy_unseeded_request_is_unchanged():
    policy = Policy()
    server = WebsocketPolicyServer(policy)
    asyncio.run(server._handler(Connection([{"prompt": "pick"}])))  # noqa: SLF001
    assert policy.noises == [None]
