import asyncio
import queue
import threading

import jax
import numpy as np
from openpi_client.ttc_action_chunk_broker import TTCActionChunkBroker
from openpi_client.websocket_client_policy import WebsocketClientPolicy
import websockets.asyncio.server

from openpi.models import rtc
from openpi.models.rtc_test import small_pi05  # noqa: F401
from openpi.policies.policy import Policy
from openpi.serving.websocket_policy_server import WebsocketPolicyServer


def test_real_websocket_client_server_ttc_and_legacy_requests(small_pi05):  # noqa: F811
    model, obs, _ = small_pi05
    model.ttc = rtc.TTCConfig(enabled=True)
    policy = Policy(model, output_transforms=[lambda data: {"actions": data["actions"][:, :2]}])
    server = WebsocketPolicyServer(policy, metadata=policy.metadata)
    ready = queue.Queue()

    async def serve():
        stop = asyncio.Event()
        async with websockets.asyncio.server.serve(
            server._handler, "127.0.0.1", 0, compression=None, max_size=None  # noqa: SLF001
        ) as listener:
            ready.put((listener.sockets[0].getsockname()[1], asyncio.get_running_loop(), stop))
            await stop.wait()

    thread = threading.Thread(target=lambda: asyncio.run(serve()), daemon=True)
    thread.start()
    port, loop, stop = ready.get(timeout=5)
    client = WebsocketClientPolicy("127.0.0.1", port)
    broker = None
    try:
        observation = jax.tree.map(lambda x: np.asarray(x[0]), obs.to_dict())
        legacy = client.infer(observation)
        assert legacy["actions"].shape == (5, 2)
        assert "ttc" not in legacy
        broker = TTCActionChunkBroker(client, replan_steps=2, delay_steps=2)
        for _ in range(10):
            action = broker.infer(observation)["actions"]
            assert action.shape == (2,)
            assert np.isfinite(action).all()
        broker.reset()
        assert broker.infer(observation)["actions"].shape == (2,)
    finally:
        if broker is not None:
            broker.close()
        client._ws.close()  # noqa: SLF001
        loop.call_soon_threadsafe(stop.set)
        thread.join(timeout=5)
        assert not thread.is_alive()
