"""Check all six existing replicas with one recorded observation each; no physics."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'packages/openpi-client/src')]
import imageio.v2 as imageio
import numpy as np
import websockets.sync.client
from openpi_client import msgpack_numpy
from examples.libero.eval_nominal50 import RUN, IDENTITY, dump

episode = next(RUN.glob('episodes/*/task_*/seed_*/result.json'))
result = json.loads(episode.read_text())
reader = imageio.get_reader(episode.parent / 'rollout.mp4')
frame = reader.get_data(0)
reader.close()
assert frame.shape == (256, 512, 3)


def check(index):
    started = time.monotonic()
    with websockets.sync.client.connect(f'ws://127.0.0.1:{18760 + index}',
            compression=None, max_size=None, ping_interval=None, ping_timeout=None,
            open_timeout=30) as socket:
        metadata = msgpack_numpy.unpackb(socket.recv(timeout=30))
        assert metadata['eval_identity'] == IDENTITY and metadata['replica'] == index
        packet = {'observation/image': np.ascontiguousarray(frame[:, :256]),
                  'observation/wrist_image': np.ascontiguousarray(frame[:, 256:]),
                  'observation/state': np.zeros(8, dtype=np.float32),
                  'prompt': result['language'], '_libero_eval_noise_seed': 0}
        socket.send(msgpack_numpy.Packer().pack(packet))
        response = socket.recv(timeout=60)
        assert not isinstance(response, str), response
        prediction = msgpack_numpy.unpackb(response)
        actions = np.asarray(prediction['actions'])
        assert actions.shape == (10, 7) and np.isfinite(actions).all()
    return dict(replica=index, gpu=metadata['gpu'], response_seconds=time.monotonic()-started,
                model_identity=metadata['eval_identity'], action_shape=list(actions.shape), finite=True)


with ThreadPoolExecutor(max_workers=6) as pool:
    checks = list(pool.map(check, range(6)))
dump(RUN / 'resume_inference_response_audit.json', dict(complete=True, replicas=checks,
    source_recorded_episode=str(episode), scope='Transport/model readiness only; no simulation or scored episode'))
print(json.dumps(checks), flush=True)
