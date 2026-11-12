"""Rebuild uploaded observations from compressed sources, accepting only frozen SHAs."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import traceback


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def reconstruct(entry, bundle, data, old, archive):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import numpy as np
    from PIL import Image
    import imageio.v2 as imageio

    destination = data / entry['relative_path']
    if destination.exists():
        previous = sha(destination)
        if previous == entry['expected_sha256']:
            return dict(episode_index=entry['episode_index'], status='preexisting_verified', sha256=previous)
        saved = archive / entry['relative_path']
        saved.parent.mkdir(parents=True, exist_ok=True)
        assert not saved.exists(), saved
        destination.replace(saved)
        write(saved.with_suffix('.receipt.json'), dict(original_path=str(destination), preserved_path=str(saved),
              observed_sha256=previous, expected_sha256=entry['expected_sha256'], reason='incomplete earlier transfer preserved'))
    schema_path = bundle / entry['schema_relative']
    numeric_path = bundle / entry['numeric_relative']
    assert sha(schema_path) == entry['schema_sha256']
    assert sha(numeric_path) == entry['numeric_sha256']
    schema = pa.ipc.read_schema(pa.BufferReader(schema_path.read_bytes()))
    numeric = pa.ipc.open_file(numeric_path).read_all()
    assert numeric.num_rows == entry['frames']
    columns = {name: numeric[name] for name in numeric.column_names}
    if not entry['trajectory_is_tolerance']:
        original = pq.read_table(old / entry['relative_path'], columns=['image', 'wrist_image'])
        assert original.num_rows == entry['frames']
        columns.update({name: original[name] for name in original.column_names})
    else:
        for camera in ('image', 'wrist_image'):
            video = entry['videos'][camera]
            path = bundle / video['path']
            assert sha(path) == video['sha256']
            frames = []
            reader = imageio.get_reader(str(path), format='ffmpeg')
            try:
                for pixels in reader:
                    assert pixels.shape == (256, 256, 3) and pixels.dtype == np.uint8
                    buffer = io.BytesIO()
                    Image.fromarray(pixels).save(buffer, format='PNG', compress_level=3)
                    frames.append(dict(bytes=buffer.getvalue(), path=None))
            finally:
                reader.close()
            assert len(frames) == entry['frames']
            columns[camera] = pa.array(frames, type=schema.field(camera).type)
    table = pa.Table.from_arrays([columns[field.name] for field in schema], schema=schema)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f'.{destination.name}.reconstruct-{os.getpid()}.tmp')
    pq.write_table(table, temporary, compression='snappy')
    digest = sha(temporary)
    assert digest == entry['expected_sha256'], (entry['episode_index'], digest, entry['expected_sha256'])
    assert temporary.stat().st_size == entry['expected_bytes']
    temporary.replace(destination)
    return dict(episode_index=entry['episode_index'], status='rebuilt_exact_sha', sha256=digest)


def main(run):
    # Match the dependency import order already used successfully on this server.
    import torch
    import pyarrow

    bundle = run / 'exact_transport_bundle'
    request_path = bundle / 'transport_request.json'
    request = json.loads(request_path.read_text())
    assert sha(Path(__file__)) == request['reconstructor_sha256']
    data = Path('/mnt/SharedData/czl/lerobot/caozilin/libero_reconstructed_joint13_successmix20_20261109')
    old = Path('/mnt/SharedData/czl/lerobot/caozilin/libero_reconstructed_full_20261101')
    assert sha(data / 'transfer_manifest.json') == request['transfer_manifest_sha256']
    manifest = json.loads((data / 'transfer_manifest.json').read_text())
    expected = {x['path']: x for x in manifest['files']}
    assert len(request['entries']) == len({e['episode_index'] for e in request['entries']}) == 1875
    for entry in request['entries']:
        original = expected[entry['relative_path']]
        assert entry['expected_sha256'] == original['sha256'] and entry['expected_bytes'] == original['bytes']
    archive = run / 'transport_transition/partial_files'
    with (run / 'exact_transport_reconstruction.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(run / 'exact_transport_reconstruction_process.json', dict(pid=os.getpid(), request_sha256=sha(request_path)))
        results = []
        errors = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = {executor.submit(reconstruct, e, bundle, data, old, archive): e for e in request['entries']}
            for future in concurrent.futures.as_completed(futures):
                entry = futures[future]
                try:
                    results.append(future.result())
                except Exception:
                    errors.append(dict(episode_index=entry['episode_index'], relative_path=entry['relative_path'], traceback=traceback.format_exc()))
                write(run / 'exact_transport_reconstruction_state.json', dict(stage='reconstructing', processed=len(results) + len(errors),
                      preexisting=sum(x['status'] == 'preexisting_verified' for x in results), rebuilt=sum(x['status'] == 'rebuilt_exact_sha' for x in results), errors=len(errors), total=1875))
        result = dict(complete=not errors, episodes_verified=len(results), results=sorted(results, key=lambda x: x['episode_index']),
                      errors=errors, transfer_manifest_sha256=request['transfer_manifest_sha256'],
                      production_dataset_bytes_identical_to_local=not errors, scope='Parquet reconstruction only; full upload verification still required')
        write(run / 'exact_transport_reconstruction_completion.json', result)
        write(run / 'exact_transport_reconstruction_state.json', dict(stage='complete' if not errors else 'needs_original_file_fallback',
              episodes_verified=len(results), errors=len(errors), total=1875))
        print(json.dumps(dict(complete=not errors, verified=len(results), errors=len(errors))), flush=True)
        return 0 if not errors else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(main(args.run_dir))
