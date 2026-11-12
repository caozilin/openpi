"""Package audited numbers and original compressed videos; preserve dataset bytes."""
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path('/media/czl/sata/franka_my_code/openpi')
RECORD = Path(__file__).resolve().parent
DATA = ROOT / 'data/lerobot/caozilin/libero_reconstructed_joint13_successmix20_20261109'
BUNDLE = Path('/tmp/libero_successmix20_exact_transport_20261109')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    manifest_path = DATA / 'transfer_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads((DATA / 'meta/joint13_independent_audit.json').read_text())
    assert audit['complete'] and audit['episodes'] == 1875
    request = json.loads((DATA / 'replacement_request.json').read_text())
    selected = {e['episode_index']: e for e in request['entries']}
    expected = {e['path']: e for e in manifest['files']}
    BUNDLE.mkdir(exist_ok=True)
    entries = []
    for i in range(1875):
        relative = f'data/chunk-{i // 1000:03d}/episode_{i:06d}.parquet'
        source = DATA / relative
        schema = pq.read_schema(source)
        schema_bytes = schema.serialize().to_pybytes()
        schema_sha = hashlib.sha256(schema_bytes).hexdigest()
        schema_relative = f'schemas/{schema_sha}.arrow'
        schema_path = BUNDLE / schema_relative
        schema_path.parent.mkdir(exist_ok=True)
        if not schema_path.exists():
            schema_path.write_bytes(schema_bytes)
        numbers = pq.read_table(source, columns=[n for n in schema.names if n not in ('image', 'wrist_image')])
        numeric_relative = f'numeric/{i:06d}.arrow'
        numeric_path = BUNDLE / numeric_relative
        numeric_path.parent.mkdir(exist_ok=True)
        with pa.ipc.new_file(numeric_path, numbers.schema, options=pa.ipc.IpcWriteOptions(compression='zstd')) as writer:
            writer.write_table(numbers)
        entry = dict(episode_index=i, relative_path=relative, expected_sha256=expected[relative]['sha256'],
                     expected_bytes=expected[relative]['bytes'], frames=numbers.num_rows,
                     numeric_relative=numeric_relative, numeric_sha256=sha(numeric_path),
                     schema_relative=schema_relative, schema_sha256=schema_sha,
                     trajectory_is_tolerance=int(i in selected), videos={})
        if i in selected:
            item = selected[i]
            for camera, name in (('image', 'main_rgb.mp4'), ('wrist_image', 'wrist_rgb.mp4')):
                original = Path(item['replay']['folder']) / name
                assert sha(original) == item['replay']['artifact_sha256'][name]
                video_relative = f'videos/{i:06d}/{name}'
                link = BUNDLE / video_relative
                link.parent.mkdir(parents=True, exist_ok=True)
                if link.is_symlink():
                    assert link.resolve() == original.resolve()
                else:
                    assert not link.exists()
                    link.symlink_to(original)
                entry['videos'][camera] = dict(path=video_relative, sha256=sha(original), bytes=original.stat().st_size)
        entries.append(entry)
        if (i + 1) % 100 == 0:
            print('Packaged', i + 1, '/1875', flush=True)
    helper = ROOT / 'examples/libero/reconstruct_joint13_exact_transport_server.py'
    transport = dict(schema='libero_exact_transport/1.0', transfer_manifest_sha256=sha(manifest_path),
                     reconstructor_sha256=sha(helper), episodes=1875, frames=312374,
                     nominal_episodes=1574, tolerance_episodes=301, entries=entries,
                     compatibility_evidence='transport_compatibility/full_pilot_reconstruction.json',
                     protocol='transport only; every final Parquet must match the original frozen SHA256')
    (BUNDLE / 'transport_request.json').write_text(json.dumps(transport, indent=2) + '\n')
    files = sorted(p for p in BUNDLE.rglob('*') if p.is_file())
    (RECORD / 'exact_transport_files.txt').write_text(''.join(str(p.relative_to(BUNDLE)) + '\n' for p in files))
    receipt = dict(complete=True, bundle=str(BUNDLE), files=len(files), transport_bytes=sum(p.stat().st_size for p in files),
                   original_dataset_bytes=manifest['total_bytes'], episodes=1875, tolerance_videos=602,
                   transfer_manifest_sha256=sha(manifest_path), transport_request_sha256=sha(BUNDLE / 'transport_request.json'),
                   reconstructor_sha256=sha(helper), dataset_modified=False)
    (RECORD / 'exact_transport_preparation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
