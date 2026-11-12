"""Complete the same frozen upload using compact, byte-identical reconstruction."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback

ROOT = Path('/media/czl/sata/franka_my_code/openpi')
RECORD = Path(__file__).resolve().parent
DATA = ROOT / 'data/lerobot/caozilin/libero_reconstructed_joint13_successmix20_20261109'
RUN = '/mnt/SharedData/czl/runs/pi05_libero_joint13_successmix20_lora_6gpu_bs72_10k_20261109'
REMOTE = '/mnt/SharedData/czl/lerobot/caozilin/libero_reconstructed_joint13_successmix20_20261109'
SSH = ['ssh', '-S', '/tmp/libero_joint13_20261109.sock', '-o', 'BatchMode=yes', 'czl@100.73.28.99']
RSH = 'ssh -S /tmp/libero_joint13_20261109.sock -o BatchMode=yes'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def stage(name, **fields):
    write(RECORD / 'upload_state.json', dict(stage=name, method='exact_compact_transport', **fields))


def main():
    with (RECORD / 'upload.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        transition = read(RECORD / 'transport_transition/local_stop_verification.json')
        assert transition['complete'] and transition['only_original_uploader_and_rsync_stopped']
        preparation = read(RECORD / 'exact_transport_preparation.json')
        assert preparation['complete'] and preparation['episodes'] == 1875
        assert read(RECORD / 'local_pipeline_state.json')['stage'] == 'complete_local_dataset'
        manifest_sha = sha(DATA / 'transfer_manifest.json')
        assert manifest_sha == preparation['transfer_manifest_sha256']
        bundle = Path(preparation['bundle'])
        assert sha(bundle / 'transport_request.json') == preparation['transport_request_sha256']
        helper = ROOT / 'examples/libero/reconstruct_joint13_exact_transport_server.py'
        assert sha(helper) == preparation['reconstructor_sha256']
        write(RECORD / 'upload_process.json', dict(pid=os.getpid(), method='exact_compact_transport'))
        stage('uploading_compact_transport', transport_bytes=preparation['transport_bytes'], original_dataset_bytes=preparation['original_dataset_bytes'], transfer_manifest_sha256=manifest_sha)
        subprocess.run(['rsync', '-rltL', '--checksum', '--partial', '--stats', '--files-from=' + str(RECORD / 'exact_transport_files.txt'), '-e', RSH, str(bundle) + '/', f'czl@100.73.28.99:{RUN}/exact_transport_bundle/'], check=True)
        manifest = read(DATA / 'transfer_manifest.json')
        paths = [e['path'] for e in manifest['files'] if not e['path'].startswith('data/')]
        paths.append('transfer_manifest.json')
        metadata = RECORD / 'exact_transport_metadata_files.txt'
        metadata.write_text(''.join(name + '\n' for name in paths))
        subprocess.run(['rsync', '-rlt', '--checksum', '--files-from=' + str(metadata), '-e', RSH, str(DATA) + '/', f'czl@100.73.28.99:{REMOTE}/'], check=True)
        subprocess.run(['rsync', '-a', '-e', RSH, str(helper), str(ROOT / 'examples/libero/verify_successmix20_joint13_upload_server.py'), f'czl@100.73.28.99:{RUN}/'], check=True)
        stage('remote_reconstructing_exact_sha', transfer_manifest_sha256=manifest_sha)
        command = f'CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/czl/openpi/.venv/bin/python -u {RUN}/{helper.name} --run-dir {RUN}'
        result = subprocess.run([*SSH, command], check=False)
        assert result.returncode in (0, 2), ('Reconstructor infrastructure failure', result.returncode)
        receipt = RECORD / 'exact_transport_reconstruction_completion.json'
        subprocess.run(['rsync', '-a', '-e', RSH, f'czl@100.73.28.99:{RUN}/exact_transport_reconstruction_completion.json', str(receipt)], check=True)
        reconstruction = read(receipt)
        assert reconstruction['transfer_manifest_sha256'] == manifest_sha
        if reconstruction['errors']:
            # A codec/serialization mismatch never changes the accepted dataset.
            # Send the original audited Parquet for precisely those entries instead.
            fallback = RECORD / 'exact_transport_fallback_files.txt'
            fallback.write_text(''.join(e['relative_path'] + '\n' for e in reconstruction['errors']))
            stage('uploading_original_fallback_files', files=len(reconstruction['errors']), transfer_manifest_sha256=manifest_sha)
            subprocess.run(['rsync', '-rlt', '--checksum', '--partial', '--files-from=' + str(fallback), '-e', RSH, str(DATA) + '/', f'czl@100.73.28.99:{REMOTE}/'], check=True)
        stage('remote_verifying', transfer_manifest_sha256=manifest_sha)
        command = f'HF_DATASETS_CACHE={RUN}/reader_cache JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/czl/openpi/.venv/bin/python -u {RUN}/verify_successmix20_joint13_upload_server.py'
        subprocess.run([*SSH, command], check=True)
        subprocess.run(['rsync', '-a', '-e', RSH, f'czl@100.73.28.99:{RUN}/lerobot_upload_verification.json', str(RECORD / 'lerobot_upload_verification.json')], check=True)
        verification = read(RECORD / 'lerobot_upload_verification.json')
        assert verification['complete'] and verification['transfer_manifest_sha256'] == manifest_sha
        write(RECORD / 'exact_transport_completion.json', dict(complete=True, dataset_bytes_identical_to_frozen_manifest=True,
              transfer_manifest_sha256=manifest_sha, preexisting_verified=sum(e['status'] == 'preexisting_verified' for e in reconstruction['results']),
              rebuilt_verified=sum(e['status'] == 'rebuilt_exact_sha' for e in reconstruction['results']),
              original_file_fallbacks=len(reconstruction['errors']), verification_sha256=sha(RECORD / 'lerobot_upload_verification.json')))
        stage('complete', files=len(manifest['files']), bytes=manifest['total_bytes'], transfer_manifest_sha256=manifest_sha, training_pending=True)


if __name__ == '__main__':
    try:
        main()
    except BlockingIOError:
        raise SystemExit('An uploader already holds the lock; existing state was left untouched.')
    except BaseException:
        stage('failed', traceback=traceback.format_exc())
        raise
