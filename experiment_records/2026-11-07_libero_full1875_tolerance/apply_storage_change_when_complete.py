"""Apply the tested future default only after this frozen batch fully exits."""
from pathlib import Path
import hashlib
import json
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = Path('/media/czl/sata/franka_my_code/franka_mujoco')
RUN = ROOT / 'analysis_results/libero_relative_clearance_eps01_ema85_full1875_20261107'

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    receipt = HERE / 'storage_change_applied.json'
    manifest = read(HERE / 'pending_storage_change.json')
    if receipt.exists():
        assert all(sha(ROOT / x['path']) == x['after_sha256'] for x in manifest['files'])
        print('Already applied; verified current hashes')
        return
    completion = read(RUN / 'task_completion.json')
    audit = read(RUN / 'annotation_independent_final_audit.json')
    assert completion['complete'] and completion['independent_audit_complete']
    assert audit['complete'] and audit['episodes'] == 1875
    for record in ('pipeline_process.json', 'annotating_process.json', 'auditing_process.json', 'recovery_process.json'):
        if not (RUN / record).exists():
            continue
        item = read(RUN / record)
        proc = Path('/proc') / str(item['pid'])
        if proc.exists():
            command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
            assert not (str(RUN) in command or str(HERE / 'pipeline.py') in command or str(HERE / 'recover_gp_then_audit.py') in command), 'Original process still active'
    for item in manifest['files']:
        path = ROOT / item['path']
        assert (sha(path) if path.exists() else None) == item['before_sha256'], f"User changes conflict: {path}"
    patch = HERE / 'pending_storage_change.patch'
    subprocess.run(['git', 'apply', '--check', str(patch)], cwd=ROOT, check=True)
    subprocess.run(['git', 'apply', str(patch)], cwd=ROOT, check=True)
    assert all(sha(ROOT / x['path']) == x['after_sha256'] for x in manifest['files'])
    (HERE / 'storage_change_applied.json').write_text(json.dumps(dict(
        complete=True, applies_to='future new requests only; no existing result migration',
        source_manifest=str(HERE / 'pending_storage_change.json'),
        frozen_batch_completion_sha256=sha(RUN / 'task_completion.json'),
        frozen_batch_audit_sha256=sha(RUN / 'annotation_independent_final_audit.json'),
        production_files=manifest['files']), indent=2)+'\n')
    print('Applied source-demo metadata reference default; existing results untouched')

if __name__ == '__main__':
    main()
