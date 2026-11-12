"""Preserve committed episodes and archive interrupted, uncommitted attempts."""
from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.libero.eval_nominal50 import RUN, read, dump, validate


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    validate(read(RUN / 'evaluation_request.json'))
    locks = []
    for name in ('pipeline.lock', 'evaluation.lock', 'independent_audit.lock'):
        handle = (RUN / name).open('a')
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(handle)
    records = ['pipeline_process.json', 'evaluation_process.json',
               'independent_audit_process.json'] + [f'client_{i}_process.json' for i in range(6)]
    for name in records:
        pid = read(RUN / name)['pid']
        assert not Path(f'/proc/{pid}').exists(), (name, pid, 'Inspect live process before resuming')
    committed = sorted(RUN.glob('episodes/*/task_*/seed_*/result.json'))
    assert len(committed) == 1503
    artifacts = {}
    keys = []
    for path in committed:
        result = read(path)
        assert result['complete']
        keys.append((result['suite'], result['task_id'], result['seed']))
        for name in ('result.json', 'trajectory.json', 'trajectory.npz', 'rollout.mp4'):
            artifact = path.parent / name
            artifacts[str(artifact)] = sha(artifact)
    assert len(keys) == len(set(keys))
    pending = sorted(p for p in RUN.glob('episodes/*/task_*/seed_*')
                     if p.is_dir() and not (p / 'result.json').exists())
    errors = list(RUN.glob('episodes/*/task_*/seed_*/infrastructure_error.json'))
    assert len(errors) == 0
    stamp = datetime.datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')
    archive = RUN / 'attempts' / f'0002_host_restart_{stamp}'
    archive.mkdir(parents=True, exist_ok=False)
    snapshot = archive / 'run_snapshot'
    snapshot.mkdir()
    for path in RUN.iterdir():
        if path.is_file() and path.suffix in ('.json', '.log'):
            shutil.copy2(path, snapshot / path.name)
    receipt = dict(complete=False, resumed_at=datetime.datetime.now().astimezone().isoformat(),
        authorization='User requested continuation and monitoring every15minutes on2026-11-06',
        reason='Processes disappeared without terminal state; current host boot2026-11-06 15:33:14, last result update11:32; exact shutdown cause unconfirmed',
        committed_episodes=len(committed), committed_successes=sum(read(p)['success'] for p in committed),
        committed_artifact_sha256=artifacts, historical_infrastructure_errors=3,
        archived_uncommitted_directories=[str(p.relative_to(RUN)) for p in pending],
        request_sha256=sha(RUN / 'evaluation_request.json'),
        completed_result_selection='Preserve every committed success and failure without replacement',
        remaining_episodes=1600-len(committed), archive=str(archive))
    dump(archive / 'resume_receipt.json', receipt)
    for path in pending:
        target = archive / path.relative_to(RUN)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(target))
    assert not list(RUN.glob('episodes/*/task_*/seed_*/infrastructure_error.json'))
    assert all(sha(Path(p)) == digest for p, digest in artifacts.items())
    receipt['complete'] = True
    dump(archive / 'resume_receipt.json', receipt)
    dump(RUN / 'resume_after_host_restart.json', receipt)
    with (RUN / 'RESUME_HISTORY.md').open('a') as stream:
        stream.write('\n## 第二次恢复\n\n11月6日15:41检查发现本机15:33启动，原评测/审计进程消失，日志停在11:32，确切中断原因未证实。\n'
                     '保留1503条已提交回合（1433成功及70任务失败）全部6012份artifact SHA；只归档未提交目录及旧状态日志，继续97条。\n'
                     f'归档证据：`{archive.relative_to(RUN)}`。活动infra错误0，历史3次超时仍保留。\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'committed_artifact_sha256'}), flush=True)


if __name__ == '__main__':
    main()
