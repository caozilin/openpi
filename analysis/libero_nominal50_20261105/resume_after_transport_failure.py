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
    assert len(committed) == 762
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
    assert len(errors) == 3 and all(p.parent in pending for p in errors)
    stamp = datetime.datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')
    archive = RUN / 'attempts' / f'0001_transport_timeout_{stamp}'
    archive.mkdir(parents=True, exist_ok=False)
    snapshot = archive / 'run_snapshot'
    snapshot.mkdir()
    for path in RUN.iterdir():
        if path.is_file() and path.suffix in ('.json', '.log'):
            shutil.copy2(path, snapshot / path.name)
    receipt = dict(complete=False, resumed_at=datetime.datetime.now().astimezone().isoformat(),
        authorization='User requested continuation and monitoring every15minutes on2026-11-06',
        reason='Three inference response timeouts; committed task failures are preserved',
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
    dump(RUN / 'resume_after_transport_failure.json', receipt)
    (RUN / 'RESUME_HISTORY.md').write_text(
        '# 断点恢复记录\n\n'
        '11月5日发生3次推理响应600秒超时，基础设施中断未计入任务失败。\n'
        '11月6日按用户授权恢复专用SSH连接并复用原六个推理副本。\n'
        '保留762条已完成回合（723成功及39任务失败）的全部3048份artifact，逐文件SHA不变。\n'
        f'仅将{len(pending)}个未提交目录和旧错误/日志快照归档至`{archive.relative_to(RUN)}`。\n'
        '剩余838条按原冻结请求继续；不按成败替换已完成回合。历史基础设施错误保留于归档。\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'committed_artifact_sha256'}), flush=True)


if __name__ == '__main__':
    main()
