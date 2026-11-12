"""Freeze formal400 provenance and merge a disjoint audited1600 extension."""

import argparse
import csv
import datetime
import hashlib
from pathlib import Path

from examples.libero.eval_nominal50 import ROOT, RUN, BASELINE, IDENTITY, REMOTE, SUITES, assignments, read, dump
from examples.libero.summarize_nominal10 import sha, stats, audited_rows


def key(row):
    return row['suite'], row['task_id'], row['seed']


def merge(old, new):
    assert len(old) == 400 and {key(r) for r in old} == {(s, t, z) for s in SUITES for t in range(10) for z in range(10)}
    assert len(new) == 1600 and {key(r) for r in new} == {(s, t, z) for s in SUITES for t in range(10) for z in range(10, 50)}
    assert all(r['solver_failure_policy'] == 'hold_last_accepted_reference' for r in old + new)
    return sorted(old + new, key=lambda r: (SUITES.index(r['suite']), r['task_id'], r['seed']))


def freeze_baseline(run):
    formal = read(BASELINE / 'combined_formal_400.json')
    assert formal['complete'] and formal['formal_baseline'] and not formal['targeted_failed_retests_included']
    assert formal['episodes'] == 400 and formal['successes'] == 380
    manifests = {str(BASELINE / name): sha(BASELINE / name) for name in
        ('combined_formal_400.json', 'formal_400_audit.json', 'task_completion.json')}
    for source in formal['audit_chain']:
        p = Path(source['path'])
        assert sha(p) == source['sha256']
        manifests[str(p)] = source['sha256']
    audits = {p: read(Path(p)) for p in {r['source_audit'] for r in formal['results']}}
    artifacts = {}
    for r in formal['results']:
        assert 'failed_retry' not in r['source_result']
        audit = audits[r['source_audit']]
        assert audit['complete']
        evidence = next(e for e in audit['results'] if key(e) == key(r))
        result_path = Path(r['source_result'])
        result = read(result_path)
        assert result['complete'] and result['success'] == evidence['success'] == r['success']
        assert result['max_consecutive_solver_failures'] == 20
        assert result['server_metadata']['checkpoint'].endswith('/10000')
        for name, digest in evidence['artifact_sha256'].items():
            p = result_path.parent / name
            assert sha(p) == digest, p
            artifacts[str(p)] = digest
    receipt = dict(complete=True, formal_baseline=True, episodes=400, successes=380,
        targeted_failed_retests_included=False, results=formal['results'],
        artifact_sha256=artifacts, manifest_sha256=manifests,
        success_tail_seconds=0, success_criterion='first_native_success',
        checked_at=datetime.datetime.now().astimezone().isoformat())
    dump(run / 'preserved_formal400.json', receipt)
    return receipt


def prepare(run):
    import numpy as np
    import torch
    from examples.libero.success_continuation import request_fields

    run.mkdir(parents=True, exist_ok=True)
    if (run / 'evaluation_request.json').exists():
        raise ValueError('Request already frozen; do not overwrite')
    receipt = freeze_baseline(run)
    tasks = read(BASELINE / 'official_tasks.json')
    arrays = {}
    with np.load(BASELINE / 'official_initial_states.npz', allow_pickle=False) as previous:
        for task in tasks:
            path = Path(task['source_init_states'])
            assert sha(path) == task['source_init_sha256']
            initial = torch.load(path, weights_only=False, map_location='cpu')
            assert len(initial) >= 50
            for seed in range(50):
                state = initial[seed].detach().cpu().numpy() if hasattr(initial[seed], 'detach') else np.asarray(initial[seed])
                assert np.isfinite(state).all()
                name = f"{task['suite']}_{task['task_id']}_{seed}"
                arrays[name] = state
                if seed < 6:
                    assert np.array_equal(previous[name], state)
    np.savez_compressed(run / 'official_initial_states.npz', **arrays)
    dump(run / 'official_tasks.json', tasks)
    server = (BASELINE / 'serve_formal_nominal_server.py').read_text()
    old_identity = read(BASELINE / 'evaluation_request.json')['server_eval_identity']
    old_remote = read(BASELINE / 'evaluation_request.json')['remote_inference_run']
    server = server.replace(old_remote, REMOTE).replace(old_identity, IDENTITY)
    server_path = run / 'serve_nominal50_server.py'
    server_path.write_text(server)
    baseline_request = read(BASELINE / 'evaluation_request.json')
    source_paths = [Path(p) for p in baseline_request['source_sha256'] if 'frozen_shared_runtime' in p]
    source_paths += [ROOT / name for name in (
        'examples/libero/tasktol_actions.py', 'experiments/libero_tasktol_compat/mujoco_joint_servo.py',
        'examples/libero/eval_reconstructed_libero_local.py', 'examples/libero/solver_failure_hold.py',
        'examples/libero/success_continuation.py', 'examples/libero/eval_nominal50.py',
        'examples/libero/eval_formal_nominal.py', 'examples/libero/audit_formal_nominal.py',
        'examples/libero/nominal50_protocol.py', 'examples/libero/nominal50_pipeline.py',
        'examples/libero/summarize_nominal10.py')]
    source_paths += [server_path, run / 'official_tasks.json', run / 'official_initial_states.npz', run / 'preserved_formal400.json']
    request = {k: v for k, v in baseline_request.items() if k in (
        'client_location', 'server_role', 'tasks', 'gpu_ids', 'ports', 'checkpoint_step', 'replan_steps',
        'action_horizon', 'state_enabled', 'controller', 'max_steps', 'gripper_deadband',
        'connection_policy', 'tolerance_enabled', 'semantic_mask', 'solver_failure_policy',
        'max_consecutive_solver_failures', 'solver_failure_timeout_seconds')}
    request.update(protocol='nominal50.seed10to49.v1', authorized=True, expected_episodes=1600,
        seeds=list(range(10, 50)), target_combined_episodes=2000, target_trials_per_task=50,
        worker_assignments=assignments(), selected_episodes=[j for group in assignments().values() for j in group],
        source_sha256={str(p.resolve()): sha(p) for p in source_paths},
        preserved_manifest_sha256=receipt['manifest_sha256'],
        baseline_run=str(BASELINE), formal_baseline_manifest=str(BASELINE / 'combined_formal_400.json'),
        eval_identity=IDENTITY, server_eval_identity=IDENTITY, remote_inference_run=REMOTE,
        server_entry_name=server_path.name, server_entry_sha256=sha(server_path),
        authorization='User: expand the prior400 nominal rollouts to50 per task using six server GPUs.',
        selection_rule='Keep formal seed0..9 unchanged; execute all seed10..49 once, independent of outcomes; no failed-only retries.',
        success_tail_difference='Preserved400 stopped on first success; new1600 continue1s per latest user default. All success rates use first native success.',
        created_at=datetime.datetime.now().astimezone().isoformat(), **request_fields())
    dump(run / 'evaluation_request.json', request)
    from examples.libero.eval_nominal50 import validate
    validate(request)
    print(dict(prepared=True, expected_new=1600, preserved=400, initial_states=len(arrays)), flush=True)


def finalize(run):
    receipt = read(run / 'preserved_formal400.json')
    request = read(run / 'evaluation_request.json')
    for path, expected in receipt['artifact_sha256'].items():
        assert sha(Path(path)) == expected, path
    for path, expected in receipt['manifest_sha256'].items():
        assert sha(Path(path)) == expected, path
    new, audit = audited_rows(run, 'results', 'new_once_seed10to49_hold20_postsuccess1s')
    rows = merge(receipt['results'], new)
    summary = stats(rows)
    assert summary['episodes'] == 2000 and all(t['episodes'] == 50 for t in summary['per_task'])
    result = dict(complete=True, formal_baseline=True, **summary, results=rows,
        new_episodes=1600, preserved_episodes=400, targeted_failed_retests_included=False,
        selection_rule=request['selection_rule'], success_tail_difference=request['success_tail_difference'],
        source_audit=audit, preserved_receipt_sha256=sha(run / 'preserved_formal400.json'))
    dump(run / 'combined_formal_2000.json', result)
    with (run / 'combined_formal_2000.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    added = stats(new)
    lines = ['# 正式无容差50次/任务评测', '',
        f"共{summary['successes']}/2000成功（{summary['success_rate']:.2%}）；40任务各seed0–49共50次。",
        f"保留正式400：380/400；新增seed10–49：{added['successes']}/1600。", '',
        '只按预定seed分区合并，未按结果挑选或替换失败回合。',
        '固定最终10000步模型、本地仿真、服务器6卡推理、mask000、失败保持参考且连续20帧终止。',
        '新增回合成功后继续闭环1秒，旧400成功即结束；成功率统一按首次原生成功，不以最终稳定成功替换。', '',
        '|任务集|成功/500|成功率|', '|---|---:|---:|']
    for suite in SUITES:
        s = summary['per_suite'][suite]
        lines.append(f"|{suite}|{s['successes']}/500|{s['successes']/500:.2%}|")
    lines += ['', '|任务集|task_id（0起始）|成功/50|', '|---|---:|---:|']
    for t in summary['per_task']:
        lines.append(f"|{t['suite']}|{t['task_id']:02d}|{t['successes']}/50|")
    lines += ['', '新增1600全部原生成功谓词、初态、每帧动作/状态/25物理步、失败保持和计数、成功后延长段、视频全帧读取及artifact SHA经独立审计。旧400所有1600份artifact SHA再次核对。']
    (run / 'NOMINAL_50SEED_REPORT.md').write_text('\n'.join(lines) + '\n')
    return dict(complete=True, episodes=2000, successes=summary['successes'], new_successes=added['successes'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'finalize'))
    parser.add_argument('--run-dir', type=Path, default=RUN)
    args = parser.parse_args()
    print(prepare(args.run_dir) if args.action == 'prepare' else finalize(args.run_dir), flush=True)


if __name__ == '__main__':
    main()
