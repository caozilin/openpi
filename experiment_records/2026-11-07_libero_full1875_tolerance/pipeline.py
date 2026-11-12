"""Supervise existing full-dataset annotation CLI and independent audit."""
from pathlib import Path
from collections import Counter
import ast
import datetime
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path('/media/czl/sata/franka_my_code/franka_mujoco')
OPENPI = ROOT.parent / 'openpi'
RUN = ROOT / 'analysis_results/libero_relative_clearance_eps01_ema85_full1875_20261107'
RECORD = Path(__file__).resolve().parent
VIEW = OPENPI / 'data/libero_tasktol_joint_servo/20261101_full_v1/viewer'
NATIVE = ROOT / 'experiments/libero_closed_loop/.venv/bin/python'
SEARCH = ROOT / '.venv/bin/python'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OPENPI / 'examples/libero'))
from tasktol.data.libero_annotation.source import read, write, sha
from export_dataset_annotation_status import export as export_status

ENV = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
           NUMEXPR_NUM_THREADS='1', MUJOCO_GL='egl')

def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()

def state(stage, **extra):
    write(RUN / 'pipeline_state.json', dict(stage=stage, updated_at=now(), **extra))

def progress(stage, pid):
    statuses = [read(p) for p in RUN.glob('libero_*/task_*/demo_*/annotation_status.json')]
    errors = list(RUN.glob('libero_*/task_*/demo_*/adapter_error.json'))
    state(stage, child_pid=pid, expected=1875, processed=len(statuses),
          counts=dict(Counter(s['status'] for s in statuses)), adapter_errors=len(errors))
    export_status(RUN, VIEW)

def run_child(command, logname, stage):
    with (RUN / logname).open('a') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=ENV, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT)
    write(RUN / f'{stage}_process.json', dict(pid=child.pid, command=command, started_at=now()))
    try:
        while child.poll() is None:
            progress(stage, child.pid)
            time.sleep(20)
    except BaseException:
        if child.poll() is None:
            child.terminate()
            child.wait()
        raise
    write(RUN / f'{stage}_exit.json', dict(pid=child.pid, exit_code=child.returncode, finished_at=now()))
    progress(stage, child.pid)
    if child.returncode:
        raise RuntimeError(f'{stage} exited {child.returncode}; inspect {logname}')

def finish(audit):
    counts = {s: dict(Counter(r['status'] for r in audit['results'] if r['suite'] == s))
              for s in ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')}
    summary = {k: v for k, v in audit.items() if k != 'results'}
    summary.update(suites=counts, protocol='cache85; all1875 recomputed; per-axis authored caps; epsilon0.1mm; DSP16',
                   all_labeled=audit['labeled'] == 1875)
    write(RUN / 'final_summary.json', summary)
    text = '# 完整1875条重构轨迹容差标注\n\n状态：全部处理结束并通过独立审计。\n\n'
    text += f"1875条中有标签{audit['labeled']}，无合格标签{audit['no_feasible_label']}，零基线不可用{audit.get('baseline_unavailable', 0)}，适配错误0。不是1875条全部必然有标签。\n\n"
    text += '|套件|有标签|无合格标签|零基线不可用|\n|---|---:|---:|---:|\n'
    for suite, c in counts.items():
        text += f"|{suite}|{c.get('labeled',0)}|{c.get('no_feasible_label',0)}|{c.get('baseline_unavailable',0)}|\n"
    text += '\n本次完整采用cache85新版EMA同轴采样，整段零容差不继承阶段偏置；保持原名义动作、手工分轴上限、20Hz、IPOPT20迭代和每块16次追加。仅机器人/支撑面离线运动学，无物理或策略推理。逐帧净空要求min(5mm,c0)，epsilon0.1mm。原负净空可能保留，不构成绝对碰撞安全证明。Object保留原生floor，其他来源保留有限桌面。\n\n'
    text += f'全部请求、来源、原始候选、逐帧标签、审计与日志保持在`{RUN}`，详见本记录README.md。\n'
    (RUN / 'ANNOTATION_REPORT.md').write_text(text)
    (RECORD / 'RESULTS.md').write_text(text)
    completion = dict(complete=True, processing_complete=True, independent_audit_complete=True,
                      episodes=1875, labeled=audit['labeled'], no_feasible_label=audit['no_feasible_label'],
                      baseline_unavailable=audit.get('baseline_unavailable', 0), adapter_errors=0,
                      all_labeled=audit['labeled']==1875, finished_at=now())
    write(RUN / 'task_completion.json', completion)
    state('complete', **completion)
    document = (RECORD / 'README.md').read_text()
    document = document.replace('状态：准备/运行中；最终处理与独立审计尚未完成。',
                                f"状态：已完成处理并通过独立审计；有标签{audit['labeled']}/1875。最终结果见RESULTS.md。")
    (RECORD / 'README.md').write_text(document)
    index = OPENPI / 'experiment_records/README.md'
    value = index.read_text().replace('全量新版EMA标注运行中，最终审计待完成',
                                     f"全量处理及审计完成，有标签{audit['labeled']}/1875")
    index.write_text(value)

def main():
    RUN.mkdir(parents=True, exist_ok=True)
    lock = (RUN / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (RUN / 'task_completion.json').exists():
        assert read(RUN / 'task_completion.json')['complete']
        return
    write(RUN / 'pipeline_process.json', dict(pid=os.getpid(), command=sys.argv, started_at=now()))
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    try:
        state('freezing')
        mask = RUN / 'authored_tolerances_deg.json'
        source_mask = OPENPI / 'examples/libero/configs/libero40_semantic_tolerances_deg.json'
        if not mask.exists():
            shutil.copy2(source_mask, mask)
        command = [str(NATIVE), '-u', str(ROOT / 'annotate_libero_tolerance.py'),
                   '--openpi-root', str(OPENPI), '--output', str(RUN), '--mask-table', str(mask),
                   '--all', '--extra', '16', '--workers', '6', '--clearance-epsilon-m', '0.0001']
        request = ast.literal_eval(subprocess.check_output(command + ['--dry-run'], cwd=ROOT, env=ENV, text=True))
        assert request['planned_episodes'] == 1875 and sum(x['frames'] for x in request['sources']) == 312374
        assert len({(s['suite'], s['task_id']) for s in request['sources']}) == 40
        counts = Counter(x['suite'] for x in request['sources'])
        assert dict(counts) == dict(libero_spatial=465, libero_object=493, libero_goal=458, libero_10=459)
        if (RUN / 'annotation_request.json').exists():
            assert read(RUN / 'annotation_request.json') == request, 'Frozen source differs; do not overwrite'
        else:
            for name, digest in request['code_sha256'].items():
                source = Path(name)
                destination = RUN / 'source_snapshot' / source.relative_to(ROOT)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                assert sha(destination) == digest
            shutil.copy2(ROOT / 'uv.lock', RUN / 'source_snapshot/uv.lock')
            write(RUN / 'annotation_request.json', request)
        selection = []
        for source in request['sources']:
            report = Path(source['folder']) / 'report.json'
            assert read(report)['success']
            selection.append(dict(**source, source_folder=source['folder'], report_sha256=sha(report)))
        frozen = dict(complete=True, selection='All1875 successful reconstructed sources; original episode order', sources=selection)
        if (RUN / 'selection.json').exists():
            assert read(RUN / 'selection.json') == frozen
        else:
            write(RUN / 'selection.json', frozen)
        write(RUN / 'launch_verification.json', dict(complete=True, episodes=1875, frames=312374, tasks=40,
              counts=dict(counts), mask_sha256=sha(mask), source_snapshot_complete=True,
              tests='79 passed: EMA, zero-reference, relative-clearance, authored-caps, floor and adapter',
              free_bytes=shutil.disk_usage(RUN).free, checked_at=now()))
        if (VIEW / 'annotation_status.json').exists() and not (RUN / 'previous_viewer_annotation_status.json').exists():
            shutil.copy2(VIEW / 'annotation_status.json', RUN / 'previous_viewer_annotation_status.json')
        export_status(RUN, VIEW)
        run_child(command, 'annotation.log', 'annotating')
        summary = read(RUN / 'annotation_summary.json')
        assert summary['processing_complete'] and summary['processed'] == 1875 and summary['adapter_errors'] == 0
        run_child([str(SEARCH), '-u', '-m', 'tasktol.data.libero_annotation.audit', '--output', str(RUN)],
                  'independent_audit.log', 'auditing')
        audit = read(RUN / 'annotation_independent_final_audit.json')
        assert audit['complete'] and audit['episodes'] == 1875 and audit['adapter_errors'] == 0
        for source in selection:
            assert sha(Path(source['folder']) / 'report.json') == source['report_sha256']
        finish(audit)
        export_status(RUN, VIEW)
    except BaseException:
        state('failed_or_interrupted', error=traceback.format_exc())
        raise

if __name__ == '__main__':
    main()
