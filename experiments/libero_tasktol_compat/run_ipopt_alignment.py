"""Parameter-controlled runs through the existing canonical IPOPT CLI."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/media/czl/sata/franka_my_code/openpi')
TASKTOL = ROOT.parent / 'franka_mujoco'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--kp', type=float, nargs='+', required=True)
    parser.add_argument('--cases', type=int, nargs='+', default=list(range(40)))
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--joint-control-mode', choices=['native', 'osc_inertia_matched', 'osc_task_joint_null'], default='native')
    parser.add_argument('--reference-mode', default='source_goals', choices=['source_goals', 'recorded_motion', 'relabelled_actions'])
    parser.add_argument('--reference-root', type=Path)
    parser.add_argument('--tolerance-deg',type=float)
    args = parser.parse_args()
    if args.reference_mode == 'relabelled_actions' and args.reference_root is None:
        parser.error('relabelled_actions requires --reference-root')
    args.output.mkdir(parents=True, exist_ok=False)
    selection = json.loads((ROOT / 'data/libero40_mask_annotation/20261019_mainview_v1/selection_manifest.json').read_text())['tasks']
    env = dict(os.environ, PYTHONPATH=str(TASKTOL), MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    provenance = dict(
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_sha256={str(p.relative_to(TASKTOL)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            TASKTOL/'tasktol/benchmark/libero.py', TASKTOL/'tasktol/simulation/libero.py',
            TASKTOL/'tasktol/simulation/libero_joint_controller.py',
            TASKTOL/'tasktol/control/ipopt/controller.py', TASKTOL/'tasktol/control/task_space/objective.py',
        ]},
    )
    if args.reference_root is not None:
        helper=TASKTOL/'tasktol/benchmark/libero_reference_actions.py'
        provenance['source_sha256'][str(helper.relative_to(TASKTOL))]=hashlib.sha256(helper.read_bytes()).hexdigest()
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    jobs = [(case,kp) for kp in args.kp for case in args.cases]
    def run(job):
        case,kp = job
        selected = selection[case]
        dest = args.output/f'kp_{kp:g}'/selected['suite']/f"{selected['task_id']:02d}"
        dest.mkdir(parents=True,exist_ok=False)
        command = [sys.executable,'-m','tasktol.benchmark.libero','--suite',selected['suite'],
            '--task-id',str(selected['task_id']),'--output',str(dest),'--kp',str(kp),
            '--reference-mode',args.reference_mode,'--joint-control-mode',args.joint_control_mode]
        if args.reference_mode == 'relabelled_actions':
            reference=args.reference_root/selected['suite']/f"{selected['task_id']:02d}"/'reference_actions.hdf5'
            command.extend(['--reference-actions-path',str(reference.resolve())])
        if args.tolerance_deg is not None:
            command.extend(['--tolerance-deg',str(args.tolerance_deg)])
        with (dest/'run.log').open('w') as log:
            done = subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        path=dest/'report.json'
        report=json.loads(path.read_text()) if path.exists() else {}
        return dict(case=case,suite=selected['suite'],task_id=selected['task_id'],kp=kp,
            success=report.get('task_success',False),status=report.get('status','missing'),
            frames=report.get('frames'), feasible_frames=report.get('feasible_frames'),
            error=report.get('error'),returncode=done.returncode,report=str(path))
    results=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run,job) for job in jobs]):
            result=future.result();results.append(result)
            (args.output/'summary.json').write_text(json.dumps(dict(complete=len(results)==len(jobs),expected=len(jobs),results=results),indent=2)+'\n')
            print(f'{len(results)}/{len(jobs)} {json.dumps(result)}',flush=True)


if __name__=='__main__':
    main()
