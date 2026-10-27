"""Batch orchestration only; each process uses the canonical LIBERO IPOPT client."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--tolerance-deg',type=float)
    parser.add_argument('--task-ids',type=int,nargs='+',default=list(range(10)))
    parser.add_argument('--name',default='nominal40')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    project=root.parents[1]
    env=dict(os.environ,PYTHONPATH=str(project),MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    jobs=[(s,i) for s in ('libero_spatial','libero_object','libero_goal','libero_10') for i in args.task_ids]
    batch=root/'results'/args.name;batch.mkdir(parents=True,exist_ok=True)
    def run(job):
        suite,tid=job;output=batch/suite/f'{tid:02d}';output.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-m','tasktol.benchmark.libero','--suite',suite,'--task-id',str(tid),'--output',str(output)]
        if args.tolerance_deg is not None:command+=['--tolerance-deg',str(args.tolerance_deg)]
        with (output/'run.log').open('w') as log:
            done=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        path=output/'report.json'
        report=json.loads(path.read_text()) if path.exists() else {}
        return dict(suite=suite,task_id=tid,returncode=done.returncode,report=str(path),
            status=report.get('status','missing'),success=report.get('task_success',False),frames=report.get('frames'),
            feasible_frames=report.get('feasible_frames'),error=report.get('error'))
    results=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run,job) for job in jobs]):
            result=future.result();results.append(result)
            (batch/'batch_summary.json').write_text(json.dumps(dict(complete=len(results)==len(jobs),expected=len(jobs),results=results),indent=2)+'\n')
            print(f'{len(results)}/{len(jobs)} '+json.dumps(result),flush=True)


if __name__=='__main__':main()
