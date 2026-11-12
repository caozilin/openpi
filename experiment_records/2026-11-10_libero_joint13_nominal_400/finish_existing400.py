"""Audit existing400 only, then finish original report/pairing/owned cleanup."""
from pathlib import Path
import sys,subprocess,fcntl,os,json,traceback
REC=Path(__file__).resolve().parent
sys.path.insert(0,str(REC))
import pipeline_v2 as p
r=p.RUN
lock=(r/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
p.dump(r/'pipeline_process.json',dict(pid=os.getpid(),command=sys.argv,mode='audit_and_finish_only'))
try:
 p.validate(p.read(r/'evaluation_request.json'))
 amendment=p.read(REC/'residual_norm_audit_amendment.json')
 for name,h in amendment['new_sources'].items():assert p.sha(Path(name))==h
 for name,h in p.read(Path(amendment['preserved_all400_artifacts'])).items():assert p.sha(Path(name))==h
 p.status('auditing',audit_version=3)
 with (r/'independent_audit_v3.log').open('a') as log:
  child=subprocess.Popen([str(p.PYTHON),'-u',str(REC/'audit_evaluation_v3.py'),'--run-dir',str(r)],cwd=p.ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  p.dump(r/'audit_v3_launch.json',dict(pid=child.pid,command=str(REC/'audit_evaluation_v3.py')))
  assert child.wait()==0,'Full read-only v3 audit failed; preserve all results'
 for name,h in p.read(Path(amendment['preserved_all400_artifacts'])).items():assert p.sha(Path(name))==h
 result=p.finish_report();p.write_paired_comparison();p.stop_inference()
 result.update(historical_audit_implementation_interruptions=2,residual_norm_audit_amendment=str(REC/'residual_norm_audit_amendment.json'),numeric_comparison_audit_amendment=str(REC/'numeric_comparison_audit_amendment.json'))
 p.dump(r/'task_completion.json',dict(result,all_required_audits_complete=True,own_inference_and_tunnels_stopped=True,finished_at=p.now()))
 p.status('complete',**result)
 with (REC/'README.md').open('a') as f:f.write('\n\n最终400审计通过，368/400成功，逐seed配对报告已生成，六推理副本及专用隧道定向退出；历史2项审计实现错误保留，未重跑已提交成败。详见RESULTS.md及两份audit_amendment。\n')
except BaseException:
 p.status('failed_or_interrupted',error=traceback.format_exc());raise
