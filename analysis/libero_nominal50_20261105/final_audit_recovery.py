"""Finish a completed rollout set after the follow-audit exit-check race."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,shutil,subprocess,sys,traceback
ROOT=Path('/media/czl/sata/franka_my_code/openpi')
sys.path.insert(0,str(ROOT))
from examples.libero.eval_nominal50 import RUN,PYTHON,read,dump,validate
from examples.libero.nominal50_protocol import finalize

def main():
    lock=(RUN/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    validate(read(RUN/'evaluation_request.json'))
    if (RUN/'task_completion.json').exists():
        assert read(RUN/'task_completion.json')['complete'];return
    for name in ['evaluation_process.json','independent_audit_process.json']+[f'client_{i}_process.json' for i in range(6)]:
        assert not Path(f"/proc/{read(RUN/name)['pid']}").exists(),name
    for name in ['evaluation.lock','independent_audit.lock']:
        with (RUN/name).open('r') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    completion=read(RUN/'evaluation_completion.json')
    assert completion['complete'] and completion['completed_episodes']==1600
    assert completion['exit_codes']==[0]*6 and completion['infrastructure_errors']==0
    assert read(RUN/'inference_shutdown_verification.json')['all_exited']
    assert read(RUN/'local_cleanup_verification.json')['complete']
    attempt=RUN/'attempts/0003_follow_audit_process_exit_race'
    attempt.mkdir(exist_ok=False)
    for name in ['pipeline_state.json','independent_audit_state.json','independent_audit_process.json','independent_audit.log','pipeline.log']:
        shutil.copy2(RUN/name,attempt/name)
    dump(attempt/'diagnosis.json',dict(complete=True,reason='Audit read completion marker before evaluation manager process fully exited',
        original_audit_reached_1600=(RUN/'independent_audit.log').read_text().count('Audited ')>=1600,
        rollout_results_modified=False,source_code_modified=False,all_original_processes_now_exited=True,
        correction='Run same pinned read-only auditor without --follow after manager exit; no rollout repetition'))
    dump(RUN/'pipeline_process.json',dict(pid=os.getpid(),command=sys.argv,recovery='post_exit_readonly_audit'))
    try:
        with (RUN/'independent_audit.log').open('a') as log:
            child=subprocess.Popen([str(PYTHON),'-u','-m','examples.libero.audit_formal_nominal','--run-dir',str(RUN)],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        dump(RUN/'independent_audit_launch.json',dict(pid=child.pid,follow=False,recovery='post_exit_readonly_audit'))
        dump(RUN/'pipeline_state.json',dict(stage='auditing_completed_rollouts',audit=child.pid,rollouts_complete=1600,rollouts_restarted=False))
        assert child.wait()==0,'Independent audit failed; retain all evidence'
        assert read(RUN/'evaluation_independent_final_audit.json')['complete']
        for name in ['resume_after_transport_failure.json','resume_after_host_restart.json']:
            receipt=read(RUN/name)
            for path,digest in receipt['committed_artifact_sha256'].items():
                assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
        result=finalize(RUN)
        history=(RUN/'RESUME_HISTORY.md').read_text()
        with (RUN/'NOMINAL_50SEED_REPORT.md').open('a') as report:
            report.write('\n\n## 基础设施中断与审计恢复\n\n'+history+'\n')
            report.write('1600条新增回合已正常提交，六客户端exit0。历史3次通信超时未计为任务失败，全部已提交成功及失败均保持原文件SHA。\n')
            report.write('首次跟随审计在内容核验完成后的进程退出检查遇到竞态；确认主控及客户端真实退出后，使用同一冻结源码重新完整只读核验1600条，未重跑任何计分回合。\n')
            report.write('服务器本次六副本及六转发已定向停止，用户可视化服务未干预。\n')
        dump(RUN/'task_completion.json',dict(result,all_required_audits_complete=True,own_inference_and_tunnels_stopped=True,
            viewer_processes_untouched=True,historical_transport_errors=3,interrupted_attempts_archived=True,
            follow_audit_exit_race_recovered=True,completed_at=datetime.datetime.now().astimezone().isoformat()))
        dump(RUN/'pipeline_state.json',dict(stage='complete',**result))
        print(result,flush=True)
    except BaseException:
        dump(RUN/'pipeline_state.json',dict(stage='final_audit_recovery_failed',error=traceback.format_exc()))
        raise
if __name__=='__main__':main()
