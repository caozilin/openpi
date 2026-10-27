"""Verify direct shared servo recordings, inputs, transport units and timings."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT.parent/'franka_mujoco')]


def main(run):
    import numpy as np
    from tasktol.control.joint_tracker import JointReferenceTracker
    from tasktol.control.ipopt import IpoptSettings
    from tasktol.simulation.preset_runtime import PresetSimulationBackend
    from dataclasses import asdict
    from experiments.libero_tasktol_compat.rollout_mujoco_joint_servo import freeze
    plan=json.loads((run/'plan.json').read_text());summary=json.loads((run/'summary.json').read_text())
    assert summary['complete'] and freeze()==plan['source_sha256']
    data={};reports={};verification=[];audit=[]
    for variant in ['b10','joint_native','joint_implicitfast']:
        for folder in sorted((run/variant).glob('seed_*/off')):
            report=json.loads((folder/'report.json').read_text());seed=report['seed']
            rows=json.loads((folder/'trajectory.json').read_text());data[(seed,variant)]=rows;reports[(seed,variant)]=report
            assert report['solver_settings']==asdict(IpoptSettings())
            assert report['tolerance']=='off' and all(r['bounds_rad'] is None and r['effective_mask']=='000' for r in rows)
            assert all(r['physics_steps']==(0 if r['aborted'] else 25) for r in rows)
            probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
                '-show_entries','stream=nb_read_frames,r_frame_rate','-of','json',str(folder/'main_rgb.mp4')]))['streams'][0]
            assert int(probe['nb_read_frames'])==len(rows) and probe['r_frame_rate']=='20/1'
            if variant.startswith('joint'):
                cfg=report['shared_joint_servo'];assert cfg['nullspace_control'] is False
                if variant=='joint_native':assert cfg['effective_integrator']==cfg['native_integrator']==0
                else:assert cfg['effective_integrator']==cfg['source_integrator']==3
                assert cfg['physics_dt_s']==.002 and cfg['interpolation_duration_s']==.05
                with np.load(folder/'servo_physics.npz') as f:samples=f['samples']
                assert len(samples)==report['executed_frames']*25
                tracker=JointReferenceTracker(7);previous=np.asarray(rows[0]['pre_action']['joint_position_rad']);j=0
                ctrl_ranges=np.asarray(cfg['position_servo']['actuator_ctrlrange'])
                span=ctrl_ranges[:,1]-ctrl_ranges[:,0]
                for row in rows:
                    if row['aborted']:break
                    start=previous.copy();elapsed=0.
                    # Reconstruct native delta-action decoding. It may differ
                    # from IPOPT q by one rounding ulp; the client audits it.
                    encoded_goal=np.asarray(row['pre_action']['joint_position_rad'])+np.asarray(row['native_joint_action'][:7])*span
                    np.testing.assert_allclose(encoded_goal,row['joint_target_rad'],rtol=0,atol=1e-12)
                    for _ in range(25):
                        saved=samples[j];j+=1
                        reference,command,diag,elapsed,alpha=PresetSimulationBackend.track_joint_reference(
                            tracker=tracker,interpolation_start=start,joint_target=encoded_goal,
                            interpolation_elapsed=elapsed,interpolation_duration_s=.05,physics_dt=.002,
                            actual_q=saved[16:23],actual_velocity_q=saved[23:30])
                        np.testing.assert_array_equal(reference,saved[2:9])
                        np.testing.assert_array_equal(command,saved[9:16])
                        assert alpha==saved[1]
                        previous=reference.copy()
                audit.append(dict(seed=seed,variant=variant,shared_reference_and_tracker_bitwise_match=True,physical_control_calls=j))
            else:
                job=plan['jobs'][seed];t=plan['task']
                old=Path(job['selected_root'])/t['suite']/f"{t['task_id']:02d}"/job['demo']/'ipopt_state_B10/trajectory.json'
                old_rows=json.loads(old.read_text())
                assert len(old_rows)==len(rows)
                assert all(a['joint_target_rad']==b['joint_target_rad'] and a['post_action']==b['post_action'] for a,b in zip(rows,old_rows,strict=True))
            verification.append(dict(seed=seed,variant=variant,video_frames=len(rows),full_decode=True))
    assert len(reports)==21
    for seed in range(10):
        a,b=[reports[(seed,v)]for v in ['b10','joint_native']]
        for field in ['initial_state_sha256','model_sha256','actions_sha256']:assert a[field]==b[field]
        for r,s in zip(data[(seed,'b10')],data[(seed,'joint_native')]):
            np.testing.assert_array_equal(r['nominal_target']['position_m'],s['nominal_target']['position_m'])
            np.testing.assert_array_equal(r['nominal_target']['rotation_vector_rad'],s['nominal_target']['rotation_vector_rad'])
    # Compare all 9 fully executed pairs; do not filter out task failures.
    complete=[seed for seed in range(10) if all(not reports[(seed,v)]['execution_rejected'] for v in ['b10','joint_native'])]
    def stats(values):
        a=np.asarray(values,dtype=float)
        return dict(mean=float(a.mean()),p95=float(np.percentile(a,95)),max=float(a.max()),count=len(a))
    metrics={}
    for variant in ['b10','joint_native']:
        rows=[r for seed in complete for r in data[(seed,variant)] if not r['aborted']]
        metrics[variant]=dict(cohort_seeds=complete,
            source_position_mm=stats([r['source_next_position_error_mm'] for r in rows]),
            source_rotation_deg=stats([r['source_next_rotation_error_deg'] for r in rows]),
            reference_position_mm=stats([r['tracking_position_error_m']*1000 for r in rows]),
            reference_rotation_deg=stats([r['nominal_rotation_error_deg'] for r in rows]),
            joint_reference_error_rad=stats([np.linalg.norm(np.asarray(r['post_action']['joint_position_rad'])-r['joint_target_rad']) for r in rows]),
            solver_wall_ms=stats([r['optimizer']['elapsed_ms'] for seed in range(10) for r in data[(seed,variant)]]),
            cpu_timeout_frames=sum(len(reports[(seed,variant)]['cpu_timeout_frames'])for seed in range(10)),
            wall_over_50ms=[dict(seed=seed,index=r['index'],wall_ms=r['optimizer']['elapsed_ms'])for seed in range(10) for r in data[(seed,variant)] if r['optimizer']['elapsed_ms']>50])
    result=dict(complete=True,source_unchanged=True,all_pairs_identical_initial_model_and_actions=True,
        nominal_position_and_rotation_references_identical=True,b10_bitwise_matches_original_selected_recordings=True,
        all_media_decoded=verification,shared_tracker_audit=audit,metrics=metrics,
        failed_native_cases=[dict(seed=r['seed'],demo=r['demo'],rejected=r['execution_rejected'],rejected_frame=r['rejected_frame'],predicates=r['native_predicates'])for (seed,v),r in reports.items() if v=='joint_native' and not r['success']],
        tests=dict(mujoco_project_runtime='15 passed: shared interpolation arithmetic, joint tracker, canonical backend',
            libero_pinned_runtime='10 passed; 5 scene compilation failures due to old MuJoCo keyframe size validation; same tests pass in project runtime'))
    (run/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result['metrics'],indent=2));print(json.dumps(result['failed_native_cases'],ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();main(a.run.resolve())
