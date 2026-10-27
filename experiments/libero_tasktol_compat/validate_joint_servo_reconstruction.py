"""Audit reconstructed reference trials and export selected state/action data."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(Path(__file__).parent),str(ROOT.parent/'franka_mujoco')]


def main(out):
    import h5py
    import numpy as np
    from tasktol.control.ipopt import IpoptSettings
    from tasktol.control.joint_tracker import JointReferenceTracker
    from tasktol.simulation.preset_runtime import PresetSimulationBackend
    from tasktol.control.task_space.objective import ObjectiveSettings
    from experiments.libero_tasktol_compat.reconstruct_joint_servo import freeze,accepts,score
    from experiments.libero_tasktol_compat.osc_reference import integrate_reference_actions
    plan=json.loads((out/'plan.json').read_text());summary=json.loads((out/'summary.json').read_text())
    assert summary['complete'] and freeze()==plan['source_sha256']
    selections=sorted(summary['results'],key=lambda x:x['seed'])
    audit=[];all_solver=[];selected_rows=[];baseline_rows=[];b10_rows=[];per_seed=[]
    for selection in selections:
        seed=selection['seed'];job=plan['jobs'][seed];task=plan['task']
        initial_ref=Path(job['selected_root'])/task['suite']/f"{task['task_id']:02d}"/job['demo']/'state_reference.hdf5'
        old=Path(plan['baseline'])/'joint_native'/f'seed_{seed:02d}/off'
        old_report=json.loads((old/'report.json').read_text());expected=selection['baseline']
        with h5py.File(job['source_hdf5']) as f:source_grip=f['data'][job['demo']]['actions'][:-1,6]
        for trial in selection['trials']:
            folder=Path(trial['folder']);report=json.loads((folder/'report.json').read_text())
            rows=json.loads((folder/'trajectory.json').read_text());diagnostic=json.loads((folder/'solver_observations.json').read_text())
            assert report['solver_settings']==asdict(IpoptSettings())
            assert report['objective_settings']==asdict(ObjectiveSettings.delta_q_squared_conditioning_safeguard())
            assert report['initial_state_sha256']==old_report['initial_state_sha256']
            assert report['model_sha256']==old_report['model_sha256']
            assert report['shared_joint_servo']==old_report['shared_joint_servo']
            assert all(r['bounds_rad'] is None and r['effective_mask']=='000' and r['physics_steps']==(0 if r['aborted'] else 25) for r in rows)
            assert report['shared_joint_servo']['nullspace_control'] is False
            with h5py.File(folder/'state_reference.hdf5') as f:
                actions=f['actions'][()];p=f['positions'][()];r=f['rotations'][()]
                assert f.attrs['projection_posture_weight']==0.
                rebuilt_p,rebuilt_r=integrate_reference_actions(actions,p[0],r[0])
                np.testing.assert_allclose(rebuilt_p,p[1:],rtol=0,atol=1e-10)
                np.testing.assert_allclose(rebuilt_r,r[1:],rtol=0,atol=1e-10)
                np.testing.assert_array_equal(actions[:,6],source_grip)
                assert len(actions)==report['planned_frames']
            probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
                '-show_entries','stream=nb_read_frames,r_frame_rate','-of','json',str(folder/'main_rgb.mp4')]))['streams'][0]
            assert int(probe['nb_read_frames'])==len(rows) and probe['r_frame_rate']=='20/1'
            with np.load(folder/'servo_physics.npz') as f:samples=f['samples']
            assert len(samples)==25*report['executed_frames']
            tracker=JointReferenceTracker(7);previous=np.asarray(rows[0]['pre_action']['joint_position_rad']);j=0
            ranges=np.asarray(report['shared_joint_servo']['position_servo']['actuator_ctrlrange']);span=np.diff(ranges,axis=1)[:,0]
            for row in rows:
                np.testing.assert_allclose(row['action'],actions[row['index']],atol=0,rtol=0)
                if row['aborted']:continue
                assert row['optimizer']['feasible'] and row['native_joint_goal_error_rad']<1e-10
                elapsed=0.;start=previous.copy()
                target=np.asarray(row['pre_action']['joint_position_rad'])+np.asarray(row['native_joint_action'][:7])*span
                for _ in range(25):
                    sample=samples[j];j+=1
                    ref,command,_,elapsed,alpha=PresetSimulationBackend.track_joint_reference(
                        tracker=tracker,interpolation_start=start,joint_target=target,interpolation_elapsed=elapsed,
                        interpolation_duration_s=.05,physics_dt=.002,actual_q=sample[16:23],actual_velocity_q=sample[23:30])
                    np.testing.assert_array_equal(ref,sample[2:9]);np.testing.assert_array_equal(command,sample[9:16]);assert alpha==sample[1]
                    previous=ref.copy()
            metrics=score(folder,job,task,folder/'state_reference.hdf5')
            assert metrics==trial['metrics'] and accepts(expected,metrics)==trial['accepted']
            if trial['accepted']:expected=metrics
            all_solver.extend(dict(seed=seed,iteration=trial['iteration'],**d)for d in diagnostic)
            audit.append(dict(seed=seed,iteration=trial['iteration'],success=report['success'],rejected=report['execution_rejected'],
                media_decoded_frames=len(rows),shared_servo_bitwise_verified_steps=j))
        assert expected==selection['selected']['metrics']
        selected=Path(selection['selected']['folder']);report=json.loads((selected/'report.json').read_text())
        chosen=json.loads((selected/'trajectory.json').read_text());selected_rows.extend(r for r in chosen if not r['aborted'])
        before=json.loads((old/'trajectory.json').read_text())
        original_b10=json.loads((Path(plan['baseline'])/'b10'/f'seed_{seed:02d}/off/trajectory.json').read_text())
        # Identical complete cohort for controller/reference comparison. Keep
        # the newly recovered seed 7 separately in selected_all10.
        if not old_report['execution_rejected']:
            baseline_rows.extend(before);b10_rows.extend(original_b10)
        per_seed.append(dict(seed=seed,demo=job['demo'],baseline_success=old_report['success'],success=report['success'],
            selected_folder=str(selected),baseline_source_position_mm=old_report['tracking']['source_position_mm'],
            source_position_mm=report['tracking']['source_position_mm'],source_rotation_deg=report['tracking']['source_rotation_deg'],
            reference_position_mm=report['tracking']['nominal_position_mm'],reference_rotation_deg=report['tracking']['nominal_rotation_deg'],
            baseline_cost=selection['baseline']['objective'],cost=selection['selected']['metrics']['objective']))
    def stats(values):
        a=np.asarray(values,dtype=float);return dict(mean=float(a.mean()),p95=float(np.percentile(a,95)),max=float(a.max()),count=len(a))
    def metrics(rows):return dict(source_position_mm=stats([r['source_next_position_error_mm']for r in rows]),
        source_rotation_deg=stats([r['source_next_rotation_error_deg']for r in rows]),
        reference_position_mm=stats([1000*r['tracking_position_error_m']for r in rows]),
        reference_rotation_deg=stats([r['nominal_rotation_error_deg']for r in rows]))
    selected_common=[r for s in selections if s['seed']!=7 for r in json.loads((Path(s['selected']['folder'])/'trajectory.json').read_text()) if not r['aborted']]
    # State/action-only artifact. Image data is not fabricated from lossy MP4s.
    with h5py.File(out/'selected_state_actions.hdf5','w') as f:
        f.attrs.update(schema='tasktol_joint_servo_state_actions_v1',control_hz=20,physics_hz=500,
            action_units='xyz metres, world rotvec radians, original gripper',observation_timing='pre_action',
            camera_images_included=False,replay_plan=str(out/'plan.json'),
            note='Use exact validated native sampler and shared position-servo override; not stock OSC replay.')
        data=f.create_group('data');attempts=f.create_group('attempts')
        for s in selections:
            folder=Path(s['selected']['folder']);report=json.loads((folder/'report.json').read_text())
            rows=json.loads((folder/'trajectory.json').read_text());n=report['executed_frames']
            g=attempts.create_group(s['demo'])
            with h5py.File(s['selected']['reference_path'])as ref:
                g.create_dataset('actions',data=ref['actions'][:n]);g.create_dataset('reference_positions_m',data=ref['positions'][()])
                g.create_dataset('reference_rotations',data=ref['rotations'][()])
            with np.load(folder/'states.npz')as states:
                actual=np.concatenate([states['pre_action_states'][:n],states['final_state'][None]])
                g.create_dataset('states',data=actual[:-1]);g.create_dataset('next_states',data=actual[1:])
                np.testing.assert_allclose(np.diff(actual[:,0]),.05,rtol=0,atol=1e-10)
            g.create_dataset('joint_targets_rad',data=[r['joint_target_rad']for r in rows if not r['aborted']])
            g.attrs.update(success=report['success'],executed_frames=n,planned_frames=report['planned_frames'],
                recording_folder=str(folder),video=str(folder/'main_rgb.mp4'),report_json=json.dumps(report))
            if report['success']:
                assert n==report['planned_frames'] and not report['execution_rejected'];data[s['demo']]=g
    result=dict(complete=True,source_unchanged=True,default_optimizer_unchanged=True,
        initial_model_controller_identical_to_baseline=True,original_gripper_and_20hz_verified=True,
        robot_joint_tracking_penalty=False,finger_joint_tracking_penalty=False,audit=audit,per_seed=per_seed,
        counts=dict(baseline_success=sum(r['baseline_success']for r in per_seed),success=sum(r['success']for r in per_seed),total=len(per_seed)),
        comparison_common9=dict(b10=metrics(b10_rows),joint_before=metrics(baseline_rows),joint_reconstructed=metrics(selected_common)),
        selected_all10=metrics(selected_rows),
        solver=dict(attempts=len(all_solver),wall_ms=stats([d['problem_solve_wall_ms']for d in all_solver if d['solver_called']]),
            ik_wall_ms=stats([d['ik_wall_ms']for d in all_solver]),ik_cpu_ms=stats([d['ik_cpu_ms']for d in all_solver]),
            cpu_timeouts=[d for d in all_solver if d.get('status_code')==-4],
            wall_over_50ms=[d for d in all_solver if d.get('problem_solve_wall_ms',0)>50],
            ik_wall_over_50ms=[d for d in all_solver if d['ik_wall_ms']>50]),
        dataset='selected_state_actions.hdf5',camera_images_included=False)
    (out/'validation.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:result[k]for k in ['counts','comparison_common9','selected_all10','solver']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();main(a.run.resolve())
