"""Verify success-only dual-camera data and build 40 task HDF5 indexes."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(Path(__file__).parent),str(ROOT.parent/'franka_mujoco')]


def verify_episode(out,record,task):
    import h5py
    import numpy as np
    from scipy.spatial.transform import Rotation
    from tasktol.control.ipopt import IpoptSettings
    from tasktol.control.task_space.objective import ObjectiveSettings
    from tasktol.control.joint_tracker import JointReferenceTracker
    from tasktol.simulation.preset_runtime import PresetSimulationBackend
    from experiments.libero_tasktol_compat.osc_reference import integrate_reference_actions
    folder=Path(record['folder']);report=json.loads((folder/'report.json').read_text())
    rows=json.loads((folder/'trajectory.json').read_text());n=len(rows)
    assert record['success'] and report['success'] and not report['execution_rejected']
    assert n==report['executed_frames']==report['planned_frames']
    assert report['solver_settings']==asdict(IpoptSettings())
    assert report['objective_settings']==asdict(ObjectiveSettings.delta_q_squared_conditioning_safeguard())
    assert report['shared_joint_servo']['nullspace_control'] is False
    assert report['shared_joint_servo']['effective_integrator']==0
    assert report['shared_joint_servo']['interpolation_duration_s']==.05
    assert report['shared_joint_servo']['physics_dt_s']==.002
    with h5py.File(folder/'state_reference.hdf5')as f:
        actions=f['actions'][()];anchor_p=f['positions'][0];anchor_r=f['rotations'][0]
        p,r=integrate_reference_actions(actions,anchor_p,anchor_r)
        np.testing.assert_allclose(p,f['positions'][1:],atol=1e-10,rtol=0)
        np.testing.assert_allclose(r,f['rotations'][1:],atol=1e-10,rtol=0)
        source_p=f['source_positions_m'][()];source_r=f['source_rotations'][()]
    with h5py.File(task['source_hdf5'])as f:
        source=f['data'][record['demo']];source_states=source['states'][()]
        source_actions=source['actions'][()]
        source_hash=hashlib.sha256(source_states.tobytes()+source_actions.tobytes()+str(source.attrs['model_file']).encode()).hexdigest()
    assert source_hash==record['source_input_sha256'] and n==len(source_states)-1
    np.testing.assert_array_equal(actions[:,6],source_actions[:-1,6])
    with h5py.File(folder/'recording.hdf5')as f,np.load(folder/'states.npz')as states:
        assert list(f['data'])==[record['demo']];g=f['data'][record['demo']]
        assert g.attrs['success'] and g.attrs['num_samples']==n
        assert f.attrs['control_hz']==20 and f.attrs['physics_hz']==500
        np.testing.assert_array_equal(g['actions'][()],actions)
        np.testing.assert_array_equal(g['states'][()],states['pre_action_states'])
        actual=np.concatenate([states['pre_action_states'],states['final_state'][None]])
        np.testing.assert_array_equal(g['next_states'][()],actual[1:])
        np.testing.assert_array_equal(actual[0],source_states[0])
        np.testing.assert_allclose(np.diff(actual[:,0]),.05,rtol=0,atol=1e-10)
        assert hashlib.sha256(actual[0].tobytes()).hexdigest()==report['initial_state_sha256']
        for name in ['agentview_rgb','eye_in_hand_rgb']:
            d=g['obs'][name];assert d.shape==(n,256,256,3) and d.dtype==np.uint8
            # Read every chunk to verify compression/data integrity.
            for i in range(n):assert np.asarray(d[i]).shape==(256,256,3)
        for name,size in [('ee_states',6),('ee_pos',3),('ee_ori',3),('joint_states',7),('gripper_states',2)]:
            assert g['obs'][name].shape==(n,size) and np.all(np.isfinite(g['obs'][name][()]))
        assert g['dones'][-1]==1 and g['rewards'][-1]==1
        # Shared frames use the logical EE rotation offset; native observation
        # state remains native and is not replaced with source-state values.
        for i,row in enumerate(rows):
            assert not row['aborted'] and row['physics_steps']==25 and row['optimizer']['feasible']
            assert row['effective_mask']=='000' and row['bounds_rad'] is None
            np.testing.assert_array_equal(row['action'],actions[i])
            err=float(np.linalg.norm(np.asarray(row['post_action']['position_m'])-source_p[i+1])*1000)
            assert abs(err-row['source_next_position_error_mm'])<1e-9
            rot=Rotation.from_rotvec(row['post_action']['rotation_vector_rad']).as_matrix()
            er=np.linalg.norm(Rotation.from_matrix(rot@source_r[i+1].T).as_rotvec())*180/np.pi
            assert abs(er-row['source_next_rotation_error_deg'])<1e-9
    with np.load(folder/'servo_physics.npz')as f:samples=f['samples']
    assert len(samples)==n*25
    tracker=JointReferenceTracker(7);previous=np.asarray(rows[0]['pre_action']['joint_position_rad']);j=0
    ranges=np.asarray(report['shared_joint_servo']['position_servo']['actuator_ctrlrange']);span=np.diff(ranges,axis=1)[:,0]
    for row in rows:
        start=previous.copy();elapsed=0.
        goal=np.asarray(row['pre_action']['joint_position_rad'])+np.asarray(row['native_joint_action'][:7])*span
        np.testing.assert_allclose(goal,row['joint_target_rad'],atol=1e-12,rtol=0)
        for _ in range(25):
            sample=samples[j];j+=1
            q,command,_,elapsed,alpha=PresetSimulationBackend.track_joint_reference(tracker=tracker,
                interpolation_start=start,joint_target=goal,interpolation_elapsed=elapsed,
                interpolation_duration_s=.05,physics_dt=.002,actual_q=sample[16:23],actual_velocity_q=sample[23:30])
            np.testing.assert_array_equal(q,sample[2:9]);np.testing.assert_array_equal(command,sample[9:16])
            assert alpha==sample[1];previous=q.copy()
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
        '-show_entries','stream=nb_read_frames,r_frame_rate','-of','json',str(folder/'main_rgb.mp4')]))['streams'][0]
    assert int(probe['nb_read_frames'])==n and probe['r_frame_rate']=='20/1'
    return dict(case=record['case'],demo=record['demo'],frames=n,physics_steps=j,full_decode=True,
        cameras=['agentview_rgb','eye_in_hand_rgb'],servo_bitwise_verified=True,
        position_errors_mm=[r['source_next_position_error_mm']for r in rows],
        rotation_errors_deg=[r['source_next_rotation_error_deg']for r in rows],report=report)


def verify_preflight(out,record):
    import h5py
    import numpy as np
    from experiments.libero_tasktol_compat import reconstruct_joint_servo_dataset as dataset
    from experiments.libero_tasktol_compat import rollout_mujoco_joint_servo as servo
    from experiments.libero_tasktol_compat import rollout_mask_tolerance as runner
    plan=json.loads((out/'plan.json').read_text());case=record['case'];entry=plan['tasks'][case]
    job=next(j for j in entry['candidates']if j['demo']==record['demo']);folder=Path(record['folder'])
    dest=out/'.preflight_replay'/f"case_{case:02d}_{record['demo']}"
    ref=dataset.reconstruction.load_reference(folder/'state_reference.hdf5')
    runner.dump(dest/'joint_native/plan.json',dict(case=case,task=entry['task'],jobs=[job],source_sha256=plan['source_sha256']))
    try:
        with h5py.File(folder/'recording.hdf5')as file:
            group=file['data'][record['demo']]
            def compare(index,action,obs,env):
                np.testing.assert_array_equal(obs['agentview_image'],group['obs/agentview_rgb'][index])
                np.testing.assert_array_equal(obs['robot0_eye_in_hand_image'],group['obs/eye_in_hand_rgb'][index])
                np.testing.assert_array_equal(env.sim.get_state().flatten(),group['states'][index])
            report=servo.worker(dest,0,'joint_native',reference_transform=lambda original,session:ref,observation_callback=compare)
        assert report['success'];cold=dest/'joint_native/seed_00/off'
        for name in ['states.npz','servo_physics.npz']:
            with np.load(cold/name)as a,np.load(folder/name)as b:
                for k in a.files:np.testing.assert_array_equal(a[k],b[k])
        record['independent_replay_bitwise_identical']=True
        record['preflight_replay_all_native_images_bitwise_identical']=True
        runner.dump(out/'records'/f'case_{case:02d}'/(record['demo']+'.json'),record)
    finally:
        if dest.exists():shutil.rmtree(dest)


def main(out,partial=False,replay_preflight=False):
    import h5py
    import numpy as np
    from experiments.libero_tasktol_compat import reconstruct_joint_servo_dataset as dataset
    from experiments.libero_tasktol_compat import rollout_mask_tolerance as runner
    plan=json.loads((out/'plan.json').read_text());assert dataset.freeze()==plan['source_sha256']
    selection=json.loads(dataset.SELECTION.read_text())['tasks'];results=[];task_indexes=[];failures=[]
    for entry in plan['tasks']:
        case=entry['case'];records=[json.loads(p.read_text())for p in sorted((out/'records'/f'case_{case:02d}').glob('*.json'))]
        successes=[r for r in records if r.get('success')]
        if partial and len(successes)<10:continue
        assert len(successes)==10,f'case{case}: {len(successes)} successes'
        index=out/'datasets'/entry['task']['suite']/(entry['task']['task_name']+'_demo.hdf5');index.parent.mkdir(parents=True,exist_ok=True)
        total=0
        with h5py.File(index,'w')as file:
            data=file.create_group('data');file.attrs.update(language=entry['task']['language'],source_plan=str(out/'plan.json'),
                control_hz=20,physics_hz=500,action_units='metres and world rotvec radians plus original gripper',
                image_orientation='native LIBERO; rotate once at training export')
            for record in successes:
                if not record.get('independent_replay_bitwise_identical'):
                    assert replay_preflight,'Missing cold replay evidence'
                    verify_preflight(out,record)
                audit=verify_episode(out,record,selection[case]);results.append(audit);total+=audit['frames']
                relative=os.path.relpath(Path(record['folder'])/'recording.hdf5',index.parent)
                data[record['demo']]=h5py.ExternalLink(relative,'/data/'+record['demo'])
            data.attrs['total']=total;file.attrs['successful_episodes']=10
        task_indexes.append(dict(case=case,suite=entry['task']['suite'],task_id=entry['task']['task_id'],path=str(index),episodes=10,frames=total))
        print(json.dumps(dict(verified_task=case,verified_episodes=len(results))),flush=True)
        for record in records:
            for trial in record.get('trials',[]):
                if not trial.get('skipped') and not trial['success']:
                    failures.append(dict(case=case,demo=record['demo'],**trial))
        runner.dump(out/'validation_progress.json',dict(tasks_verified=len(task_indexes),episodes_verified=len(results)))
    if partial:return
    assert len(results)==400 and len(task_indexes)==40
    assert not list((out/'.scratch').glob('*')),'Transient trials remain'
    all_episode_folders=list((out/'episodes').glob('*/*/*/report.json'));assert len(all_episode_folders)==400
    for p in all_episode_folders:assert json.loads(p.read_text())['success']
    def stats(values):
        a=np.asarray(values,dtype=float);return dict(mean=float(a.mean()),p95=float(np.percentile(a,95)),max=float(a.max()),count=len(a))
    position=[x for r in results for x in r.pop('position_errors_mm')];rotation=[x for r in results for x in r.pop('rotation_errors_deg')]
    validation=dict(complete=True,tasks=40,episodes=400,task_indexes=task_indexes,episode_checks=results,
        source_position_mm=stats(position),source_rotation_deg=stats(rotation),
        original_gripper_verified=True,original_speed_verified=True,default_ipopt_verified=True,
        all_selected_independently_replayed=True,all_servo_samples_independently_verified=True,
        success_only_storage=True,failed_trial_records=len(failures),
        robot_joint_tracking_penalty=False,finger_joint_tracking_penalty=False,source_unchanged=dataset.freeze()==plan['source_sha256'])
    runner.dump(out/'validation.json',validation);runner.dump(out/'failure_records.json',failures)
    runner.dump(out/'manifest.json',dict(complete=True,episodes=400,tasks=task_indexes,validation='validation.json',
        policy=plan['policy'],schema='native_LIBERO_HDF5_pre_action_dual_camera_physical_reference_actions'))
    print(json.dumps({k:validation[k]for k in ['complete','tasks','episodes','source_position_mm','source_rotation_deg','failed_trial_records']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--partial',action='store_true')
    p.add_argument('--replay-preflight',action='store_true');a=p.parse_args();main(a.run.resolve(),a.partial,a.replay_preflight)
