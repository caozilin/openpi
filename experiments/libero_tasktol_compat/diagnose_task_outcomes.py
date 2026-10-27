"""Evaluate native task predicates on retained endpoints, without physics steps.

Source endpoint is the last available recorded state (before the omitted final
source action), not a claim about the source rollout's eventual success.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT.parent/'franka_mujoco'),str(ROOT/'third_party/libero')]


def diagnose(run,partial=False):
    import h5py
    import numpy as np
    from examples.libero import regenerate_libero_dataset as recorder
    from experiments.libero_tasktol_compat.rerecord_state_reference import plain
    from libero.libero import get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv
    tasks=json.loads((ROOT/'data/libero40_mask_annotation/20261019_mainview_v1/selection_manifest.json').read_text())['tasks']
    summary=json.loads((run/'summary.json').read_text())
    if not partial:assert summary['complete']
    results=[]
    for job in summary['results']:
        if 'error'in job or job['selected']['success']:continue
        root=Path(job['selected_root']);task=tasks[job['case']];demo=job['demo']
        folder=root/task['suite']/f"{task['task_id']:02d}"/demo/'ipopt_state_B10'
        report=json.loads((folder/'report.json').read_text())
        trajectory=json.loads((folder/'trajectory.json').read_text())
        native_goal_at_stop=bool(trajectory[-1]['success'])
        with h5py.File(task['processed_hdf5']) as f:
            config=json.loads(f.attrs['configuration']);order=config['demo_ids'];number=order.index(demo)
            before=json.loads(f['regeneration/episodes'][order[number-1]].attrs['rng_state']) if number else None
            after=json.loads(f['regeneration/episodes'][demo].attrs['rng_state'])
        with h5py.File(task['source_hdf5'])as f:
            initial=f['data'][demo]['states'][0];source_final=f['data'][demo]['states'][-1]
            archived_xml=str(f['data'][demo].attrs['model_file'])
        with h5py.File(folder/'recording.hdf5')as f:actual_final=f['attempts'][demo]['tasktol/final_state'][()]
        env=ControlEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task['suite']/(task['task_name']+'.bddl')),
            controller='OSC_POSE',use_camera_obs=False,has_offscreen_renderer=False,control_freq=20)
        try:
            env.seed(config['seed']);env.env.object_property_initializers*=number+1
            recorder._restore_rng(before if before is not None else dict(numpy=recorder._rng_state()['numpy'],python=after['python']))
            env.reset();env.set_init_state(initial)
            assert plain(recorder._rng_state())==after
            model=env.sim.model._model
            digest=hashlib.sha256(b''.join(np.asarray(getattr(model,k)).tobytes()for k in ['body_pos','body_quat','body_mass','body_inertia','geom_friction','jnt_range'])).hexdigest()
            assert digest==report['model_sha256']
            def endpoint(state):
                env.set_init_state(state)
                predicates=[dict(predicate=p,satisfied=bool(env.env._eval_predicate(p)))for p in env.env.parsed_problem['goal_state']]
                success=bool(env.check_success());assert success==all(p['satisfied']for p in predicates)
                objects={}
                for pred in env.env.parsed_problem['goal_state']:
                    for name in pred[1:]:
                        item=env.env.object_states_dict[name]
                        if hasattr(item,'get_geom_state'):objects[name]=plain(item.get_geom_state())
                return dict(success=success,predicates=predicates,goal_object_geometry=objects,time_s=float(state[0]))
            original=endpoint(source_final);actual=endpoint(actual_final)
            result=dict(case=job['case'],suite=task['suite'],task_id=task['task_id'],demo=demo,root=str(root),
                execution_rejected=report['execution_rejected'],executed_frames=report['executed_frames'],expected_frames=report['expected_frames'],
                recorded_final_success=report['success'],recorded_ever_success=report['ever_success'],
                recorded_native_goal_at_stop=native_goal_at_stop,
                source_last_known_state=original,reconstructed_final_forwarded_state=actual,
                forwarded_goal_changed_from_recording=actual['success']!=native_goal_at_stop,physical_model_hash_matches=True,physics_steps=0)
            if not original['success']:
                # Separate missing terminal transition from possible fixture
                # placement differences. Use the native XML reset API and the
                # original assets; never run an alternative physics rollout.
                from experiments.libero_tasktol_compat.audit_original_osc_samples import remap_xml
                mapped,asset_mapping=remap_xml(archived_xml)
                env.reset_from_xml_string(mapped)
                result['archived_xml_source_last_known_state']=endpoint(source_final)
                result['archived_xml_sha256']=hashlib.sha256(archived_xml.encode()).hexdigest()
                result['archived_asset_mapping']=asset_mapping
            results.append(result)
            print(json.dumps(dict(case=job['case'],demo=demo,rejected=report['execution_rejected'],source_last_known_success=original['success'],reconstructed_forwarded_success=actual['success'])),flush=True)
        finally:env.close()
    result=dict(complete=summary['complete'],sampled_episodes=summary['completed'],failed_selected_episodes=len(results),
        full_execution_task_failures=sum(not r['execution_rejected']for r in results),solver_rejections=sum(r['execution_rejected']for r in results),
        source_last_known_goal_true=sum(r['source_last_known_state']['success']for r in results),
        forwarded_goal_changed=sum(r['forwarded_goal_changed_from_recording']for r in results),
        caveat='Native predicates under matched reconstruction model; source endpoint precedes omitted final source action. Forwarding removes native last-2ms derived-state cache lag. Neither endpoint query changes recorded success or training eligibility.',
        results=results,physics_steps=0)
    (run/('task_outcomes_partial.json'if partial else'task_outcomes.json')).write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--partial',action='store_true');a=p.parse_args();diagnose(a.run.resolve(),a.partial)
