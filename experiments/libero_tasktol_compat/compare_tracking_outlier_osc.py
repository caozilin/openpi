"""One diagnostic native-OSC replay of identical reconstructed pose actions.

Not a dataset recording. Dynamics and recording use the maintained worker.
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments/libero_tasktol_compat'),str(ROOT.parent/'franka_mujoco'),str(ROOT/'third_party/libero')]


def main():
    import h5py
    import numpy as np
    import rerecord_state_reference as runner
    from robosuite.controllers.osc import OperationalSpaceController
    meta=json.loads((ROOT/'experiments/libero_tasktol_compat/results/task_pose_default_ipopt400_fixed_v2/tracking_outlier_pilot.json').read_text())
    parent=Path(meta['root']);dest=parent/'libero_10/02/demo_21'
    with h5py.File(dest/'state_reference.hdf5')as f:reference={k:v[()]for k,v in f.items()}
    runner.state_reference=lambda *a,**kw:{k:v.copy()for k,v in reference.items()}
    out=ROOT/'experiments/libero_tasktol_compat/results/tracking_outlier32_same_actions_native_osc_v1';out.mkdir(exist_ok=False)
    controls=[];original=OperationalSpaceController.run_controller
    def observe(controller):
        torque=original(controller)
        controls.append(dict(time_s=float(controller.sim.data.time),raw_torque_nm=torque.tolist(),
            clipped=bool(np.any((torque<controller.actuator_min)|(torque>controller.actuator_max)))))
        return torque
    OperationalSpaceController.run_controller=observe
    job=dict(case=32,suite='libero_10',task_id=2,demo='demo_21')
    try:result=runner.worker(job,out,1,['osc_state'],1,'preserve')
    finally:OperationalSpaceController.run_controller=original
    (out/'summary.json').write_text(json.dumps(runner.plain(result),indent=2)+'\n')
    folder=out/'libero_10/02/demo_21/osc_state'
    rows=json.loads((folder/'trajectory.json').read_text())
    baseline=json.loads((dest/'ipopt_state_B10/report.json').read_text());report=json.loads((folder/'report.json').read_text())
    assert report['model_sha256']==baseline['model_sha256'] and report['initial_state_sha256']==baseline['initial_state_sha256']
    assert report['source_input_sha256']==baseline['source_input_sha256']
    with h5py.File(folder.parent/'state_reference.hdf5')as f:np.testing.assert_array_equal(f['actions'][()],reference['actions'])
    old=json.loads((dest/'ipopt_state_B10/trajectory.json').read_text())
    def errors(rows):
        p=np.array([r['source_next_position_error_m']for r in rows])*1000
        r=np.degrees([r['source_next_rotation_error_rad']for r in rows])
        return dict(mean_position_mm=float(p.mean()),p95_position_mm=float(np.percentile(p,95)),max_position_mm=float(p.max()),mean_rotation_deg=float(r.mean()),max_rotation_deg=float(np.max(r)))
    comparison=dict(diagnostic_only=True,training_data=False,same_reference_actions=True,model_and_initial_state_match=True,
        original_ipopt_root=str(parent),osc_success=report['success'],ipopt_success=baseline['success'],
        frames=len(rows),native_osc_errors=errors(rows),ipopt_B10_errors=errors(old),
        native_osc_torque_clipped_steps=sum(c['clipped']for c in controls),physics_steps=len(controls),
        original_reference_sha256=hashlib.sha256((dest/'state_reference.hdf5').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    (out/'native_controls.json').write_text(json.dumps(controls,indent=2)+'\n')
    print(json.dumps(comparison,indent=2),flush=True)

if __name__=='__main__':main()
