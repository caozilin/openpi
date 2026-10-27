"""Initialization regression against frozen native LIBERO source evidence."""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(mujoco.__version__ != '3.2.3', reason='Source replay is pinned to MuJoCo 3.2.3')


@pytest.mark.parametrize('case', [4, 23, 38])
def test_processed_context_restores_source_scene_without_extra_physics(case, tmp_path, monkeypatch):
    import h5py

    root=Path('/media/czl/sata/franka_my_code/openpi')
    monkeypatch.syspath_prepend(str(root/'third_party/libero'))
    monkeypatch.setenv('NUMBA_CACHE_DIR',str(tmp_path/'numba'))
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from tasktol.benchmark.libero_replay_state import processed_reset_context, restore_processed_state

    selected=json.loads((root/'data/libero40_mask_annotation/20261019_mainview_v1/selection_manifest.json').read_text())['tasks'][case]
    with h5py.File(selected['processed_hdf5'],'r') as file:
        context=processed_reset_context(file,selected['demo_id'])
        initial=file['data'][selected['demo_id']]['tasktol/pre_action_states'][0]
    env=OffScreenRenderEnv(
        bddl_file_name=str(Path(get_libero_path('bddl_files'))/selected['suite']/(selected['task_name']+'.bddl')),
        controller='JOINT_POSITION',camera_names=['agentview'],camera_heights=256,camera_widths=256,control_freq=20,
    )
    try:
        sampler_count=len(env.env.object_property_initializers)
        obs,metadata=restore_processed_state(env,initial,context)
        assert metadata['reset_rng_matches_source']
        assert metadata['property_sampler_count']==sampler_count*(context['prior_resets']+2)
        assert metadata['extra_physics_steps']==0
        assert env.env.timestep==0
        np.testing.assert_array_equal(env.sim.get_state().flatten(),initial)
        assert env.robots[0].controller.new_update
        np.testing.assert_array_equal(env.robots[0].gripper.current_action,[1.,-1.])
        # Verify physical fixture geometry rather than renderer-dependent
        # pixels. These poses were checked against source frames under the
        # source GPU renderer; software EGL differs photometrically.
        fixtures=json.loads((Path(__file__).parent/'fixtures/libero_source_reset_fixtures.json').read_text())[str(case)]
        for name,expected in fixtures.items():
            np.testing.assert_allclose(metadata['fixtures'][name]['position'],expected['position'],atol=1e-12,rtol=0)
            np.testing.assert_allclose(metadata['fixtures'][name]['rotation'],expected['rotation'],atol=1e-12,rtol=0)
    finally:
        env.close()


def test_inconsistent_reset_history_is_rejected(tmp_path,monkeypatch):
    import h5py

    root=Path('/media/czl/sata/franka_my_code/openpi')
    monkeypatch.syspath_prepend(str(root/'third_party/libero'))
    monkeypatch.setenv('NUMBA_CACHE_DIR',str(tmp_path/'numba'))
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from tasktol.benchmark.libero_replay_state import processed_reset_context,restore_processed_state
    selected=json.loads((root/'data/libero40_mask_annotation/20261019_mainview_v1/selection_manifest.json').read_text())['tasks'][4]
    with h5py.File(selected['processed_hdf5'],'r') as file:
        context=processed_reset_context(file,selected['demo_id'])
        initial=file['data'][selected['demo_id']]['tasktol/pre_action_states'][0]
    context['prior_resets']=0
    env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/selected['suite']/(selected['task_name']+'.bddl')),controller='JOINT_POSITION',camera_names=['agentview'],camera_heights=32,camera_widths=32)
    try:
        with pytest.raises(RuntimeError,match='Source reset RNG did not match'):
            restore_processed_state(env,initial,context)
    finally:
        env.close()


def test_zero_tolerance_matches_nominal_across_stage_and_subtask_changes(tmp_path,monkeypatch):
    import h5py

    root=Path('/media/czl/sata/franka_my_code/openpi')
    monkeypatch.syspath_prepend(str(root/'third_party/libero'))
    monkeypatch.setenv('NUMBA_CACHE_DIR',str(tmp_path/'numba'))
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from tasktol.benchmark.libero_replay_state import processed_reset_context,restore_processed_state
    from tasktol.simulation.libero import LiberoTaskTolSession
    selected=json.loads((root/'data/libero40_mask_annotation/20261019_mainview_v1/selection_manifest.json').read_text())['tasks'][0]
    with h5py.File(selected['processed_hdf5'],'r') as file:
        context=processed_reset_context(file,selected['demo_id'])
        initial=file['data'][selected['demo_id']]['tasktol/pre_action_states'][0]
    transitions=[('APP',0),('ENG',0),('EXEC',0),('DIS',0),('APP',1),('ENG',1)]
    controls=[
        np.array([.001,0,0,0,.002,0,-1]),np.array([0,.001,0,.001,0,0,-1]),
        np.array([0,0,-.001,0,0,.001,-1]),np.array([-.001,0,0,0,-.002,0,-1]),
        np.array([0,-.001,0,-.001,0,0,-1]),np.array([0,0,.001,0,0,-.001,-1]),
    ]
    results=[]
    for zero_enabled in [False,True]:
        env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/selected['suite']/(selected['task_name']+'.bddl')),controller='JOINT_POSITION',camera_names=['agentview'],camera_heights=32,camera_widths=32,control_freq=20)
        try:
            restore_processed_state(env,initial,context)
            session=LiberoTaskTolSession(env)
            rows=[]
            for action,(phase,subtask) in zip(controls,transitions):
                _,row=session.step(action,phase=phase,subtask_index=subtask,semantic_mask='111',bounds_rad=np.zeros(6) if zero_enabled else None)
                assert row['optimizer']['feasible'] and not row['aborted']
                assert row['effective_mask']=='000'
                rows.append(row)
            results.append(rows)
        finally:
            env.close()
    for nominal,zero in zip(*results):
        np.testing.assert_allclose(zero['joint_target_rad'],nominal['joint_target_rad'],atol=1e-8,rtol=0)
        np.testing.assert_array_equal(zero['nominal_target']['position_m'],nominal['nominal_target']['position_m'])
        np.testing.assert_allclose(zero['post_action']['joint_position_rad'],nominal['post_action']['joint_position_rad'],atol=1e-8,rtol=0)
