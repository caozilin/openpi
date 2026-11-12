"""Recover native LIBERO program state omitted from flat physics snapshots.

This only initializes an environment. It does not step physics or implement a
controller. The source regeneration ledger makes reset history reproducible.
"""

from __future__ import annotations

import hashlib
import json
import random

import numpy as np


def processed_reset_context(file, demo_id):
    configuration = json.loads(file.attrs['configuration'])
    order = configuration['demo_ids']
    index = order.index(demo_id)
    ledger = file['regeneration/episodes']
    return dict(
        source_seed=configuration['seed'],
        settling_steps=configuration['settle_steps'],
        prior_resets=index,
        before_rng=None if index == 0 else json.loads(ledger[order[index-1]].attrs['rng_state']),
        after_rng=json.loads(ledger[demo_id].attrs['rng_state']),
    )


def restore_processed_state(env, initial, context):
    """Restore a settled source snapshot into a newly constructed native env.

    Native hard resets append property samplers without clearing their list.
    Replicate the identical immutable sampler entries from preceding resets so
    the next native reset consumes exactly the source random stream. Verify
    this against its ledger rather than accepting a plausible random layout.
    """
    native = env.env
    original_samplers = tuple(native.object_property_initializers)
    native.object_property_initializers = list(original_samplers) * (context['prior_resets'] + 1)
    env.seed(context['source_seed'])
    before = context['before_rng']
    if before is not None:
        state = before['numpy']
        np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), state[2], state[3], state[4]))
        state = before['python']
        random.setstate((state[0], tuple(state[1]), state[2]))
    env.reset()
    actual = np.random.get_state()
    expected = context['after_rng']['numpy']
    matched = (
        actual[0] == expected[0]
        and actual[1].tolist() == expected[1]
        and list(actual[2:]) == expected[2:]
    )
    if not matched:
        raise RuntimeError(
            'Source reset RNG did not match: cannot claim identical fixture layout; '
            'the source may have resumed with a shorter reset history'
        )
    reset_site_rotation = env.sim.data.site_xmat[env.robots[0].eef_site_id].reshape(3, 3).copy()
    obs = env.set_init_state(initial)
    physics_steps = int(round(native.control_timestep / env.sim.model.opt.timestep))
    for robot in env.robots:
        robot.controller.new_update = True
        # Rehydrate the known source settling history through the gripper's
        # own accumulator. This neither moves fingers nor advances physics.
        for _ in range(context['settling_steps'] * physics_steps):
            robot.gripper.format_action(np.array([-1.0]))
    fixtures = {
        name: dict(
            position=env.sim.model.body_pos[env.sim.model.body_name2id(obj.root_body)].tolist(),
            rotation=env.sim.model.body_quat[env.sim.model.body_name2id(obj.root_body)].tolist(),
        )
        for name, obj in native.fixtures_dict.items()
    }
    metadata = dict(
        mode='source_context', source_seed=context['source_seed'], prior_resets=context['prior_resets'],
        property_sampler_count=len(native.object_property_initializers), reset_rng_matches_source=matched,
        context_sha256=hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest(),
        fixtures=fixtures, initial_gripper_action=env.robots[0].gripper.current_action.tolist(),
        source_initial_held_site_rotation=reset_site_rotation.tolist(), extra_physics_steps=0,
    )
    return obs, metadata
