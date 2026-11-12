"""Small numerical audit of the real training step; no checkpoints or cloud runs.

Run with CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 XLA_PYTHON_CLIENT_PREALLOCATE=false.
The full pi0.5 model is replaced with a tiny model so identical global batches
fit on both one GPU and seven GPUs. Training, optimizer, loss groups, sharding,
and the data loader are the project's actual implementations.
"""
import argparse
import dataclasses
import functools
import json
from pathlib import Path
import sys

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts'))
import train
from openpi.models import model as model_lib
from openpi.models.pi0 import _grouped_action_loss_with_breakdown
from openpi.shared import array_typing as at
from openpi.training import config as config_lib
from openpi.training import data_loader, optimizer, sharding, utils


class TinyModel(model_lib.BaseModel):
    def __init__(self, action_dim, dtype, add_noise):
        super().__init__(action_dim, 16, 48)
        self.kernel = nnx.Param(jnp.asarray(np.linspace(-.1, .1, 7 * action_dim).reshape(7, action_dim), jnp.float32))
        self.bias = nnx.Param(jnp.zeros(action_dim))
        self.dtype = dtype
        self.add_noise = add_noise

    def compute_loss_with_breakdown(self, rng, observation, actions, *, train=False):
        inputs = sharding.activation_sharding_constraint(observation.state)
        prediction = (inputs.astype(self.dtype) @ self.kernel.value.astype(self.dtype)).astype(jnp.float32)
        prediction = prediction[:, None, :] + self.bias.value
        if self.add_noise:
            prediction = prediction + .01 * jax.random.normal(rng, actions.shape)
        squared = jnp.square(prediction - actions)
        return _grouped_action_loss_with_breakdown(
            squared, (('action', 0, 7, 1.), ('rotation_tolerance', 7, 13, .5)), observation.action_loss_weight,
        )

    def compute_loss(self, rng, observation, actions, *, train=False):
        return self.compute_loss_with_breakdown(rng, observation, actions, train=train)[0]

    def sample_actions(self, rng, observation, **kwargs):
        raise NotImplementedError


def maximum_difference(a, b, *, rtol, atol, norm_bound=None):
    maximum = 0.
    maximum_relative_norm = 0.
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        x, y = np.asarray(x), np.asarray(y)
        relative_norm = float(np.linalg.norm(x.astype(float) - y.astype(float)) / max(np.linalg.norm(x.astype(float)), 1e-12))
        maximum_relative_norm = max(maximum_relative_norm, relative_norm)
        if norm_bound is None:
            np.testing.assert_allclose(x, y, rtol=rtol, atol=atol)
        else:
            # BF16 dot products quantize partial sums differently at different
            # batch partitions. A relative L2 bound avoids unstable relative
            # errors for individual near-zero entries; report raw errors too.
            assert relative_norm <= norm_bound, f'BF16 relative L2 error {relative_norm} > {norm_bound}'
        if x.size:
            maximum = max(maximum, float(np.max(np.abs(x.astype(float) - y.astype(float)))))
    return {'max_absolute': maximum, 'max_leaf_relative_l2': maximum_relative_norm}


def run_case(batch_size, action_dim, dtype, optimizer_name, add_noise):
    generator = np.random.default_rng(123)
    x = generator.normal(size=(batch_size, 7)).astype(np.float32)
    actions = generator.normal(size=(batch_size, 16, action_dim)).astype(np.float32)
    weights = np.where(np.arange(batch_size) % 11 < 3, 4., 1.).astype(np.float32)
    observation = model_lib.Observation(
        images={'base': np.zeros((batch_size, 2, 2, 3), np.float32)},
        image_masks={'base': np.ones(batch_size, bool)}, state=x, action_loss_weight=weights,
    )
    config = dataclasses.replace(
        config_lib.get_config('debug_pi05'), batch_size=batch_size, freeze_filter=nnx.Nothing,
        optimizer=optimizer.SGD(lr=.01) if optimizer_name == 'sgd' else optimizer.AdamW(),
        # A visible update magnitude makes a wrong gradient scale easier to
        # detect. This is a numerical diagnostic, not a proposed training LR.
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=1, peak_lr=1e-3, decay_lr=1e-3), ema_decay=None,
    )
    model = TinyModel(action_dim, dtype, add_noise)
    graph, params = nnx.split(model)
    tx = optimizer.create_optimizer(config.optimizer, config.lr_schedule)
    opt_state = jax.tree_util.tree_map_with_path(
        lambda path, value: jnp.full_like(value, 9000) if getattr(path[-1], 'name', None) == 'count' else value,
        tx.init(params),
    )
    initial = utils.TrainState(step=jnp.asarray(9000), params=params, model_def=graph, tx=tx,
                               opt_state=opt_state, ema_decay=None, ema_params=None)
    results = []
    for count in (1, 7):
        mesh = jax.sharding.Mesh(np.array(jax.devices()[:count]).reshape(count, 1),
                                 (sharding.BATCH_AXIS, sharding.FSDP_AXIS))
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        data_spec = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        state_spec = sharding.fsdp_sharding(jax.eval_shape(lambda: initial), mesh)
        state = jax.device_put(initial, state_spec)
        batch = jax.tree.map(lambda a: jax.make_array_from_process_local_data(data_spec, a), (observation, actions))
        rng = jax.device_put(jax.random.key(42), replicated)
        step = jax.jit(functools.partial(train.train_step, config),
                       in_shardings=(replicated, state_spec, data_spec), out_shardings=(state_spec, replicated))
        infos = []
        for _ in range(3):
            with sharding.set_mesh(mesh):
                state, info = step(rng, state, batch)
            infos.append(jax.device_get(info))
        jax.block_until_ready(state)
        # Explicit .lower() uses ArgInfo placeholders that the project's
        # runtime type checker does not accept; normal training above retains
        # all type checks. Disable checks only for compiler-text inspection.
        with at.disable_typechecking(), sharding.set_mesh(mesh):
            hlo = step.lower(rng, state, batch).compile().as_text()
        results.append({'state': jax.device_get(state), 'infos': infos, 'collectives': hlo.count('all-reduce')})
    rtol, atol = (5e-3, 3e-5) if dtype == jnp.bfloat16 else (2e-5, 2e-6)
    first, second = results
    norm_bound = .01 if dtype == jnp.bfloat16 else None
    info_error = maximum_difference(first['infos'], second['infos'], rtol=rtol, atol=atol, norm_bound=norm_bound)
    parameter_error = maximum_difference(first['state'].params, second['state'].params, rtol=rtol, atol=atol, norm_bound=norm_bound)
    optimizer_error = maximum_difference(first['state'].opt_state, second['state'].opt_state, rtol=rtol, atol=atol, norm_bound=norm_bound)
    assert second['collectives'] > 0, 'No collective communication in compiled seven-device training step'
    return {'batch': batch_size, 'action_dim': action_dim, 'dtype': str(dtype), 'optimizer': optimizer_name,
            'add_noise': add_noise, 'updates': 3, 'max_metric_difference': info_error,
            'max_parameter_difference': parameter_error, 'max_optimizer_state_difference': optimizer_error,
            'comparison': {'elementwise_rtol': rtol, 'elementwise_atol': atol, 'relative_l2_bound': norm_bound},
            'compiled_all_reduce_mentions': second['collectives'],
            'single_device_metrics': [{k: float(v) for k, v in row.items()} for row in first['infos']],
            'seven_device_metrics': [{k: float(v) for k, v in row.items()} for row in second['infos']], 'passed': True}


def check_loader():
    mesh = sharding.make_mesh(1)
    data_spec = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    dataset = [{'id': np.asarray(i, np.int32)} for i in range(168)]
    loader = data_loader.TorchDataLoader(dataset, local_batch_size=84, sharding=data_spec,
                                        shuffle=False, num_batches=1, num_workers=0)
    batch = next(iter(loader))['id']
    shards = sorted([np.asarray(s.data).tolist() for s in batch.addressable_shards], key=lambda row: row[0])
    assert [i for shard in shards for i in shard] == list(range(84))
    assert all(len(row) == 12 for row in shards) and len(shards) == 7
    return {'global_shape': list(batch.shape), 'sample_ids_per_device': shards, 'passed': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert len(jax.devices()) == 7
    output = {'devices': [str(d) for d in jax.devices()], 'jax_version': jax.__version__,
              'process_count': jax.process_count(), 'scope': 'Small-model numerical audit, not full pi0.5 equivalence',
              'loader': check_loader(), 'cases': []}
    for case in [(84, 13, jnp.float32, 'sgd', False), (84, 32, jnp.float32, 'adamw', False),
                 (28, 32, jnp.bfloat16, 'adamw', False), (84, 13, jnp.float32, 'adamw', True)]:
        result = run_case(*case)
        output['cases'].append(result)
        print('CASE_PASSED', json.dumps(result), flush=True)
    output['passed'] = True
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print('AUDIT_PASSED', args.output, flush=True)


if __name__ == '__main__':
    main()
