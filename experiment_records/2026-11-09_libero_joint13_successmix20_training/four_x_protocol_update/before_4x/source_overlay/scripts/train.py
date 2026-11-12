import dataclasses
import functools
import logging
import math
import pathlib
import platform
import time
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import swanlab
import tqdm_loggable.auto as tqdm

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.experiment_tracking as experiment_tracking
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss, chunked_loss_info = model.compute_loss_with_breakdown(rng, observation, actions, train=True)
        loss_info = {name: jnp.mean(value) for name, value in chunked_loss_info.items()}
        return jnp.mean(chunked_loss), loss_info

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, loss_info), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        **loss_info,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def validation_step(
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
    rng: at.KeyArrayLike,
) -> dict[str, at.Array]:
    """Compute loss without augmentation, gradients, or optimizer updates."""
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    observation, actions = batch
    loss, breakdown = model.compute_loss_with_breakdown(rng, observation, actions, train=False)
    return {"loss": jnp.mean(loss), **{name: jnp.mean(value) for name, value in breakdown.items()}}


def run_validation(validation_loader: _data_loader.DataLoader, pvalidation_step, state, rng, *, name: str):
    started = time.perf_counter()
    totals = None
    count = 0
    for index, batch in enumerate(validation_loader):
        metrics = pvalidation_step(state, batch, jax.random.fold_in(rng, index))
        totals = metrics if totals is None else jax.tree.map(lambda x, y: x + y, totals, metrics)
        count += 1
    if count == 0:
        raise ValueError(f"Validation dataset {name!r} contains no complete batches.")
    averages = jax.device_get(jax.tree.map(lambda x: x / count, totals))
    prefix = f"validation/{name}"
    result = {f"{prefix}/{metric}": float(value) for metric, value in averages.items()}
    result[f"{prefix}/num_batches"] = count
    result[f"{prefix}/duration_seconds"] = time.perf_counter() - started
    return result


def validation_batch_size(train_batch_size: int, factor: float, num_devices: int) -> int:
    """Round down to a sharding-compatible batch size, keeping one example per device."""
    return max(num_devices, int(train_batch_size * factor) // num_devices * num_devices)


def validation_batch_count(train_batch_size: int, reference_batches: int, val_batch_size: int) -> int:
    """Keep the validation sample budget stable when its batch size changes."""
    return math.ceil(reference_batches * train_batch_size / val_batch_size)


def _same_lerobot_dataset(first: _config.DataConfig, second: _config.DataConfig) -> bool:
    def dataset_root(data_config):
        root = data_config.lerobot_dataset_root
        return pathlib.Path(root).expanduser().resolve() if root is not None else None

    first_root, second_root = dataset_root(first), dataset_root(second)
    return (first_root is not None and first_root == second_root) or (
        first.repo_id == second.repo_id and first_root == second_root
    )


def create_validation_loaders(config: _config.TrainConfig, train_data_config: _config.DataConfig, data_sharding):
    """Load each independent validation dataset with the training transforms and statistics."""
    validation_datasets = config.validation_datasets
    if not validation_datasets:
        return {}, 0
    if train_data_config.rlds_data_dir is not None:
        raise ValueError("Loss validation currently supports LeRobot datasets only.")

    # Check all sources before starting any validation data-loader workers.
    data_configs = {"training": train_data_config}
    for dataset in validation_datasets:
        data_config = dataclasses.replace(
            train_data_config,
            repo_id=dataset.repo_id,
            lerobot_dataset_root=dataset.lerobot_dataset_root,
            hf_datasets_cache_dir=dataset.hf_datasets_cache_dir,
            lerobot_arrow_cache_dir=dataset.lerobot_arrow_cache_dir,
        )
        for other_name, other_config in data_configs.items():
            if _same_lerobot_dataset(data_config, other_config):
                raise ValueError(f"Validation dataset {dataset.name!r} must be distinct from {other_name!r}.")
        data_configs[f"validation/{dataset.name}"] = data_config

    batch_size = validation_batch_size(config.batch_size, config.validation_batch_size_factor, jax.device_count())
    num_batches = validation_batch_count(config.batch_size, config.validation_num_batches, batch_size)
    loaders = {}
    for dataset in validation_datasets:
        logging.info(
            "Validation %s: repo_id=%s, batch size=%d, max batches=%d (training batch size=%d, factor=%g)",
            dataset.name,
            dataset.repo_id,
            batch_size,
            num_batches,
            config.batch_size,
            config.validation_batch_size_factor,
        )
        loaders[dataset.name] = _data_loader.create_torch_data_loader(
            data_configs[f"validation/{dataset.name}"],
            model_config=config.model,
            action_horizon=config.model.action_horizon,
            batch_size=batch_size,
            sharding=data_sharding,
            shuffle=False,
            num_batches=num_batches,
            repeat=False,
            num_workers=config.num_workers,
            seed=config.seed,
        )
    return loaders, batch_size


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    experiment_tracking.init_swanlab(config, resuming=resuming)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    validation_loaders, val_batch_size = create_validation_loaders(config, data_loader.data_config(), data_sharding)

    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Images are normalized to [-1, 1] by the model input transforms.
    if config.swanlab_enabled and not resuming:
        images_to_log = [
            experiment_tracking.camera_image(
                np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1)
            )
            for i in range(min(5, len(next(iter(batch[0].images.values())))))
        ]
        swanlab.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    pvalidation_step = None
    if validation_loaders:
        pvalidation_step = jax.jit(
            validation_step,
            in_shardings=(train_state_sharding, data_sharding, replicated_sharding),
            out_shardings=replicated_sharding,
        )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        completed_step = step + 1
        step_metrics = {}
        infos.append(info)
        if completed_step % config.log_interval == 0 or completed_step == config.num_train_steps:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {completed_step}: {info_str}")
            step_metrics.update({key: float(value) for key, value in reduced_info.items()})
            infos = []

        save_checkpoint = completed_step % config.save_interval == 0 or completed_step == config.num_train_steps
        if save_checkpoint:
            for name, validation_loader in validation_loaders.items():
                with sharding.set_mesh(mesh):
                    validation_metrics = run_validation(
                        validation_loader, pvalidation_step, train_state, train_rng, name=name
                    )
                prefix = f"validation/{name}"
                validation_metrics[f"{prefix}/batch_size"] = val_batch_size
                validation_metrics[f"{prefix}/num_samples"] = (
                    validation_metrics[f"{prefix}/num_batches"] * val_batch_size
                )
                pbar.write(f"Step {completed_step} validation [{name}]: {validation_metrics}")
                step_metrics.update(validation_metrics)
        if step_metrics and config.swanlab_enabled:
            # Keep training and all validation metrics aligned to the same completed step.
            swanlab.log(step_metrics, step=completed_step)
        if save_checkpoint:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, completed_step)
        if completed_step < config.num_train_steps:
            batch = next(data_iter)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()
    swanlab.finish()


if __name__ == "__main__":
    main(_config.cli())
