"""Shared SwanLab initialization and camera visualization for both trainers."""

import dataclasses
import logging
import os
from typing import TYPE_CHECKING

import numpy as np
import swanlab

if TYPE_CHECKING:
    from openpi.training.config import TrainConfig


def init_swanlab(config: "TrainConfig", *, resuming: bool):
    """Resume cloud tracking when possible; legacy checkpoints start a new experiment.

    Model checkpoint restoration is independent of the tracking backend. In
    particular, a legacy wandb_id.txt is never treated as a SwanLab experiment ID.
    Offline/local sessions start separate logs, as SwanLab only resumes online runs.
    """
    mode = (config.swanlab_mode or os.environ.get("SWANLAB_MODE", "online")).lower()
    if mode == "cloud":
        mode = "online"
    if not config.swanlab_enabled or mode == "disabled":
        return swanlab.init(mode="disabled")

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    kwargs = {
        "project": config.project_name,
        "experiment_name": config.exp_name,
        "config": dataclasses.asdict(config),
        "mode": mode,
    }
    if config.swanlab_workspace is not None:
        kwargs["workspace"] = config.swanlab_workspace
    if config.swanlab_log_dir is not None:
        kwargs["logdir"] = config.swanlab_log_dir

    run_id_path = ckpt_dir / "swanlab_id.txt"
    if resuming and mode == "online":
        if run_id_path.is_file():
            run_id = run_id_path.read_text().strip()
            if not run_id:
                raise ValueError(f"SwanLab experiment ID is empty in {run_id_path}.")
            kwargs.update(id=run_id, resume="must")
        else:
            logging.info("No SwanLab experiment ID in %s; starting a new experiment for the resumed model.", ckpt_dir)
    elif resuming:
        logging.info("SwanLab %s mode starts a new log session; model training resumes from its checkpoint.", mode)

    run = swanlab.init(**kwargs)
    # Offline IDs cannot be resumed online until their logs have been synced.
    # Keep any existing cloud ID intact when temporarily recording offline.
    if mode == "online":
        run_id_path.write_text(run.id)
    return run


def camera_image(image: np.ndarray) -> swanlab.Image:
    """Convert an HWC camera image in the model's [-1, 1] range to RGB bytes."""
    pixels = np.asarray(image)
    if pixels.dtype != np.uint8:
        pixels = np.clip((pixels + 1.0) * 127.5, 0, 255).astype(np.uint8)
    return swanlab.Image(pixels)
