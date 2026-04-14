"""
Inspect the final training inputs produced by an OpenPI config.

This script runs the same data pipeline used during training:

- LeRobot dataset loading
- repack transforms
- robot/data transforms
- normalization with computed norm stats
- model transforms

It then saves a small visualization package to disk so you can sanity-check that
the actual model inputs look correct before training.

Example:
    uv run examples/franka/inspect_training_inputs.py \
      --config-name pi05_franka_lora \
      --num-samples 16 \
      --output-dir ./debug/inspect_pi05_franka_lora
"""

from __future__ import annotations

import csv
import json
import math
import pathlib
from typing import Any

import imageio.v2 as imageio
import numpy as np
import tyro

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


def _to_numpy(x: Any) -> np.ndarray:
    return np.asarray(x)


def _to_uint8_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image
    image = np.clip((image + 1.0) * 0.5 * 255.0, 0.0, 255.0).astype(np.uint8)
    return image


def _make_panel(images: list[np.ndarray], pad: int = 4) -> np.ndarray:
    valid = [_to_uint8_image(img) for img in images]
    height = max(img.shape[0] for img in valid)
    width = sum(img.shape[1] for img in valid) + pad * (len(valid) - 1)
    panel = np.zeros((height, width, 3), dtype=np.uint8)

    cursor = 0
    for img in valid:
        h, w = img.shape[:2]
        panel[:h, cursor : cursor + w] = img
        cursor += w + pad
    return panel


def _sample_summary(
    sample_index: int,
    obs,
    actions: np.ndarray,
) -> dict[str, Any]:
    state = _to_numpy(obs.state)
    base_mask = bool(_to_numpy(obs.image_masks["base_0_rgb"]))
    left_mask = bool(_to_numpy(obs.image_masks["left_wrist_0_rgb"]))
    right_mask = bool(_to_numpy(obs.image_masks["right_wrist_0_rgb"]))

    token_len = 0
    token_preview: list[int] = []
    if obs.tokenized_prompt is not None:
        tokens = _to_numpy(obs.tokenized_prompt).astype(np.int32)
        token_len = int(tokens.shape[-1])
        token_preview = tokens[: min(16, len(tokens))].tolist()

    return {
        "sample_index": sample_index,
        "state_shape": list(state.shape),
        "state_min": float(state.min()),
        "state_max": float(state.max()),
        "state_mean": float(state.mean()),
        "state_std": float(state.std()),
        "actions_shape": list(actions.shape),
        "actions_min": float(actions.min()),
        "actions_max": float(actions.max()),
        "actions_mean": float(actions.mean()),
        "actions_std": float(actions.std()),
        "action_horizon": int(actions.shape[0]),
        "action_dim": int(actions.shape[1]),
        "base_image_mask": base_mask,
        "left_wrist_image_mask": left_mask,
        "right_wrist_image_mask": right_mask,
        "tokenized_prompt_len": token_len,
        "tokenized_prompt_preview": token_preview,
        "state_preview": state[: min(8, len(state))].tolist(),
        "first_action_preview": actions[0].tolist(),
    }


def main(
    config_name: str,
    *,
    num_samples: int = 16,
    output_dir: str = "./debug/inspect_training_inputs",
):
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")

    config = _config.get_config(config_name)
    num_batches = math.ceil(num_samples / config.batch_size)
    loader = _data_loader.create_data_loader(
        config,
        shuffle=False,
        num_batches=num_batches,
    )

    output_path = pathlib.Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    samples_dir = output_path / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    overview_panels: list[np.ndarray] = []
    saved = 0

    for batch_obs, batch_actions in loader:
        batch_size = int(_to_numpy(batch_obs.state).shape[0])
        for i in range(batch_size):
            if saved >= num_samples:
                break

            obs_i = type(batch_obs)(
                images={k: _to_numpy(v)[i] for k, v in batch_obs.images.items()},
                image_masks={k: _to_numpy(v)[i] for k, v in batch_obs.image_masks.items()},
                state=_to_numpy(batch_obs.state)[i],
                tokenized_prompt=None if batch_obs.tokenized_prompt is None else _to_numpy(batch_obs.tokenized_prompt)[i],
                tokenized_prompt_mask=None
                if batch_obs.tokenized_prompt_mask is None
                else _to_numpy(batch_obs.tokenized_prompt_mask)[i],
                token_ar_mask=None if batch_obs.token_ar_mask is None else _to_numpy(batch_obs.token_ar_mask)[i],
                token_loss_mask=None if batch_obs.token_loss_mask is None else _to_numpy(batch_obs.token_loss_mask)[i],
            )
            actions_i = _to_numpy(batch_actions)[i]

            base_image = _to_uint8_image(obs_i.images["base_0_rgb"])
            left_image = _to_uint8_image(obs_i.images["left_wrist_0_rgb"])
            right_image = _to_uint8_image(obs_i.images["right_wrist_0_rgb"])

            panel = _make_panel([base_image, left_image, right_image])
            imageio.imwrite(samples_dir / f"sample_{saved:04d}.png", panel)
            overview_panels.append(panel)

            summary = _sample_summary(saved, obs_i, actions_i)
            summaries.append(summary)

            np.savez_compressed(
                samples_dir / f"sample_{saved:04d}.npz",
                state=_to_numpy(obs_i.state),
                actions=actions_i,
                base_image=base_image,
                left_wrist_image=left_image,
                right_wrist_image=right_image,
                image_mask_base=_to_numpy(obs_i.image_masks["base_0_rgb"]),
                image_mask_left=_to_numpy(obs_i.image_masks["left_wrist_0_rgb"]),
                image_mask_right=_to_numpy(obs_i.image_masks["right_wrist_0_rgb"]),
                tokenized_prompt=np.array([], dtype=np.int32)
                if obs_i.tokenized_prompt is None
                else _to_numpy(obs_i.tokenized_prompt),
                tokenized_prompt_mask=np.array([], dtype=bool)
                if obs_i.tokenized_prompt_mask is None
                else _to_numpy(obs_i.tokenized_prompt_mask),
            )

            saved += 1

        if saved >= num_samples:
            break

    if not summaries:
        raise RuntimeError("No samples were produced by the data loader.")

    with open(output_path / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "config_name": config_name,
                "num_samples": saved,
                "output_dir": str(output_path),
                "samples": summaries,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(output_path / "summary.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sample_index",
            "state_min",
            "state_max",
            "state_mean",
            "state_std",
            "actions_min",
            "actions_max",
            "actions_mean",
            "actions_std",
            "action_horizon",
            "action_dim",
            "base_image_mask",
            "left_wrist_image_mask",
            "right_wrist_image_mask",
            "tokenized_prompt_len",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow({k: row[k] for k in fieldnames})

    if overview_panels:
        max_width = max(panel.shape[1] for panel in overview_panels)
        total_height = sum(panel.shape[0] for panel in overview_panels) + 4 * (len(overview_panels) - 1)
        overview = np.zeros((total_height, max_width, 3), dtype=np.uint8)
        cursor = 0
        for panel in overview_panels:
            h, w = panel.shape[:2]
            overview[cursor : cursor + h, :w] = panel
            cursor += h + 4
        imageio.imwrite(output_path / "overview.png", overview)

    print(f"Saved inspection package to: {output_path}")
    print(f"Overview image: {output_path / 'overview.png'}")
    print(f"Per-sample images/arrays: {samples_dir}")
    print(f"Summaries: {output_path / 'summary.json'} and {output_path / 'summary.csv'}")


if __name__ == "__main__":
    tyro.cli(main)
