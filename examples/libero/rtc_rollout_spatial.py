from __future__ import annotations

import dataclasses
import inspect
import json
import logging
import math
import pathlib
import time
from typing import Any

import imageio
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 448
LIBERO_SPATIAL_MAX_STEPS = 220
_TORCH_LOAD_PATCHED = False


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str | None = None

    task_id: int = 0
    init_state_id: int = 0
    seed: int = 7
    resize_size: int = 224
    num_steps_wait: int = 10
    executed_steps: int = 5
    max_steps: int = LIBERO_SPATIAL_MAX_STEPS
    delay_seconds: float = 0.3
    control_freq: int = 20

    rtc_inference_delay: int = 0
    rtc_execution_horizon: int | None = None
    rtc_prefix_attention_schedule: str | None = None
    rtc_max_guidance_weight: float | None = None

    output_dir: pathlib.Path = pathlib.Path("data/libero/rtc_rollout")
    video_fps: int = 10


def main(args: Args) -> None:
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port, api_key=args.api_key)
    metadata = client.get_server_metadata()
    logging.info("Server metadata: %s", metadata)

    rtc_metadata = metadata.get("rtc", {})
    if not rtc_metadata.get("enabled", False):
        raise RuntimeError(f"Server RTC is not enabled. Metadata rtc={rtc_metadata}")

    task_suite = _get_libero_spatial_suite()
    task = task_suite.get_task(args.task_id)
    _patch_torch_load_for_libero_init_states()
    initial_states = task_suite.get_task_init_states(args.task_id)
    if args.init_state_id >= len(initial_states):
        raise ValueError(f"init_state_id={args.init_state_id} out of range for {len(initial_states)} initial states")

    delay_steps = round(args.delay_seconds * args.control_freq)
    logging.info("Simulated post-chunk delay: %.3fs = %d env steps", args.delay_seconds, delay_steps)

    summaries = [
        _run_rollout(
            args=args,
            client=client,
            metadata=metadata,
            task=task,
            initial_state=initial_states[args.init_state_id],
            mode=mode,
            delay_steps=delay_steps,
        )
        for mode in ("rtc", "no_rtc")
    ]

    comparison_path = args.output_dir / f"rtc_vs_no_rtc_task{args.task_id}_init{args.init_state_id}_comparison.json"
    comparison_path.write_text(json.dumps({"rollouts": summaries}, indent=2, ensure_ascii=False))
    logging.info("Comparison: %s", comparison_path)


def _run_rollout(
    *,
    args: Args,
    client: _websocket_client_policy.WebsocketClientPolicy,
    metadata: dict[str, Any],
    task,
    initial_state,
    mode: str,
    delay_steps: int,
) -> dict[str, Any]:
    np.random.seed(args.seed)
    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
    try:
        env.reset()
        obs = env.set_init_state(initial_state)

        for _ in range(args.num_steps_wait):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

        replay_images: list[np.ndarray] = []
        policy_actions: list[np.ndarray] = []
        all_actions: list[np.ndarray] = []
        action_is_delay: list[bool] = []
        states: list[np.ndarray] = []
        rewards: list[float] = []
        chunk_logs: list[dict[str, Any]] = []

        prev_chunk_left_over: np.ndarray | None = None
        done = False
        info: dict[str, Any] = {}
        env_step = 0
        chunk_index = 0

        while env_step < args.max_steps and not done:
            request = _make_policy_request(obs, task_description, args.resize_size)
            using_rtc = mode == "rtc" and prev_chunk_left_over is not None
            rtc_inference_delay = args.rtc_inference_delay
            if using_rtc:
                request["rtc"] = _make_rtc_request(
                    args,
                    prev_chunk_left_over,
                    inference_delay=rtc_inference_delay,
                )

            infer_start = time.monotonic()
            response = client.infer(request)
            client_ms = (time.monotonic() - infer_start) * 1000

            actions = np.asarray(response["actions"], dtype=np.float32)
            model_actions = _require_rtc_model_actions(
                response, f"{mode}_chunk_{chunk_index}_{'rtc' if using_rtc else 'no_rtc'}"
            )
            if not 0 < args.executed_steps < len(actions):
                raise ValueError(
                    f"executed_steps must be in [1, {len(actions) - 1}], got {args.executed_steps}"
                )

            steps_to_execute = min(args.executed_steps, args.max_steps - env_step)
            chunk_log = {
                "chunk_index": chunk_index,
                "start_env_step": env_step,
                "mode": mode,
                "using_rtc": using_rtc,
                "prev_chunk_left_over_shape": None
                if prev_chunk_left_over is None
                else list(prev_chunk_left_over.shape),
                "rtc_inference_delay": rtc_inference_delay if using_rtc else None,
                "post_chunk_delay_steps_requested": delay_steps if mode == "no_rtc" else 0,
                "actions_shape": list(actions.shape),
                "model_actions_shape": list(model_actions.shape),
                "policy_steps_executed": steps_to_execute,
                "client_ms": client_ms,
                "policy_timing": _jsonable(response.get("policy_timing", {})),
                "server_timing": _jsonable(response.get("server_timing", {})),
                "state": _jsonable(np.asarray(request["observation/state"])),
            }

            chunk_policy_rewards = []
            for action in actions[:steps_to_execute]:
                obs, reward, done, info = _step_and_record(
                    env=env,
                    obs=obs,
                    action=action,
                    replay_images=replay_images,
                    states=states,
                    all_actions=all_actions,
                    rewards=rewards,
                    action_is_delay=action_is_delay,
                    is_delay=False,
                )
                policy_actions.append(action)
                chunk_policy_rewards.append(float(reward))
                env_step += 1
                if done or env_step >= args.max_steps:
                    break

            chunk_delay_rewards = []
            delay_steps_executed = 0
            while mode == "no_rtc" and delay_steps_executed < delay_steps and not done and env_step < args.max_steps:
                obs, reward, done, info = _step_and_record(
                    env=env,
                    obs=obs,
                    action=np.asarray(LIBERO_DUMMY_ACTION, dtype=np.float32),
                    replay_images=replay_images,
                    states=states,
                    all_actions=all_actions,
                    rewards=rewards,
                    action_is_delay=action_is_delay,
                    is_delay=True,
                )
                chunk_delay_rewards.append(float(reward))
                delay_steps_executed += 1
                env_step += 1

            chunk_log["end_env_step"] = env_step
            chunk_log["policy_rewards"] = chunk_policy_rewards
            chunk_log["delay_steps_executed"] = delay_steps_executed
            chunk_log["delay_rewards"] = chunk_delay_rewards
            chunk_log["done_after_chunk"] = bool(done)
            chunk_log["info_after_chunk"] = _jsonable(info)
            chunk_logs.append(chunk_log)

            prev_chunk_left_over = model_actions[steps_to_execute:]
            if mode == "rtc" and len(prev_chunk_left_over) <= delay_steps:
                logging.warning(
                    "RTC overlap is exhausted for next chunk: tail=%d, simulated_delay=%d.",
                    len(prev_chunk_left_over),
                    delay_steps,
                )
            if len(prev_chunk_left_over) == 0:
                logging.warning("No model-action tail remains after executing %d policy steps; stopping.", steps_to_execute)
                break

            chunk_index += 1

        if replay_images:
            replay_images.append(_make_replay_image(obs))

        suffix = "success" if done else "failure"
        task_segment = _safe_name(task_description)
        prefix = f"{mode}_rollout_task{args.task_id}_init{args.init_state_id}_{suffix}"
        video_path = args.output_dir / f"{prefix}.mp4"
        imageio.mimwrite(video_path, replay_images, fps=args.video_fps)

        summary = {
            "test_type": "full_rollout_rtc_vs_no_rtc_with_post_chunk_delay",
            "mode": mode,
            "task_suite": "libero_spatial",
            "task_id": args.task_id,
            "init_state_id": args.init_state_id,
            "seed": args.seed,
            "same_condition_note": (
                "The LIBERO env seed, task, and init state are identical across modes. "
                "The websocket server does not expose per-request diffusion noise control, "
                "so model sampling noise is not reset between rollout modes."
            ),
            "task_description": task_description,
            "server_metadata": _jsonable(metadata),
            "success": bool(done),
            "num_env_steps": env_step,
            "num_policy_actions": len(policy_actions),
            "num_delay_actions": int(np.sum(action_is_delay)) if action_is_delay else 0,
            "num_chunks": len(chunk_logs),
            "num_steps_wait": args.num_steps_wait,
            "executed_steps": args.executed_steps,
            "max_steps": args.max_steps,
            "delay_seconds": args.delay_seconds,
            "control_freq": args.control_freq,
            "no_rtc_dummy_delay_steps": delay_steps,
            "rtc_dummy_delay_steps": 0,
            "total_reward": float(np.sum(rewards)) if rewards else 0.0,
            "final_info": _jsonable(info),
            "video_path": str(video_path),
            "rtc_request_defaults": {
                "inference_delay": args.rtc_inference_delay,
                "execution_horizon": args.rtc_execution_horizon,
                "prefix_attention_schedule": args.rtc_prefix_attention_schedule,
                "max_guidance_weight": args.rtc_max_guidance_weight,
            },
            "chunks": chunk_logs,
        }

        summary_path = args.output_dir / f"{prefix}.json"
        arrays_path = args.output_dir / f"{prefix}.npz"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        np.savez_compressed(
            arrays_path,
            policy_actions=np.asarray(policy_actions, dtype=np.float32),
            all_actions=np.asarray(all_actions, dtype=np.float32),
            action_is_delay=np.asarray(action_is_delay, dtype=np.bool_),
            states=np.asarray(states, dtype=np.float32),
            rewards=np.asarray(rewards, dtype=np.float32),
        )

        logging.info("%s LIBERO rollout finished.", mode.upper())
        logging.info("Task: %s", task_segment)
        logging.info("Success: %s", done)
        logging.info("Env steps: %d", env_step)
        logging.info("Policy actions: %d", len(policy_actions))
        logging.info("Delay actions: %d", summary["num_delay_actions"])
        logging.info("Chunks: %d", len(chunk_logs))
        logging.info("Total reward: %.6f", summary["total_reward"])
        logging.info("Video: %s", video_path)
        logging.info("Summary: %s", summary_path)
        logging.info("Arrays: %s", arrays_path)
        return summary
    finally:
        env.close()


def _make_policy_request(obs: dict[str, Any], task_description: str, resize_size: int) -> dict[str, Any]:
    img_raw = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img_raw = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img_raw, resize_size, resize_size))
    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img_raw, resize_size, resize_size))
    state = _make_state(obs)

    return {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": state,
        "prompt": str(task_description),
    }


def _make_replay_image(obs: dict[str, Any]) -> np.ndarray:
    img_raw = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    return image_tools.convert_to_uint8(img_raw)


def _step_and_record(
    *,
    env,
    obs: dict[str, Any],
    action: np.ndarray,
    replay_images: list[np.ndarray],
    states: list[np.ndarray],
    all_actions: list[np.ndarray],
    rewards: list[float],
    action_is_delay: list[bool],
    is_delay: bool,
):
    replay_images.append(_make_replay_image(obs))
    states.append(_make_state(obs).astype(np.float32))
    obs, reward, done, info = env.step(action.tolist())
    all_actions.append(np.asarray(action, dtype=np.float32))
    rewards.append(float(reward))
    action_is_delay.append(is_delay)
    return obs, reward, done, info


def _make_state(obs: dict[str, Any]) -> np.ndarray:
    return np.concatenate(
        (
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    )


def _make_rtc_request(args: Args, prev_chunk_left_over: np.ndarray, *, inference_delay: int) -> dict[str, Any]:
    request = {
        "enabled": True,
        "prev_chunk_left_over": prev_chunk_left_over,
        "inference_delay": inference_delay,
    }
    if args.rtc_execution_horizon is not None:
        request["execution_horizon"] = args.rtc_execution_horizon
    if args.rtc_prefix_attention_schedule is not None:
        request["prefix_attention_schedule"] = args.rtc_prefix_attention_schedule
    if args.rtc_max_guidance_weight is not None:
        request["max_guidance_weight"] = args.rtc_max_guidance_weight
    return request


def _require_rtc_model_actions(response: dict[str, Any], label: str) -> np.ndarray:
    rtc_payload = response.get("rtc")
    if not isinstance(rtc_payload, dict) or "model_actions" not in rtc_payload:
        raise RuntimeError(f"{label} response does not include rtc.model_actions. Keys={list(response.keys())}")
    return np.asarray(rtc_payload["model_actions"], dtype=np.float32)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):  # noqa: UP038
        return [_jsonable(item) for item in value]
    return value


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_")


def _get_libero_env(task, resolution: int, seed: int):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _get_libero_spatial_suite():
    from libero.libero import benchmark

    return benchmark.get_benchmark_dict()["libero_spatial"]()


def _patch_torch_load_for_libero_init_states() -> None:
    """LIBERO init-state files predate PyTorch 2.6's weights_only default."""
    import torch

    global _TORCH_LOAD_PATCHED  # noqa: PLW0603
    if _TORCH_LOAD_PATCHED:
        return

    original_torch_load = torch.load
    supports_weights_only = "weights_only" in inspect.signature(original_torch_load).parameters

    def torch_load_with_legacy_default(*args, **kwargs):
        if supports_weights_only:
            kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = torch_load_with_legacy_default
    _TORCH_LOAD_PATCHED = True


def _quat2axisangle(quat):
    quat = np.asarray(quat, dtype=np.float64).copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
