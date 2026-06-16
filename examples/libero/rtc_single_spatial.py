from __future__ import annotations

import dataclasses
import inspect
import json
import logging
import math
import pathlib
import time
from typing import Any

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 448
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

    rtc_inference_delay: int = 0
    rtc_execution_horizon: int | None = None
    rtc_prefix_attention_schedule: str | None = None
    rtc_max_guidance_weight: float | None = None

    output_dir: pathlib.Path = pathlib.Path("data/libero/rtc_debug")


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

    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
    try:
        env.reset()
        obs = env.set_init_state(initial_states[args.init_state_id])

        for _ in range(args.num_steps_wait):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

        initial_request = _make_policy_request(obs, task_description, args.resize_size)
        initial_observation_debug = _build_observation_debug(obs, initial_request)

        initial_start = time.monotonic()
        initial_response = client.infer(initial_request)
        initial_client_ms = (time.monotonic() - initial_start) * 1000
        initial_actions = np.asarray(initial_response["actions"], dtype=np.float32)
        initial_model_actions = _require_rtc_model_actions(initial_response, "initial")
        if not 0 < args.executed_steps < len(initial_actions):
            raise ValueError(
                f"executed_steps must be in [1, {len(initial_actions) - 1}], got {args.executed_steps}"
            )

        executed_rewards = []
        done = False
        info = {}
        for action in initial_actions[: args.executed_steps]:
            obs, reward, done, info = env.step(action.tolist())
            executed_rewards.append(float(reward))
            if done:
                logging.warning("LIBERO episode ended while executing the initial chunk prefix.")
                break

        actual_executed_steps = len(executed_rewards)
        executed_actions = initial_actions[:actual_executed_steps]
        if actual_executed_steps != args.executed_steps:
            raise RuntimeError(
                f"LIBERO episode ended after {actual_executed_steps} steps before reaching D={args.executed_steps}."
            )

        tail_model_actions = initial_model_actions[actual_executed_steps:]
        after_execution_request = _make_policy_request(obs, task_description, args.resize_size)
        after_execution_observation_debug = _build_observation_debug(obs, after_execution_request)

        no_rtc_start = time.monotonic()
        no_rtc_response = client.infer(after_execution_request)
        no_rtc_client_ms = (time.monotonic() - no_rtc_start) * 1000
        no_rtc_model_actions = _require_rtc_model_actions(no_rtc_response, "no_rtc_after_execution")

        rtc_request = dict(after_execution_request)
        rtc_request["rtc"] = _make_rtc_request(args, tail_model_actions)

        rtc_start = time.monotonic()
        rtc_response = client.infer(rtc_request)
        rtc_client_ms = (time.monotonic() - rtc_start) * 1000
        rtc_model_actions = _require_rtc_model_actions(rtc_response, "rtc_after_execution")

        summary = _build_summary(
            args=args,
            metadata=metadata,
            task_description=task_description,
            initial_observation_debug=initial_observation_debug,
            after_execution_observation_debug=after_execution_observation_debug,
            initial_response=initial_response,
            no_rtc_response=no_rtc_response,
            rtc_response=rtc_response,
            initial_client_ms=initial_client_ms,
            no_rtc_client_ms=no_rtc_client_ms,
            rtc_client_ms=rtc_client_ms,
            initial_model_actions=initial_model_actions,
            tail_model_actions=tail_model_actions,
            no_rtc_model_actions=no_rtc_model_actions,
            rtc_model_actions=rtc_model_actions,
            executed_rewards=executed_rewards,
            done_after_execution=done,
            info_after_execution=info,
        )

        summary_path = args.output_dir / "rtc_single_spatial_summary.json"
        arrays_path = args.output_dir / "rtc_single_spatial_arrays.npz"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        np.savez_compressed(
            arrays_path,
            initial_state=np.asarray(initial_request["observation/state"]),
            after_execution_state=np.asarray(after_execution_request["observation/state"]),
            initial_actions=initial_actions,
            executed_actions=executed_actions,
            no_rtc_actions=np.asarray(no_rtc_response["actions"]),
            rtc_actions=np.asarray(rtc_response["actions"]),
            initial_model_actions=initial_model_actions,
            tail_model_actions=tail_model_actions,
            no_rtc_model_actions=no_rtc_model_actions,
            rtc_model_actions=rtc_model_actions,
            initial_observation_image=np.asarray(initial_request["observation/image"]),
            initial_observation_wrist_image=np.asarray(initial_request["observation/wrist_image"]),
            after_execution_observation_image=np.asarray(after_execution_request["observation/image"]),
            after_execution_observation_wrist_image=np.asarray(after_execution_request["observation/wrist_image"]),
        )

        logging.info("RTC LIBERO single-task smoke test finished.")
        logging.info("Summary: %s", summary_path)
        logging.info("Arrays: %s", arrays_path)
        logging.info("Initial actions shape: %s", initial_actions.shape)
        logging.info("Executed prefix shape: %s", executed_actions.shape)
        logging.info("Tail model actions shape: %s", tail_model_actions.shape)
        logging.info("No-RTC actions shape: %s", np.asarray(no_rtc_response["actions"]).shape)
        logging.info("RTC actions shape: %s", np.asarray(rtc_response["actions"]).shape)
        logging.info("No-RTC model actions shape: %s", no_rtc_model_actions.shape)
        logging.info("RTC model actions shape: %s", rtc_model_actions.shape)
        logging.info("Action L2 diff: %.6f", summary["diff"]["actions_l2"])
        logging.info("Model-action L2 diff: %.6f", summary["diff"]["model_actions_l2"])
    finally:
        env.close()


def _make_policy_request(obs: dict[str, Any], task_description: str, resize_size: int) -> dict[str, Any]:
    img_raw = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img_raw = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img_raw, resize_size, resize_size))
    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img_raw, resize_size, resize_size))
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    )

    return {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": state,
        "prompt": str(task_description),
    }


def _build_observation_debug(obs: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    return {
        "agentview_image_shape": list(np.asarray(obs["agentview_image"]).shape),
        "wrist_image_shape": list(np.asarray(obs["robot0_eye_in_hand_image"]).shape),
        "model_image_shape": list(request["observation/image"].shape),
        "model_wrist_image_shape": list(request["observation/wrist_image"].shape),
        "state": _jsonable(np.asarray(request["observation/state"])),
    }


def _make_rtc_request(args: Args, prev_chunk_left_over: np.ndarray) -> dict[str, Any]:
    request = {
        "enabled": True,
        "prev_chunk_left_over": prev_chunk_left_over,
        "inference_delay": args.rtc_inference_delay,
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


def _build_summary(
    *,
    args: Args,
    metadata: dict[str, Any],
    task_description: str,
    initial_observation_debug: dict[str, Any],
    after_execution_observation_debug: dict[str, Any],
    initial_response: dict[str, Any],
    no_rtc_response: dict[str, Any],
    rtc_response: dict[str, Any],
    initial_client_ms: float,
    no_rtc_client_ms: float,
    rtc_client_ms: float,
    initial_model_actions: np.ndarray,
    tail_model_actions: np.ndarray,
    no_rtc_model_actions: np.ndarray,
    rtc_model_actions: np.ndarray,
    executed_rewards: list[float],
    done_after_execution: bool,
    info_after_execution: dict[str, Any],
) -> dict[str, Any]:
    initial_actions = np.asarray(initial_response["actions"], dtype=np.float32)
    no_rtc_actions = np.asarray(no_rtc_response["actions"], dtype=np.float32)
    rtc_actions = np.asarray(rtc_response["actions"], dtype=np.float32)

    return {
        "test_type": "execute_initial_prefix_then_compare_no_rtc_vs_rtc",
        "task_suite": "libero_spatial",
        "task_id": args.task_id,
        "init_state_id": args.init_state_id,
        "task_description": task_description,
        "server_metadata": _jsonable(metadata),
        "execution": {
            "num_steps_wait": args.num_steps_wait,
            "executed_steps": args.executed_steps,
            "executed_rewards": executed_rewards,
            "done_after_execution": bool(done_after_execution),
            "info_after_execution": _jsonable(info_after_execution),
        },
        "rtc_request": {
            "enabled": True,
            "prev_chunk_left_over_source": "initial_model_actions[executed_steps:]",
            "prev_chunk_left_over_shape": list(tail_model_actions.shape),
            "inference_delay": args.rtc_inference_delay,
            "execution_horizon": args.rtc_execution_horizon,
            "prefix_attention_schedule": args.rtc_prefix_attention_schedule,
            "max_guidance_weight": args.rtc_max_guidance_weight,
        },
        "observation": {
            "initial": initial_observation_debug,
            "after_execution": after_execution_observation_debug,
        },
        "shapes": {
            "initial_actions": list(initial_actions.shape),
            "executed_actions": [len(executed_rewards), initial_actions.shape[1]],
            "tail_model_actions": list(tail_model_actions.shape),
            "no_rtc_actions": list(no_rtc_actions.shape),
            "rtc_actions": list(rtc_actions.shape),
            "initial_model_actions": list(initial_model_actions.shape),
            "no_rtc_model_actions": list(no_rtc_model_actions.shape),
            "rtc_model_actions": list(rtc_model_actions.shape),
        },
        "timing_ms": {
            "initial_client": initial_client_ms,
            "no_rtc_client": no_rtc_client_ms,
            "rtc_client": rtc_client_ms,
            "initial_policy": _jsonable(initial_response.get("policy_timing", {})),
            "no_rtc_policy": _jsonable(no_rtc_response.get("policy_timing", {})),
            "rtc_policy": _jsonable(rtc_response.get("policy_timing", {})),
            "initial_server": _jsonable(initial_response.get("server_timing", {})),
            "no_rtc_server": _jsonable(no_rtc_response.get("server_timing", {})),
            "rtc_server": _jsonable(rtc_response.get("server_timing", {})),
        },
        "diff": {
            "compared_responses": "rtc_after_execution - no_rtc_after_execution",
            "actions_l2": float(np.linalg.norm(rtc_actions - no_rtc_actions)),
            "actions_max_abs": float(np.max(np.abs(rtc_actions - no_rtc_actions))),
            "model_actions_l2": float(np.linalg.norm(rtc_model_actions - no_rtc_model_actions)),
            "model_actions_max_abs": float(np.max(np.abs(rtc_model_actions - no_rtc_model_actions))),
        },
    }


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
