import collections
import dataclasses
import json
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
from typing import Tuple

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 3072  # resolution used to render training data

SUITE_MAX_STEPS = {
    "libero_organize": 400,
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    num_steps_wait: int = 10
    num_trials_per_task: int = 1
    num_tasks_per_suite: int = 6
    video_out_path: str = "data/libero/videos"
    seed: int = 7


def run_single_suite(
    task_suite_name: str,
    client: _websocket_client_policy.WebsocketClientPolicy,
    model_name: str,
    args: Args,
) -> Tuple[int, int]:
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    max_steps = SUITE_MAX_STEPS.get(task_suite_name)
    if max_steps is None:
        raise ValueError(f"Unknown task suite: {task_suite_name}")

    suite_episodes, suite_successes = 0, 0
    num_tasks_to_run = min(args.num_tasks_per_suite, num_tasks_in_suite)
    for task_id in tqdm.tqdm(range(num_tasks_to_run), desc=f"{task_suite_name} tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task), desc="episodes", leave=False):
            logging.info(f"\nTask: {task_description}")

            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            replay_images = []
            replay_wrist_images = []
            trajectory_states = []
            trajectory_actions = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img_raw = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img_raw = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                    replay_images.append(image_tools.convert_to_uint8(img_raw))
                    replay_wrist_images.append(image_tools.convert_to_uint8(wrist_img_raw))

                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img_raw, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img_raw, args.resize_size, args.resize_size)
                    )

                    if not action_plan:
                        state = np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        )
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": state,
                            "prompt": str(task_description),
                        }

                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    current_state = np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    )

                    action = action_plan.popleft()

                    trajectory_states.append(current_state.tolist())
                    trajectory_actions.append(action.tolist())

                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        suite_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            suite_episodes += 1

            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            video_dir = pathlib.Path(args.video_out_path) / f"{task_suite_name}_{model_name}"
            video_dir.mkdir(parents=True, exist_ok=True)
            imageio.mimwrite(
                video_dir / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )
            imageio.mimwrite(
                video_dir / f"rollout_{task_segment}_{suffix}_wrist.mp4",
                [np.asarray(x) for x in replay_wrist_images],
                fps=10,
            )

            trajectory_data = {
                "trajectory": [
                    {"state": s, "action": a}
                    for s, a in zip(trajectory_states, trajectory_actions)
                ],
                "task_description": task_description,
                "success": bool(done),
                "num_steps": len(trajectory_states),
            }
            with open(video_dir / f"rollout_{task_segment}_{suffix}.txt", "w") as f:
                json.dump(trajectory_data, f, indent=2)

            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {suite_episodes}")
            logging.info(f"# successes: {suite_successes} ({suite_successes / suite_episodes * 100:.1f}%)")

        logging.info(f"Task success rate: {float(task_successes) / float(task_episodes)}")

    logging.info(f"Suite {task_suite_name} success rate: {float(suite_successes) / float(suite_episodes)}")
    return suite_episodes, suite_successes


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    server_metadata = client.get_server_metadata()
    model_name = server_metadata.get("config_name", server_metadata.get("model_name", "unknown_model"))

    suites_to_run = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

    total_episodes, total_successes = 0, 0
    for suite_name in suites_to_run:
        logging.info(f"\n{'='*50}")
        logging.info(f"Starting suite: {suite_name}")
        logging.info(f"{'='*50}")
        episodes, successes = run_single_suite(suite_name, client, model_name, args)
        total_episodes += episodes
        total_successes += successes

    logging.info(f"\n{'='*50}")
    logging.info(f"FINAL RESULTS")
    logging.info(f"{'='*50}")
    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")
    logging.info(f"Total successes: {total_successes}")


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
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
    tyro.cli(eval_libero)
