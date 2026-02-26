import collections
import dataclasses
import datetime
import json
import logging
import math
import pathlib
from typing import Callable, Dict, List, Optional, Tuple

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 224

DEFAULT_TASK_CONFIGS = [
    ("libero_spatial", 5),
    ("libero_object", 5),
    ("libero_goal", 5),
    ("libero_10", 5),
    ("libero_90", 3),
]

MAX_STEPS_MAP = {
    "libero_pi": 300,
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
    video_out_path: str = "data/libero/videos"
    eval_results_path: str = "data/libero/eval_results.json"
    seed: int = 7
    task_configs: List[Tuple[str, int]] = dataclasses.field(default_factory=lambda: DEFAULT_TASK_CONFIGS)
    resume: bool = False
    save_interval: int = 100


def get_max_steps(suite_name: str) -> int:
    if suite_name not in MAX_STEPS_MAP:
        raise ValueError(f"Unknown task suite: {suite_name}")
    return MAX_STEPS_MAP[suite_name]


def load_eval_results(path: pathlib.Path) -> dict:
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {"evaluations": []}


def save_eval_results(path: pathlib.Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def find_model_eval(evaluations: list, model_name: str) -> Optional[dict]:
    for eval_record in evaluations:
        if eval_record["model_name"] == model_name:
            return eval_record
    return None


def get_existing_subtask_counts(
    eval_record: dict, suite_name: str
) -> Dict[int, int]:
    if not eval_record:
        return {}

    for suite in eval_record.get("task_suites", []):
        if suite["suite_name"] == suite_name:
            return {subtask["task_id"]: subtask["total_count"] for subtask in suite.get("subtasks", [])}
    return {}


def clear_suite_data(eval_record: dict, suite_name: str) -> None:
    if not eval_record:
        return

    for suite in eval_record.get("task_suites", []):
        if suite["suite_name"] == suite_name:
            suite["subtasks"] = []
            suite["suite_summary"] = {
                "success_rate": 0.0,
                "total_successes": 0,
                "total_episodes": 0,
            }
            return


def run_single_suite(
    suite_name: str,
    num_trials_per_task: int,
    client: _websocket_client_policy.WebsocketClientPolicy,
    args: Args,
    model_name: str,
    existing_counts: Dict[int, int],
    on_episode_done: Optional[Callable] = None,
) -> dict:
    np.random.seed(args.seed)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    max_steps = get_max_steps(suite_name)

    logging.info(f"Task suite: {suite_name}")

    subtasks = []
    suite_total_episodes = 0
    suite_total_successes = 0

    for task_id in tqdm.tqdm(range(num_tasks_in_suite), desc=f"Tasks in {suite_name}"):
        existing_count = existing_counts.get(task_id, 0)

        if existing_count >= num_trials_per_task:
            logging.info(f"Skipping task {task_id}: already has {existing_count} episodes (target: {num_trials_per_task})")
            continue

        remaining_trials = num_trials_per_task - existing_count
        logging.info(f"Task {task_id}: running {remaining_trials} more episodes (existing: {existing_count}, target: {num_trials_per_task})")

        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_successes = 0
        task_episodes = 0

        for episode_idx in range(remaining_trials):
            logging.info(f"\nTask: {task_description}")

            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[existing_count + episode_idx])

            t = 0
            replay_images = []
            done = False

            logging.info(f"Starting episode {existing_count + episode_idx + 1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img_raw = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img_raw = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                    replay_images.append(image_tools.convert_to_uint8(img_raw))

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

                    action = action_plan.popleft()
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1

            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            video_dir = pathlib.Path(args.video_out_path) / f"{suite_name}_{model_name}"
            video_dir.mkdir(parents=True, exist_ok=True)
            imageio.mimwrite(
                video_dir / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            logging.info(f"Success: {done}")

            if on_episode_done is not None:
                on_episode_done()

        subtasks.append({
            "task_id": task_id,
            "task_description": task_description,
            "success_count": task_successes,
            "total_count": task_episodes,
        })

        suite_total_episodes += task_episodes
        suite_total_successes += task_successes

        logging.info(f"Task '{task_description}' success rate: {task_successes}/{task_episodes}")

    suite_summary = {
        "success_rate": suite_total_successes / suite_total_episodes if suite_total_episodes > 0 else 0.0,
        "total_successes": suite_total_successes,
        "total_episodes": suite_total_episodes,
    }

    return {
        "suite_name": suite_name,
        "num_trials_per_task": num_trials_per_task,
        "subtasks": subtasks,
        "suite_summary": suite_summary,
    }


def merge_suite_result(task_suites: list, suite_result: dict) -> None:
    existing_suite = None
    for suite in task_suites:
        if suite["suite_name"] == suite_result["suite_name"]:
            existing_suite = suite
            break

    if existing_suite is not None:
        for new_subtask in suite_result["subtasks"]:
            found = False
            for old_subtask in existing_suite["subtasks"]:
                if old_subtask["task_id"] == new_subtask["task_id"]:
                    old_subtask["success_count"] += new_subtask["success_count"]
                    old_subtask["total_count"] += new_subtask["total_count"]
                    found = True
                    break
            if not found:
                existing_suite["subtasks"].append(new_subtask)

        existing_suite["suite_summary"]["total_successes"] += suite_result["suite_summary"]["total_successes"]
        existing_suite["suite_summary"]["total_episodes"] += suite_result["suite_summary"]["total_episodes"]
        existing_suite["suite_summary"]["success_rate"] = (
            existing_suite["suite_summary"]["total_successes"] / existing_suite["suite_summary"]["total_episodes"]
            if existing_suite["suite_summary"]["total_episodes"] > 0 else 0.0
        )
    else:
        task_suites.append(suite_result)


def eval_libero(args: Args) -> None:
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    server_metadata = client.get_server_metadata()
    model_name = server_metadata.get("config_name", server_metadata.get("model_name", "unknown_model"))

    logging.info(f"Model: {model_name}")
    logging.info(f"Task configs: {args.task_configs}")
    logging.info(f"Resume mode: {args.resume}")

    eval_results_path = pathlib.Path(args.eval_results_path)
    eval_results = load_eval_results(eval_results_path)

    model_eval = find_model_eval(eval_results["evaluations"], model_name)

    if model_eval is None:
        model_eval = {
            "model_name": model_name,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            "task_suites": [],
            "overall_summary": {"success_rate": 0.0, "total_successes": 0, "total_episodes": 0},
        }
        eval_results["evaluations"].append(model_eval)
    else:
        model_eval["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if args.resume:
        for suite_name, _ in args.task_configs:
            clear_suite_data(model_eval, suite_name)

    task_suites = model_eval["task_suites"]
    overall_total_episodes = model_eval["overall_summary"]["total_episodes"]
    overall_total_successes = model_eval["overall_summary"]["total_successes"]
    episode_counter = 0

    def save_if_needed():
        nonlocal episode_counter
        episode_counter += 1
        if episode_counter % args.save_interval == 0:
            model_eval["overall_summary"] = {
                "success_rate": overall_total_successes / overall_total_episodes if overall_total_episodes > 0 else 0.0,
                "total_successes": overall_total_successes,
                "total_episodes": overall_total_episodes,
            }
            save_eval_results(eval_results_path, eval_results)
            logging.info(f"Auto-saved results after {episode_counter} episodes")

    for suite_name, num_trials in args.task_configs:
        existing_counts = {} if args.resume else get_existing_subtask_counts(model_eval, suite_name)
        suite_result = run_single_suite(
            suite_name, num_trials, client, args, model_name, existing_counts,
            on_episode_done=save_if_needed,
        )

        if suite_result["subtasks"]:
            merge_suite_result(task_suites, suite_result)
            overall_total_episodes += suite_result["suite_summary"]["total_episodes"]
            overall_total_successes += suite_result["suite_summary"]["total_successes"]

        save_eval_results(eval_results_path, eval_results)

    model_eval["overall_summary"] = {
        "success_rate": overall_total_successes / overall_total_episodes if overall_total_episodes > 0 else 0.0,
        "total_successes": overall_total_successes,
        "total_episodes": overall_total_episodes,
    }

    save_eval_results(eval_results_path, eval_results)

    logging.info(f"\n{'=' * 50}")
    logging.info(f"Evaluation completed for model: {model_name}")
    logging.info(f"Overall success rate: {model_eval['overall_summary']['success_rate']:.2%}")
    logging.info(f"Total episodes: {overall_total_episodes}")
    logging.info(f"Total successes: {overall_total_successes}")
    logging.info(f"Results saved to: {eval_results_path}")


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
