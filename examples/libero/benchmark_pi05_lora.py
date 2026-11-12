"""Run the production JAX trainer for 100 steps, recording local resource metrics."""

import argparse
import dataclasses
import datetime
import importlib.util
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import traceback
import types


def dump(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path("/home/czl/openpi"))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        raise FileExistsError("Use a new benchmark output directory")
    os.chdir(args.repository)
    sys.path.insert(0, str(args.repository / "src"))
    logging.basicConfig(level=logging.INFO)

    # Load Torch/Arrow before JAX to avoid a native allocator initialization
    # crash in the server's existing binary environment.
    print("Loading Torch/Arrow, then JAX and training modules", flush=True)
    importlib.import_module("torch")
    importlib.import_module("pyarrow")
    import jax

    from openpi.models import pi0_config
    from openpi.training import checkpoints
    from openpi.training import config
    from openpi.training import optimizer

    model = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        max_token_len=48,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )
    cfg = dataclasses.replace(
        config.get_config("pi05_libero"),
        name="pi05_libero_reconstructed_nostate_6gpu_bs96",
        exp_name=output.name,
        model=model,
        freeze_filter=model.get_freeze_filter(),
        ema_decay=None,
        data=config.LeRobotLiberoDataConfig(
            repo_id="caozilin/franka_100_nominal",
            lerobot_dataset_root="/mnt/SharedData/czl/lerobot/caozilin/franka_100_nominal",
            hf_datasets_cache_dir="/mnt/SharedData/czl/hf_datasets/caozilin/franka_100_nominal",
            assets=config.AssetsConfig(assets_dir="/mnt/SharedData/czl/assets/pi05_franka_mujoco_state_action7"),
            base_config=config.DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=96,
        num_workers=8,
        num_train_steps=100,
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=333, peak_lr=5e-5, decay_steps=10_000, decay_lr=5e-6),
        optimizer=optimizer.AdamW(clip_gradient_norm=1.0),
        fsdp_devices=2,
        log_interval=10,
        save_interval=2_000,
        keep_period=2_000,
        swanlab_enabled=False,
        swanlab_mode="disabled",
        pytorch_weight_path=None,
        checkpoint_base_dir=str(output / "checkpoints"),
        resume=False,
        overwrite=False,
    )
    dump(output / "actual_config.json", dataclasses.asdict(cfg))
    assert jax.device_count() == 6, jax.devices()
    assert cfg.data.create(cfg.assets_dirs, cfg.model).norm_stats is not None
    dump(output / "process.json", {"pid": os.getpid(), "command": sys.argv})

    phase = "initialization"
    started = time.perf_counter()
    stop = threading.Event()
    peaks = {}
    intervals = []

    def status(**kwargs):
        dump(
            output / "status.json",
            {
                "phase": phase,
                "pid": os.getpid(),
                "elapsed_seconds": time.perf_counter() - started,
                "updated_at": datetime.datetime.now().astimezone().isoformat(),
                **kwargs,
            },
        )

    def monitor():
        selected = set(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        with (output / "gpu_samples.jsonl").open("a", buffering=1) as file:
            while not stop.is_set():
                try:
                    rows = subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw",
                            "--format=csv,noheader,nounits",
                        ],
                        text=True,
                        timeout=10,
                    )
                    devices = []
                    for line in rows.splitlines():
                        values = [value.strip() for value in line.split(",")]
                        if values[0] not in selected:
                            continue
                        record = {
                            "index": int(values[0]),
                            "used_mib": float(values[1]),
                            "total_mib": float(values[2]),
                            "utilization_percent": float(values[3]),
                            "power_watts": float(values[4]),
                        }
                        devices.append(record)
                        peaks[values[0]] = max(peaks.get(values[0], 0), record["used_mib"])
                    file.write(
                        json.dumps({"elapsed": time.perf_counter() - started, "phase": phase, "devices": devices})
                        + "\n"
                    )
                except Exception as error:
                    file.write(json.dumps({"monitor_error": repr(error)}) + "\n")
                stop.wait(1)

    def allocator_stats():
        return {str(device.id): device.memory_stats() for device in jax.local_devices()}

    class Progress:
        def __init__(self, iterable, **kwargs):
            del kwargs
            self.iterable = iterable
            self.last = time.perf_counter()

        def __iter__(self):
            return iter(self.iterable)

        def write(self, message):
            print(message, flush=True)
            match = re.match(r"Step (\d+):", message)
            if match is None:
                return
            now = time.perf_counter()
            step = int(match[1])
            entry = {
                "completed_step": step,
                "seconds": now - self.last,
                "seconds_per_step": (now - self.last) / 10,
                "examples_per_second": 960 / (now - self.last),
                "metrics": message,
                "allocator": allocator_stats(),
            }
            intervals.append(entry)
            with (output / "timings.jsonl").open("a") as file:
                file.write(json.dumps(entry) + "\n")
            self.last = time.perf_counter()
            status(completed_steps=step, last_interval=entry)

    spec = importlib.util.spec_from_file_location("benchmark_train_entry", args.repository / "scripts/train.py")
    train = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = train
    spec.loader.exec_module(train)
    original_init = train.init_train_state

    def measured_init(*init_args, **init_kwargs):
        nonlocal phase
        phase = "model_and_optimizer_initialization"
        status()
        before = time.perf_counter()
        result = original_init(*init_args, **init_kwargs)
        jax.block_until_ready(result[0])
        dump(output / "initialization.json", {"seconds": time.perf_counter() - before, "allocator": allocator_stats()})
        phase = "training"
        status()
        return result

    train.init_train_state = measured_init
    train.tqdm = types.SimpleNamespace(tqdm=Progress)
    # A 100-step throughput probe ends before the configured first save at step 2000.
    # Suppress only the trainer's forced final checkpoint; keep the actual train step unchanged.
    checkpoints.save_state = lambda *_args, **_kwargs: logging.info("Performance probe: checkpoint omitted")
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    status()
    try:
        train.main(cfg)
        assert len(intervals) == 10
        assert intervals[-1]["completed_step"] == 100
        steady_seconds = sum(entry["seconds"] for entry in intervals[1:])
        phase = "complete"
        result = {
            "complete": True,
            "exit_code": 0,
            "steps": 100,
            "global_batch": 96,
            "per_gpu_batch": 16,
            "fsdp_devices": cfg.fsdp_devices,
            "devices": 6,
            "warmup_steps": 333,
            "schedule_steps": 10_000,
            "swanlab_enabled": False,
            "first_10_steps_seconds": intervals[0]["seconds"],
            "steady_steps": 90,
            "steady_seconds": steady_seconds,
            "steady_seconds_per_step": steady_seconds / 90,
            "steady_examples_per_second": 90 * 96 / steady_seconds,
            "peak_nvidia_mib": peaks.copy(),
            "allocator": allocator_stats(),
            "elapsed_seconds": time.perf_counter() - started,
            "checkpoint_saved": False,
        }
        dump(output / "result.json", result)
        status(completed_steps=100)
        print(json.dumps(result), flush=True)
    except BaseException as error:
        phase = "failed"
        dump(
            output / "result.json",
            {
                "complete": False,
                "exit_code": 1,
                "error": repr(error),
                "peak_nvidia_mib": peaks.copy(),
                "completed_intervals": len(intervals),
            },
        )
        status(error=repr(error))
        traceback.print_exc()
        raise
    finally:
        stop.set()
        thread.join(timeout=12)


if __name__ == "__main__":
    main()
