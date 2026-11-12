"""Normalize verified reconstructed LIBERO, then run the authorized 6 x 12 LoRA job."""

import argparse
import dataclasses
import datetime
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

REPOSITORY = Path("/home/czl/openpi")
DATA_HOME = Path("/mnt/SharedData/czl")
DATASET = DATA_HOME / "lerobot/caozilin/libero_reconstructed_full_20261101"
VERIFICATION = DATA_HOME / "runs/libero_reconstructed_full_20261101/lerobot_upload_verification.json"


def dump(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def child(run, stage):
    # Keep the server's existing dependencies; avoid the observed allocator import crash.
    importlib.import_module("torch")
    importlib.import_module("pyarrow")

    from openpi.models import pi0_config
    from openpi.training import config
    from openpi.training import optimizer

    spec = json.loads((run / "training_request.json").read_text())["config"]
    model = pi0_config.Pi0Config(**spec["model"])
    fields = (
        "name",
        "seed",
        "batch_size",
        "num_workers",
        "num_train_steps",
        "ema_decay",
        "fsdp_devices",
        "log_interval",
        "save_interval",
        "keep_period",
        "swanlab_enabled",
        "swanlab_mode",
    )
    cfg = dataclasses.replace(
        config.get_config("pi05_libero"),
        **{key: spec[key] for key in fields},
        exp_name="Pi05_libero_reconstructed_NoState_LoRA_6gpu_bs72_10k_20261101",
        model=model,
        freeze_filter=model.get_freeze_filter(),
        data=config.LeRobotLiberoDataConfig(
            repo_id=spec["data"]["repo_id"],
            lerobot_dataset_root=str(DATASET),
            hf_datasets_cache_dir=str(DATA_HOME / "hf_datasets" / spec["data"]["repo_id"]),
            base_config=config.DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        lr_schedule=optimizer.CosineDecaySchedule(**spec["lr_schedule"]),
        optimizer=optimizer.AdamW(**spec["optimizer"]),
        assets_base_dir=str(DATA_HOME / "assets"),
        checkpoint_base_dir=str(DATA_HOME / "checkpoints"),
        swanlab_log_dir=str(run / "swanlog"),
        pytorch_weight_path=None,
        resume=False,
        overwrite=False,
    )
    assert cfg.batch_size == 72
    assert cfg.num_train_steps == cfg.lr_schedule.decay_steps == 10_000
    assert cfg.model.pi05
    assert not cfg.model.discrete_state_input
    assert cfg.swanlab_enabled and cfg.swanlab_mode == "online"
    dump(run / "actual_config.json", dataclasses.asdict(cfg))
    if stage == "train":
        import jax

        assert jax.device_count() == 6
    else:
        original_get_config = config.get_config
        config.get_config = lambda name: cfg if name == cfg.name else original_get_config(name)
    # Spawned DataLoader workers must be able to import transforms by their real module name.
    sys.path.insert(0, str(REPOSITORY))
    module = importlib.import_module("scripts.train" if stage == "train" else "scripts.compute_norm_stats")
    if stage == "norm":
        module.main(cfg.name)
    else:
        import logging

        logging.basicConfig(level=logging.INFO)
        module.main(cfg)



def normalization_receipt(run, request, verification):
    spec = request["config"]
    inputs = {key: spec[key] for key in ("model", "data", "batch_size", "num_workers", "seed")}
    inputs["transfer_manifest_sha256"] = verification["transfer_manifest_sha256"]
    stats_path = DATA_HOME / "assets" / spec["name"] / spec["data"]["repo_id"] / "norm_stats.json"
    stats = json.loads(stats_path.read_text())["norm_stats"]
    for name, width in (("state", 8), ("actions", 7)):
        assert all(len(values) == width and all(math.isfinite(x) for x in values)
                   for values in stats[name].values())
        assert all(x >= 0 for x in stats[name]["std"])
    return {
        "complete": True,
        "inputs_sha256": hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(),
        "stats_path": str(stats_path),
        "stats_sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest(),
    }


def pipeline(run):
    lock = (run / "training.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    dump(run / "training_process.json", {"pid": os.getpid(), "command": sys.argv})

    def status(stage, **kwargs):
        dump(
            run / "training_state.json",
            {
                "stage": stage,
                "pid": os.getpid(),
                "updated_at": datetime.datetime.now().astimezone().isoformat(),
                **kwargs,
            },
        )

    try:
        request = json.loads((run / "training_request.json").read_text())
        assert request["server_script_sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        assert request["authorized"]
        assert request["cancelled_all_benchmarks"]
        verification = json.loads(VERIFICATION.read_text())
        assert verification["complete"]
        assert verification["all_uploaded_sha256_equal"]
        assert verification["portable_lerobot_reader_passed"]
        audit = json.loads((DATASET / "meta/reconstruction_audit.json").read_text())
        assert audit["complete"]
        assert audit["all_camera_pixels_equal"]
        assert audit["all_state_action_equal"]
        assert (audit["episodes"], audit["frames"], audit["tasks"], audit["fps"]) == (1875, 312374, 40, 20)
        assert shutil.disk_usage(DATA_HOME).free > 150 * 1024**3, "Need room for Arrow cache and five checkpoints"
        cache = DATA_HOME / "hf_datasets" / request["config"]["data"]["repo_id"]
        cache.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(
            SWANLAB_MODE="cloud",
            HF_HUB_OFFLINE="1",
            HF_DATASETS_OFFLINE="1",
            HF_DATASETS_CACHE=str(cache),
            HF_LEROBOT_HOME=str(DATA_HOME / "lerobot"),
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            ARROW_DEFAULT_MEMORY_POOL="system",
            ARROW_JEMALLOC_CONF="background_thread:false",
            MALLOC_CONF="background_thread:false",
            XLA_PYTHON_CLIENT_PREALLOCATE="false",
            XLA_PYTHON_CLIENT_MEM_FRACTION="0.95",
            PYTHONUNBUFFERED="1",
        )
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--run-dir", str(run)]
        receipt_path = run / "normalization_completion.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            assert receipt == normalization_receipt(run, request, verification), "Normalization inputs or file changed"
            status("normalization_verified_and_reused", stats_sha256=receipt["stats_sha256"])
        else:
            status("computing_normalization")
            cpu_env = {**env, "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu"}
            with (run / "normalization.log").open("a") as log:
                subprocess.run([*command, "--stage", "norm"], env=cpu_env, stdout=log, stderr=subprocess.STDOUT, check=True)
            dump(receipt_path, normalization_receipt(run, request, verification))
        selected = {0, 1, 2, 3, 4, 6}
        while True:
            rows = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                text=True,
            )
            busy = [
                line
                for line in rows.splitlines()
                if int(line.split(",")[0]) in selected
                and (float(line.split(",")[1]) > 1024 or float(line.split(",")[2]) > 5)
            ]
            if not busy:
                break
            status("waiting_for_idle_gpus", busy=busy)
            time.sleep(30)
        status("training", expected_steps=10_000, global_batch=72, per_gpu_batch=12)
        gpu_env = {**env, "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,6"}
        gpu_env.pop("JAX_PLATFORMS", None)
        with (run / "train.log").open("a") as log:
            subprocess.run(
                [*command, "--stage", "train"], env=gpu_env, stdout=log, stderr=subprocess.STDOUT, check=True
            )
        cfg = json.loads((run / "actual_config.json").read_text())
        checkpoint_dir = Path(cfg["checkpoint_base_dir"]) / cfg["name"] / cfg["exp_name"]
        assert all((checkpoint_dir / str(step)).is_dir() for step in range(2000, 10001, 2000))
        dump(
            run / "training_completion.json",
            {
                "complete": True,
                "exit_code": 0,
                "steps": 10_000,
                "global_batch": 72,
                "per_gpu_batch": 12,
                "checkpoint_dir": str(checkpoint_dir),
                "swanlab_enabled": True,
                "swanlab_mode": "online",
                "swanlab_log_dir": str(run / "swanlog"),
            },
        )
        status("complete", steps=10_000, checkpoint_dir=str(checkpoint_dir))
    except BaseException as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("pipeline", "norm", "train"), default="pipeline")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(REPOSITORY)
    sys.path.insert(0, str(REPOSITORY / "src"))
    try:
        if args.stage == "pipeline":
            pipeline(run_dir)
        else:
            child(run_dir, args.stage)
    except BaseException:
        traceback.print_exc()
        sys.exit(1)
