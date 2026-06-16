import dataclasses
import enum
import logging
import socket

import tyro

from openpi.models import rtc as _rtc
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.LIBERO

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Enable server-side Real-Time Chunking for JAX pi0/pi0.5 flow policies.
    rtc_enabled: bool = False
    rtc_execution_horizon: int = 5
    rtc_max_guidance_weight: float = 3.0
    rtc_prefix_attention_schedule: str = "linear"
    # Number of model-action dimensions guided by RTC. For Franka delta pose, keep this at 6 to exclude gripper sign.
    rtc_guidance_dims: int = 6

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(
    env: EnvMode, *, default_prompt: str | None = None, rtc_config: dict | None = None
) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config),
            checkpoint.dir,
            default_prompt=default_prompt,
            rtc_config=rtc_config,
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def extract_model_name_from_path(checkpoint_dir: str) -> str:
    """Extract model name from checkpoint directory path.

    从检查点目录路径中提取模型名称。

    Args:
        checkpoint_dir: Checkpoint directory path (e.g., "gs://openpi-assets/checkpoints/pi0_libero").
                        检查点目录路径(例如, "gs://openpi-assets/checkpoints/pi0_libero")。

    Returns:
        Model name extracted from the last component of the path.
        从路径最后一个组件提取的模型名称。
    """
    return checkpoint_dir.rstrip("/").split("/")[-1]


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    rtc_config = {
        "enabled": args.rtc_enabled,
        "execution_horizon": args.rtc_execution_horizon,
        "max_guidance_weight": args.rtc_max_guidance_weight,
        "prefix_attention_schedule": int(_rtc.schedule_from_string(args.rtc_prefix_attention_schedule)),
        "guidance_dims": args.rtc_guidance_dims,
    }
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config),
                args.policy.dir,
                default_prompt=args.default_prompt,
                rtc_config=rtc_config,
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt, rtc_config=rtc_config)


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    # Build model_name as "{config}/{step}" so clients can identify both
    # the training config and the checkpoint step without extra arguments.
    # 构建 model_name 为 "{config}/{step}", 客户端可直接识别训练配置和步骤号。
    match args.policy:
        case Checkpoint():
            step = extract_model_name_from_path(args.policy.dir)
            policy_metadata["model_name"] = f"{args.policy.config}/{step}"
        case Default():
            if default_checkpoint := DEFAULT_CHECKPOINT.get(args.env):
                step = extract_model_name_from_path(default_checkpoint.dir)
                policy_metadata["model_name"] = f"{default_checkpoint.config}/{step}"
            else:
                policy_metadata["model_name"] = "unknown"
    logging.info("Model name set to: %s", policy_metadata["model_name"])
    policy_metadata["rtc"] = {
        "enabled": args.rtc_enabled,
        "execution_horizon": args.rtc_execution_horizon,
        "max_guidance_weight": args.rtc_max_guidance_weight,
        "prefix_attention_schedule": args.rtc_prefix_attention_schedule,
        "guidance_dims": args.rtc_guidance_dims,
    }

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
