"""Serve TaskTol datasets and the local browser visualizer."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import json
import math
import mimetypes
from pathlib import Path
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse
import webbrowser

import numpy as np

from ...geometry import canonical_rotation_vector, rotation_matrix
from ...tolerance.annotations import box_tolerance_frame
from ..tolerance_labels import DATASET_SCHEMA_VERSION, load_rotation_tolerance


STATIC_ROOT = Path(__file__).with_name("static")
MEDIA_NAMES = frozenset(("main_rgb.mp4", "wrist_rgb.mp4"))

_GEOMETRY_OBJECT_LABELS = {
    "Upright Box": "Upright rectangular block",
    "Lying Box": "Lying rectangular block",
    "Cube": "Cube",
    "Ball": "Ball",
    "Upright Cylinder": "Upright cylinder",
    "Lying Cylinder": "Lying cylinder",
}


def dataset_display_name(scene: str, task_id: str, task_name: str) -> str:
    """Build a concise, unambiguous title for the viewer sidebar."""
    for prefix, target in (
        ("Geometry Plate - ", "tray"),
        ("Geometry Region - ", "white region"),
    ):
        if scene.startswith(prefix):
            variant = scene.removeprefix(prefix)
            object_label = _GEOMETRY_OBJECT_LABELS.get(variant, variant)
            return f"{object_label} → {target}"
    if scene == "Adjust Cylindrical Bottle":
        return "Cylindrical water bottle → upright"
    if scene == "Adjust Rectangular Bottle":
        return "Rectangular water bottle → upright"
    candidate = task_id if "->" in task_id else task_name
    return candidate.replace("->", "→").strip() or scene


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_safe(value: Any) -> Any:
    """Replace Python's non-standard non-finite JSON numbers recursively."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@lru_cache(maxsize=None)
def _nominal_main_camera(scene: str) -> dict[str, Any]:
    """Return the unrandomized MuJoCo front-camera basis for spatial plots."""
    import mujoco

    from ...simulation.loader import load_scene_model

    model = load_scene_model(scene)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front_view")
    if camera_id < 0:
        raise RuntimeError(f"Scene {scene!r} has no front_view camera")
    rotation = data.cam_xmat[camera_id].reshape(3, 3)
    return {
        "position_m": data.cam_xpos[camera_id].tolist(),
        "right_world": rotation[:, 0].tolist(),
        "up_world": rotation[:, 1].tolist(),
        "forward_world": (-rotation[:, 2]).tolist(),
        "source": "unrandomized MuJoCo front_view",
    }


@dataclass(frozen=True)
class DatasetRecord:
    relative_path: str
    metadata: dict[str, Any]
    manifest: dict[str, Any]
    instructions: Any

    def as_index_entry(self) -> dict[str, Any]:
        episodes = self.manifest.get("episodes", [])
        scene = str(self.metadata.get("scene", "Unknown scene"))
        task_id = str(self.metadata.get("task_id", self.relative_path))
        task_name = str(self.metadata.get("task_name", task_id))
        return {
            "source_type": "dataset",
            "relative_path": self.relative_path,
            "scene": scene,
            "robot": self.metadata.get("robot", {}).get("uid", "panda"),
            "task_id": task_id,
            "task_name": task_name,
            "display_name": dataset_display_name(scene, task_id, task_name),
            "frequency_hz": self.metadata.get("frequency_hz", 10.0),
            "episode_count": len(episodes),
            "episodes": episodes,
        }


@dataclass(frozen=True)
class BenchmarkRecord:
    relative_path: str
    summary: dict[str, Any]

    def as_index_entry(self) -> dict[str, Any]:
        task = self.summary.get("task", {})
        episodes = self.summary.get("episodes", [])
        scene = str(task.get("scene", "Unknown scene"))
        task_id = str(task.get("task_id", self.relative_path))
        task_name = str(task.get("task_name", task_id))
        return {
            "source_type": "benchmark",
            "relative_path": self.relative_path,
            "scene": scene,
            "robot": self.summary.get("configuration", {}).get("robot", {}).get("uid", "panda"),
            "task_id": task_id,
            "task_name": task_name,
            "display_name": dataset_display_name(scene, task_id, task_name),
            "frequency_hz": self.summary.get("configuration", {}).get("frequency_hz", 10.0),
            "episode_count": len(episodes),
            "episodes": episodes,
            "run_status": self.summary.get("status", "unknown"),
            "tolerance_mode": self.summary.get("configuration", {}).get("tolerance_mode", "off"),
        }


class DatasetCatalog:
    """Read-only, path-confined access to TaskTol dataset directories."""

    def __init__(self, root: Path, benchmark_root: Path | None = None) -> None:
        self.root = root.expanduser().resolve()
        self.benchmark_root = (
            self.root.parent / "benchmark_results"
            if benchmark_root is None else benchmark_root
        ).expanduser().resolve()
        if not self.root.is_dir() and not self.benchmark_root.is_dir():
            raise FileNotFoundError(
                "Neither dataset nor benchmark root exists: "
                f"{self.root}, {self.benchmark_root}",
            )

    def _source_root(self, source_type: str) -> Path:
        if source_type == "dataset":
            return self.root
        if source_type == "benchmark":
            return self.benchmark_root
        raise ValueError(f"Unknown viewer source type: {source_type}")

    def _resolve(self, relative_path: str, source_type: str = "dataset") -> Path:
        root = self._source_root(source_type)
        candidate = (root / unquote(relative_path)).resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError(f"Path escapes the {source_type} root")
        return candidate

    def _record(self, relative_path: str) -> DatasetRecord:
        directory = self._resolve(relative_path)
        manifest_path = directory / "manifest.json"
        metadata_path = directory / "task_metadata.json"
        if not manifest_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"Not a TaskTol dataset: {relative_path}")
        manifest = _read_json(manifest_path)
        metadata = _read_json(metadata_path)
        instruction_name = manifest.get(
            "instruction_file", metadata.get("instruction_file", "instructions.json")
        )
        instruction_path = directory / str(instruction_name)
        instructions = _read_json(instruction_path) if instruction_path.is_file() else []
        normalized = directory.relative_to(self.root).as_posix()
        return DatasetRecord(normalized, metadata, manifest, instructions)

    def _benchmark_record(self, relative_path: str) -> BenchmarkRecord:
        directory = self._resolve(relative_path, "benchmark")
        summary_path = directory / "benchmark.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Not a TaskTol benchmark run: {relative_path}")
        normalized = directory.relative_to(self.benchmark_root).as_posix()
        return BenchmarkRecord(normalized, _read_json(summary_path))

    def list_datasets(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for manifest_path in sorted(self.root.rglob("manifest.json")):
            if "failed_episodes" in manifest_path.parts:
                continue
            directory = manifest_path.parent
            if not (directory / "task_metadata.json").is_file():
                continue
            relative = directory.relative_to(self.root).as_posix()
            records.append(self._record(relative).as_index_entry())
        if self.benchmark_root.is_dir():
            for summary_path in sorted(self.benchmark_root.rglob("benchmark.json")):
                directory = summary_path.parent
                relative = directory.relative_to(self.benchmark_root).as_posix()
                records.append(self._benchmark_record(relative).as_index_entry())
        return records

    def dataset_details(
        self, relative_path: str, source_type: str = "dataset",
    ) -> dict[str, Any]:
        if source_type == "benchmark":
            record = self._benchmark_record(relative_path)
            return {
                "source_type": source_type,
                "relative_path": record.relative_path,
                "summary": record.summary,
            }
        record = self._record(relative_path)
        return {
            "source_type": source_type,
            "relative_path": record.relative_path,
            "metadata": record.metadata,
            "manifest": record.manifest,
            "instructions": record.instructions,
        }

    @staticmethod
    def _normalize_benchmark_episode(
        payload: dict[str, Any], record: BenchmarkRecord,
    ) -> dict[str, Any]:
        episode = payload.get("episode", {})
        frames = payload.get("rollout", [])
        oracle = episode.get("oracle_tolerance")
        annotations: list[dict[str, Any]] = []
        if isinstance(oracle, dict):
            phase_sources = (
                ("pregrasp", "pregrasp"),
                ("grasp", "pregrasp"),
                ("postgrasp", "postgrasp"),
                ("release", "postgrasp"),
            )
            oracle_subtasks = oracle.get("subtasks")
            sources = (
                oracle_subtasks
                if isinstance(oracle_subtasks, list) and oracle_subtasks
                else [oracle]
            )
            for subtask_index, source in enumerate(sources):
                if not isinstance(source, dict):
                    continue
                stable = {
                    name: source[name]
                    for name in ("pregrasp", "postgrasp")
                    if isinstance(source.get(name), dict)
                }
                for phase_index, (name, source_name) in enumerate(phase_sources):
                    stage = stable.get(source_name)
                    if stage is None:
                        continue
                    annotations.append({
                        "id": 4 * subtask_index + phase_index,
                        "subtask_index": subtask_index,
                        "source_phase": name,
                        "stage_target_pose": stage.get("stage_target_pose", {}),
                        "viewer_tolerance_rotation_matrix": box_tolerance_frame(rotation_matrix(
                            stage.get(
                                "stage_target_pose", {},
                            ).get(
                                "rotation_vector_base_rad",
                                [0.0, 0.0, 0.0],
                            )
                        )).tolist(),
                        "rotation_tolerance_rad": (
                            [0.0] * 6
                            if name in ("grasp", "release")
                            else stage.get("rotation_tolerance_rad", [0.0] * 6)
                        ),
                        "rotation_tolerance_profile": name,
                    })

        # Current benchmark rollouts store stage truth and policy prediction
        # per frame rather than the episode-level oracle. Build a
        # viewer-only annotation for every frame so the dataset inspector and
        # timeline can render changing model/EMA parameters without flattening
        # them into one stage-wide value.
        truth_by_stage: dict[tuple[int, str], dict[str, Any]] = {}
        for frame in frames:
            truth = frame.get("task_tolerance_ground_truth")
            if not isinstance(truth, dict):
                continue
            phase = str(truth.get("phase", frame.get("phase", "pregrasp")))
            subtask_index = max(0, int(truth.get(
                "subtask_index", frame.get("subtask_index", 0),
            )))
            truth_by_stage[(subtask_index, phase)] = truth

        phase_index_by_name = {
            "pregrasp": 0, "grasp": 1,
            "postgrasp": 2, "release": 3,
        }
        truth_phase_by_name = {
            "pregrasp": "pregrasp", "grasp": "pregrasp",
            "postgrasp": "postgrasp", "release": "postgrasp",
        }

        def matrix_value(value: Any) -> np.ndarray | None:
            try:
                matrix = np.asarray(value, dtype=float)
            except (TypeError, ValueError):
                return None
            if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
                return None
            return matrix

        def rollout_annotation(frame: dict[str, Any]) -> dict[str, Any] | None:
            phase = str(frame.get("phase", "pregrasp"))
            subtask_index = max(0, int(frame.get("subtask_index", 0)))
            truth_phase = truth_phase_by_name.get(phase, phase)
            truth = truth_by_stage.get((subtask_index, truth_phase))
            truth_pose = (
                truth.get("stage_target_pose", {})
                if isinstance(truth, dict) else {}
            )
            runtime = frame.get("tolerance")
            runtime = runtime if isinstance(runtime, dict) else {}
            prediction = frame.get("policy_task_tolerance_prediction")
            prediction = prediction if isinstance(prediction, dict) else {}
            descriptor = prediction.get("descriptor_filter")
            descriptor = descriptor if isinstance(descriptor, dict) else {}

            target_matrix = matrix_value(
                runtime.get("stage_target_rotation_matrix")
            )
            target_source = runtime.get("parameter_source")
            if target_matrix is None:
                target_matrix = matrix_value(truth_pose.get("rotation_matrix"))
                if target_matrix is not None:
                    target_source = "seeded_preset_ground_truth"
            if target_matrix is None:
                rotvec = truth_pose.get("rotation_vector_base_rad")
                try:
                    candidate = np.asarray(rotvec, dtype=float)
                except (TypeError, ValueError):
                    candidate = np.empty(0)
                if candidate.shape == (3,) and np.all(np.isfinite(candidate)):
                    target_matrix = rotation_matrix(candidate)
                    target_source = "seeded_preset_ground_truth"
            if target_matrix is None:
                return None

            tolerance_matrix = matrix_value(
                runtime.get("tolerance_frame_rotation_matrix")
            )
            if tolerance_matrix is None:
                tolerance_matrix = box_tolerance_frame(target_matrix)

            bounds = runtime.get("rotation_tolerance_rad")
            bounds_source = runtime.get("parameter_source")
            respect_runtime_axes = bool(bounds is not None)
            if not isinstance(bounds, list) or len(bounds) != 6:
                bounds = descriptor.get("rotation_tolerance_rad")
                bounds_source = "policy_tolerance_not_applied"
                respect_runtime_axes = False
            if not isinstance(bounds, list) or len(bounds) != 6:
                bounds = (
                    truth.get("rotation_tolerance_rad")
                    if isinstance(truth, dict) else None
                )
                bounds_source = "hard_coded_v1_tolerance_summary_v2"
                respect_runtime_axes = False
            if not isinstance(bounds, list) or len(bounds) != 6:
                bounds = [0.0] * 6
                bounds_source = None

            # Grasp and Release are Specific even though runtime state retains
            # the preceding stable-stage parameters for handoff telemetry.
            if phase in ("grasp", "release"):
                respect_runtime_axes = True

            position = truth_pose.get("position_m")
            if not isinstance(position, list) or len(position) != 3:
                nominal = frame.get("vla_nominal_target_pose", {})
                position = nominal.get("position_m", [0.0, 0.0, 0.0])
            return {
                "source_phase": phase,
                "subtask_index": subtask_index,
                "stage_target_pose": {
                    "position_m": position,
                    "rotation_vector_base_rad": canonical_rotation_vector(
                        target_matrix,
                    ).tolist(),
                },
                "viewer_tolerance_rotation_matrix": tolerance_matrix.tolist(),
                "rotation_tolerance_rad": bounds,
                "rotation_tolerance_profile": phase,
                "viewer_parameter_source": {
                    "stage_target_pose": target_source,
                    "rotation_tolerance": bounds_source,
                },
                "viewer_respect_runtime_ranged_axes": respect_runtime_axes,
            }

        for frame in frames:
            frame["nominal_target_pose"] = frame.get(
                "vla_nominal_target_pose",
                frame.get("controller_target_pose", frame.get("ee_state", {})),
            )
            phase = str(frame.get("phase", "pregrasp"))
            phase_index = phase_index_by_name.get(phase, 0)
            subtask_index = max(0, int(frame.get("subtask_index", 0)))
            dynamic_annotation = rollout_annotation(frame)
            if dynamic_annotation is not None:
                frame["viewer_tolerance_annotation"] = dynamic_annotation
                frame["stage_annotation_id"] = 4 * subtask_index + phase_index
            elif annotations:
                frame["stage_annotation_id"] = min(
                    4 * subtask_index + phase_index,
                    len(annotations) - 1,
                )
        task = record.summary.get("task", {})
        configuration = record.summary.get("configuration", {})
        instruction = str(episode.get("instruction", task.get("task_name", "")))
        payload["trajectory"] = frames
        payload["stage_annotations"] = annotations
        payload["dataset"] = {
            "source_type": "benchmark",
            "relative_path": record.relative_path,
            "metadata": {
                "scene": task.get("scene", episode.get("scene", "Blocks")),
                "task_id": task.get("task_id", episode.get("task", "")),
                "task_name": task.get("task_name", episode.get("task", "")),
                "frequency_hz": episode.get("frequency_hz", configuration.get("frequency_hz", 10.0)),
                "controller": episode.get("controller", configuration.get("controller")),
                "benchmark_configuration": configuration,
                "policy_server": record.summary.get("policy_server", {}),
            },
            "instructions": {"instructions": [instruction]},
        }
        return payload

    def episode(
        self,
        relative_path: str,
        episode_path: str,
        source_type: str = "dataset",
    ) -> dict[str, Any]:
        if source_type == "benchmark":
            record = self._benchmark_record(relative_path)
            known_paths = {
                str(item.get("path")) for item in record.summary.get("episodes", [])
            }
            if episode_path not in known_paths:
                raise FileNotFoundError(f"Unknown benchmark rollout: {episode_path}")
            rollout_path = self._resolve(
                f"{record.relative_path}/{episode_path}/rollout.json", "benchmark"
            )
            payload = self._normalize_benchmark_episode(
                _read_json(rollout_path), record,
            )
            payload["viewer"] = {
                "default_main_camera": _nominal_main_camera(
                    str(payload["dataset"]["metadata"].get("scene", "Blocks"))
                ),
            }
            return payload
        record = self._record(relative_path)
        known_paths = {
            str(item.get("path")) for item in record.manifest.get("episodes", [])
        }
        if episode_path not in known_paths:
            raise FileNotFoundError(f"Unknown episode: {episode_path}")
        trajectory_path = self._resolve(
            f"{record.relative_path}/{episode_path}/trajectory.json"
        )
        payload = _read_json(trajectory_path)
        if payload.get("schema_version") == DATASET_SCHEMA_VERSION:
            labels = load_rotation_tolerance(trajectory_path.parent, require_labels=False)
            payload["tolerance_label_status"] = "unlabeled" if labels is None else "labeled"
            for index, frame in enumerate(payload["trajectory"]):
                frame["rotation_tolerance_rad"] = None if labels is None else labels[index].tolist()
        payload["dataset"] = {
            "relative_path": record.relative_path,
            "metadata": record.metadata,
            "instructions": record.instructions,
        }
        payload["viewer"] = {
            # Deliberately independent of this episode's camera domain
            # randomization: spatial plots always start in the canonical main
            # RGB orientation requested for cross-episode comparison.
            "default_main_camera": _nominal_main_camera(
                str(record.metadata.get("scene", "Blocks"))
            ),
        }
        return payload

    def media(
        self,
        relative_path: str,
        episode_path: str,
        stream: str,
        source_type: str = "dataset",
    ) -> Path:
        if stream not in MEDIA_NAMES:
            raise FileNotFoundError(f"Unknown video stream: {stream}")
        if source_type == "benchmark":
            record = self._benchmark_record(relative_path)
            known_paths = {
                str(item.get("path")) for item in record.summary.get("episodes", [])
            }
            if episode_path not in known_paths:
                raise FileNotFoundError(f"Unknown benchmark rollout: {episode_path}")
            path = self._resolve(
                f"{record.relative_path}/{episode_path}/{stream}", "benchmark"
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            return path
        record = self._record(relative_path)
        known_paths = {
            str(item.get("path")) for item in record.manifest.get("episodes", [])
        }
        if episode_path not in known_paths:
            raise FileNotFoundError(f"Unknown episode: {episode_path}")
        path = self._resolve(f"{record.relative_path}/{episode_path}/{stream}")
        if not path.is_file():
            raise FileNotFoundError(path)
        return path


class DatasetViewerServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], catalog: DatasetCatalog) -> None:
        super().__init__(address, DatasetViewerHandler)
        self.catalog = catalog


class DatasetViewerHandler(BaseHTTPRequestHandler):
    server: DatasetViewerServer

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/datasets":
                self._send_json({"datasets": self.server.catalog.list_datasets()})
                return
            if parsed.path == "/api/dataset":
                query = parse_qs(parsed.query)
                self._send_json(self.server.catalog.dataset_details(
                    self._one(query, "dataset"), self._optional(query, "source", "dataset")
                ))
                return
            if parsed.path == "/api/episode":
                query = parse_qs(parsed.query)
                self._send_json(self.server.catalog.episode(
                    self._one(query, "dataset"), self._one(query, "episode"),
                    self._optional(query, "source", "dataset"),
                ))
                return
            if parsed.path == "/media":
                query = parse_qs(parsed.query)
                path = self.server.catalog.media(
                    self._one(query, "dataset"),
                    self._one(query, "episode"),
                    self._one(query, "stream"),
                    self._optional(query, "source", "dataset"),
                )
                self._send_file(path, allow_ranges=True)
                return
            self._send_static(parsed.path)
        except (FileNotFoundError, ValueError, KeyError) as error:
            self._send_json({"error": str(error)}, status=HTTPStatus.NOT_FOUND)
        except Exception as error:  # keep malformed episodes from terminating the server
            self._send_json({"error": f"Unable to read dataset: {error}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_HEAD(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/media":
                query = parse_qs(parsed.query)
                path = self.server.catalog.media(
                    self._one(query, "dataset"),
                    self._one(query, "episode"),
                    self._one(query, "stream"),
                    self._optional(query, "source", "dataset"),
                )
                self._send_file(path, allow_ranges=True, body=False)
                return
            self._send_static(parsed.path, body=False)
        except (FileNotFoundError, ValueError, KeyError):
            self.send_error(HTTPStatus.NOT_FOUND)

    @staticmethod
    def _one(query: dict[str, list[str]], name: str) -> str:
        values = query.get(name)
        if not values:
            raise ValueError(f"Missing query parameter: {name}")
        return values[0]

    @staticmethod
    def _optional(query: dict[str, list[str]], name: str, default: str) -> str:
        values = query.get(name)
        return default if not values else values[0]

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(
            _json_safe(payload),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_static(self, request_path: str, body: bool = True) -> None:
        relative = "index.html" if request_path in ("", "/") else unquote(request_path.lstrip("/"))
        path = (STATIC_ROOT / relative).resolve()
        if path != STATIC_ROOT and STATIC_ROOT not in path.parents:
            raise FileNotFoundError(relative)
        if not path.is_file():
            raise FileNotFoundError(relative)
        self._send_file(path, allow_ranges=False, body=body)

    def _send_file(self, path: Path, allow_ranges: bool, body: bool = True) -> None:
        size = path.stat().st_size
        start, end = 0, max(0, size - 1)
        status = HTTPStatus.OK
        range_header = self.headers.get("Range") if allow_ranges else None
        if range_header:
            if not range_header.startswith("bytes=") or "," in range_header:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            spec = range_header.removeprefix("bytes=")
            left, separator, right = spec.partition("-")
            if not separator:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            if left:
                start = int(left)
                end = int(right) if right else end
            else:
                suffix = int(right)
                start = max(0, size - suffix)
            end = min(end, size - 1)
            if start < 0 or start > end or start >= size:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            status = HTTPStatus.PARTIAL_CONTENT
        length = max(0, end - start + 1)
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        if allow_ranges:
            self.send_header("Accept-Ranges", "bytes")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if not body or length == 0:
            return
        with path.open("rb") as source:
            source.seek(start)
            remaining = length
            while remaining:
                chunk = source.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def log_message(self, format: str, *args: object) -> None:
        if self.path.startswith("/api") and args and str(args[1]) >= "400":
            super().log_message(format, *args)


def create_server(
    dataset_root: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    benchmark_root: Path | None = None,
) -> DatasetViewerServer:
    return DatasetViewerServer(
        (host, port), DatasetCatalog(dataset_root, benchmark_root),
    )


def run_viewer(
    dataset_root: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    benchmark_root: Path = Path("benchmark_results"),
) -> None:
    server = create_server(dataset_root, host, port, benchmark_root)
    actual_port = server.server_address[1]
    url = f"http://{host}:{actual_port}/"
    print(f"TaskTol dataset viewer: {url}")
    print(f"Dataset root: {server.catalog.root}")
    print(f"Benchmark root: {server.catalog.benchmark_root}")
    if open_browser:
        threading.Timer(0.35, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--benchmark-root", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--no-open", action="store_true", help="Do not open the default browser")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_viewer(
        args.dataset_root, args.host, args.port, not args.no_open,
        args.benchmark_root,
    )


if __name__ == "__main__":
    main()
