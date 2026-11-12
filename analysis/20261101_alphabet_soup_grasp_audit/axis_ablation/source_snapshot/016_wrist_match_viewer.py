"""Serve the matched wrist-frame and VLA action comparison viewer."""

from __future__ import annotations

import argparse
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import threading
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse
import webbrowser

from .wrist_match_analysis import (
    DEFAULT_ANALYSIS_NAME,
    DEFAULT_MATCHING_NAME,
    build_wrist_match_action_analysis,
)


STATIC_ROOT = Path(__file__).with_name("wrist_match_viewer_static")
ANALYSIS_NAME = DEFAULT_ANALYSIS_NAME
MATCHING_NAME = DEFAULT_MATCHING_NAME
CONDITIONS = frozenset(("with_tolerance", "no_tolerance"))


class WristMatchCatalog:
    """Read-only, path-confined access to paired offline benchmark runs."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)

    def _resolve(self, relative_path: str) -> Path:
        candidate = (self.root / unquote(relative_path)).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError("path escapes the offline observation root")
        return candidate

    @lru_cache(maxsize=1)
    def catalog(self) -> dict[str, Any]:
        runs: list[dict[str, Any]] = []
        for summary_path in sorted(self.root.rglob("paired_offline_benchmark.json")):
            run_directory = summary_path.parent
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            relative_run = run_directory.relative_to(self.root).as_posix()
            tasks: list[dict[str, Any]] = []
            for task in summary.get("tasks", []):
                pairs = task.get("pairs", [])
                if not pairs:
                    continue
                comparison = Path(str(pairs[0].get("comparison", "")))
                task_directory = comparison.parts[0] if comparison.parts else ""
                seeds = [
                    int(pair["seed"])
                    for pair in pairs
                    if (run_directory / Path(str(pair.get("comparison", ""))).parent
                        / MATCHING_NAME).is_file()
                ]
                if not seeds:
                    continue
                tasks.append({
                    "directory": task_directory,
                    "task_id": task.get("task_id", task_directory),
                    "task_name": task.get("task_name", task_directory),
                    "seeds": seeds,
                })
            if tasks:
                runs.append({
                    "path": relative_run,
                    "name": run_directory.name,
                    "status": summary.get("status", "unknown"),
                    "replan_steps": summary.get("configuration", {}).get(
                        "replan_steps", 10,
                    ),
                    "tasks": tasks,
                })
        return {"runs": runs}

    def seed_directory(self, run: str, task: str, seed: int) -> Path:
        directory = self._resolve(f"{run}/{task}/seed_{seed}")
        if not (directory / MATCHING_NAME).is_file():
            raise FileNotFoundError(directory)
        return directory

    def analysis(self, run: str, task: str, seed: int) -> dict[str, Any]:
        directory = self.seed_directory(run, task, seed)
        analysis_path = directory / ANALYSIS_NAME
        if analysis_path.is_file():
            return json.loads(analysis_path.read_text(encoding="utf-8"))
        return build_wrist_match_action_analysis(directory, MATCHING_NAME)

    def media(self, run: str, task: str, seed: int, condition: str) -> Path:
        if condition not in CONDITIONS:
            raise FileNotFoundError(f"unknown condition: {condition}")
        path = self.seed_directory(run, task, seed) / condition / "wrist_rgb.mp4"
        if not path.is_file():
            raise FileNotFoundError(path)
        return path


class WristMatchViewerServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], catalog: WristMatchCatalog) -> None:
        super().__init__(address, WristMatchViewerHandler)
        self.catalog = catalog


class WristMatchViewerHandler(BaseHTTPRequestHandler):
    server: WristMatchViewerServer

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/catalog":
                self._send_json(self.server.catalog.catalog())
                return
            if parsed.path == "/api/pair":
                query = parse_qs(parsed.query)
                self._send_json(self.server.catalog.analysis(
                    self._one(query, "run"),
                    self._one(query, "task"),
                    int(self._one(query, "seed")),
                ))
                return
            if parsed.path == "/media":
                query = parse_qs(parsed.query)
                path = self.server.catalog.media(
                    self._one(query, "run"),
                    self._one(query, "task"),
                    int(self._one(query, "seed")),
                    self._one(query, "condition"),
                )
                self._send_file(path, allow_ranges=True)
                return
            self._send_static(parsed.path)
        except (FileNotFoundError, ValueError, KeyError) as error:
            self._send_json({"error": str(error)}, HTTPStatus.NOT_FOUND)
        except Exception as error:
            self._send_json(
                {"error": f"unable to load comparison: {error}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def do_HEAD(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/media":
                query = parse_qs(parsed.query)
                path = self.server.catalog.media(
                    self._one(query, "run"),
                    self._one(query, "task"),
                    int(self._one(query, "seed")),
                    self._one(query, "condition"),
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
            raise ValueError(f"missing query parameter: {name}")
        return values[0]

    def _send_json(
        self, payload: Any, status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        data = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_static(self, request_path: str, body: bool = True) -> None:
        relative = "index.html" if request_path in ("", "/") else unquote(
            request_path.lstrip("/"),
        )
        path = (STATIC_ROOT / relative).resolve()
        if path != STATIC_ROOT and STATIC_ROOT not in path.parents:
            raise FileNotFoundError(relative)
        if not path.is_file():
            raise FileNotFoundError(relative)
        self._send_file(path, allow_ranges=False, body=body)

    def _send_file(
        self, path: Path, allow_ranges: bool, body: bool = True,
    ) -> None:
        size = path.stat().st_size
        start, end = 0, max(0, size - 1)
        status = HTTPStatus.OK
        range_header = self.headers.get("Range") if allow_ranges else None
        if range_header:
            if not range_header.startswith("bytes=") or "," in range_header:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            left, separator, right = range_header.removeprefix("bytes=").partition("-")
            if not separator:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            if left:
                start = int(left)
                end = int(right) if right else end
            else:
                start = max(0, size - int(right))
            end = min(end, size - 1)
            if start < 0 or start > end or start >= size:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            status = HTTPStatus.PARTIAL_CONTENT
        length = max(0, end - start + 1)
        self.send_response(status)
        self.send_header(
            "Content-Type",
            mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        )
        self.send_header("Content-Length", str(length))
        if allow_ranges:
            self.send_header("Accept-Ranges", "bytes")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if not body:
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


def run_viewer(
    root: Path,
    host: str = "127.0.0.1",
    port: int = 8766,
    open_browser: bool = True,
) -> None:
    server = WristMatchViewerServer((host, port), WristMatchCatalog(root))
    url = f"http://{host}:{server.server_address[1]}/"
    print(f"Wrist match viewer: {url}")
    print(f"Offline observation root: {server.catalog.root}")
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
    parser.add_argument("--root", type=Path, default=Path("offline_observation_results"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-open", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_viewer(args.root, args.host, args.port, not args.no_open)


if __name__ == "__main__":
    main()
