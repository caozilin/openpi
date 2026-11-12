"""Read-only live LIBERO benchmark adapter for the existing Dataset Studio UI."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import sys
import threading
import time
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.libero.serve_mask_annotation_viewer import byte_range, install_exit_signals

STATIC = Path(__file__).with_name("benchmark_viewer")
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def confined(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Path outside benchmark root")
    return path


def alive(path):
    record = read_json(path, {})
    pid = record.get("pid")
    try:
        command = (Path("/proc") / str(int(pid)) / "cmdline").read_bytes()
        return b"eval_reconstructed_libero_local.py" in command
    except (ValueError, TypeError, OSError):
        return False


COMBINED_SUFFIX = '--latest400'


class Catalog:
    def __init__(self, root):
        self.root = root.resolve()
        self.lock = threading.Lock()
        self.cached_at = 0
        self.cached = None
        self.episode_cache = OrderedDict()
        self.episode_lock = threading.Lock()
        self.result_cache = {}
        self.manifest_cache = {}

    def cached_json(self, path, default=None):
        try:
            stat = path.stat()
        except FileNotFoundError:
            return default
        fingerprint = (stat.st_mtime_ns, stat.st_size)
        previous = self.result_cache.get(path)
        if previous and previous[0] == fingerprint:
            return previous[1]
        value = read_json(path, default)
        self.result_cache[path] = (fingerprint, value)
        return value

    @staticmethod
    def expected_keys(request, tasks):
        seeds = request.get('seeds', list(range(6)))
        if not seeds or len(seeds) != len(set(seeds)) or any(type(s) is not int or s < 0 for s in seeds):
            raise ValueError('Invalid evaluation seeds')
        task_keys = {(t['suite'], t['task_id']) for t in tasks}
        if task_keys != {(s, t) for s in SUITES for t in range(10)} or len(tasks) != 40:
            raise ValueError('Unknown benchmark task set')
        possible = {(s, t, z) for s, t in task_keys for z in seeds}
        selected = request.get('selected_episodes')
        expected = possible if selected is None else {(x['suite'], x['task_id'], x['seed']) for x in selected}
        if not expected <= possible or request.get('expected_episodes') != len(expected):
            raise ValueError('Evaluation selection does not match expected episodes')
        return expected

    def manifest_sources(self, run):
        path = run / ('combined_formal_400.json' if (run / 'combined_formal_400.json').exists() else 'combined_latest_400.json')
        stat = path.stat()
        fingerprint = (stat.st_mtime_ns, stat.st_size)
        previous = self.manifest_cache.get(path)
        if previous and previous[0] == fingerprint:
            return previous[1]
        manifest = self.cached_json(path, {})
        if manifest.get('complete') is not True:
            raise ValueError('Combined evaluation is incomplete')
        sources = {}
        for row in manifest.get('results', []):
            key = (row['suite'], row['task_id'], row['seed'])
            source = Path(row['source_result']).resolve()
            tail = ('episodes', row['suite'], f"task_{row['task_id']:02d}", f"seed_{row['seed']}", 'result.json')
            if (not source.is_relative_to(self.root) or tuple(source.parts[-5:]) != tail
                    or source.parent.parent.parent.parent.parent.parent != self.root or key in sources):
                raise ValueError('Combined result source is outside the benchmark catalog or duplicated')
            sources[key] = source
        if set(sources) != {(s, t, z) for s in SUITES for t in range(10) for z in range(10)}:
            raise ValueError('Combined evaluation must contain exactly 400 unique task/seed pairs')
        self.manifest_cache[path] = (fingerprint, sources)
        return sources

    def view(self, name):
        combined = name.endswith(COMBINED_SUFFIX)
        base = name[:-len(COMBINED_SUFFIX)] if combined else name
        run = confined(self.root, base)
        request = self.cached_json(run / 'evaluation_request.json', {})
        tasks = self.cached_json(run / 'official_tasks.json', [])
        if combined:
            sources = self.manifest_sources(run)
            request = dict(request, seeds=list(range(10)), expected_episodes=400)
            request.pop('selected_episodes', None)
            formal = request.get('formal_baseline') is True
            label = '正式无容差 · 40任务×10次' if formal else '无容差 · 40任务×10次（含失败重测）'
            priority = 200 if formal else 100
        else:
            sources = None
            label = ('正式无容差 · 新测seed 0–5（240回合）' if request.get('formal_baseline') is True
                     else '分轴角度容差 · 40任务×10次' if request.get('tolerance_bounds_policy') == 'per_task_subtask_phase'
                     else '对应mask ±30° · 40任务×10次' if request.get('semantic_mask_policy') == 'user_confirmed_task_phase_masks'
                     else '新增无容差 · seed 6–9（160回合）' if request.get('seeds') == [6, 7, 8, 9]
                     else '无容差 · 17条失败重测' if request.get('expected_episodes') == 17
                     else name)
            priority = 0
        expected = self.expected_keys(request, tasks)
        return run, request, tasks, expected, sources, label, priority

    def snapshot(self):
        with self.lock:
            if self.cached is not None and time.monotonic() - self.cached_at < 2:
                return self.cached
            datasets, runs = [], []
            requests = sorted(self.root.glob('*/evaluation_request.json'), key=lambda p: p.stat().st_mtime, reverse=True)
            views = []
            for path in requests:
                if any((path.parent / file).exists() for file in ('combined_formal_400.json', 'combined_latest_400.json')):
                    views.append(path.parent.name + COMBINED_SUFFIX)
                views.append(path.parent.name)
            for name in views:
                try:
                    run, request, tasks, expected, sources, label, priority = self.view(name)
                except (ValueError, KeyError, FileNotFoundError, TypeError):
                    continue
                seeds = request.get('seeds', list(range(6)))
                entries, records, warnings = [], [], []
                committed = {}
                result_paths = sources.values() if sources is not None else (run / 'episodes').glob('*/task_*/seed_*/result.json')
                for result_path in result_paths:
                    result = self.cached_json(result_path)
                    if result:
                        committed[(result.get('suite'), result.get('task_id'), result.get('seed'))] = result
                state = self.cached_json(run / 'evaluation_state.json', {})
                for task in tasks:
                    suite, tid = task['suite'], task['task_id']
                    expected_seeds = [seed for seed in seeds if (suite, tid, seed) in expected]
                    if not expected_seeds:
                        continue
                    relative = f"{name}/tasks/{suite}/task_{tid:02d}"
                    episodes = []
                    for seed in expected_seeds:
                        result = committed.get((suite, tid, seed))
                        if result is None:
                            continue
                        if not result.get('complete'):
                            warnings.append(f'Invalid committed result: {suite}/{tid}/{seed}')
                            continue
                        tolerance_mode = 'tolerance' if result.get('tolerance_enabled') else 'off'
                        episode = dict(path=f'seed_{seed}', seed=seed, frames=result['steps'], success=result['success'],
                                       termination=result['termination'], finished_at=result.get('finished_at'), tolerance_mode=tolerance_mode)
                        episodes.append(episode)
                        records.append(dict(episode, suite=suite, task_id=tid, language=task['language'], relative_path=relative))
                    entries.append(dict(source_type='benchmark', relative_path=relative, eval_run=name,
                        eval_label=label, view_priority=priority, suite=suite, scene=suite, robot='panda', task_id=tid,
                        task_name=task['language'], display_name=f"{tid:02d} · {task['language']}", frequency_hz=20,
                        episode_count=len(episodes), expected_episodes=len(expected_seeds), success_count=sum(e['success'] for e in episodes),
                        episodes=episodes, tolerance_mode='tolerance' if request.get('tolerance_enabled') else 'off',
                        run_status='complete' if sources is not None else state.get('stage', 'unknown')))
                workers = [dict(seed=seed, alive=alive(run / f'client_{seed}_process.json'),
                                completed=sum(p['seed'] == seed for e in entries for p in e['episodes']), current_task=None) for seed in seeds]
                total = len(records)
                success = sum(r['success'] for r in records)
                runs.append(dict(name=name, label=label, stage='complete' if sources is not None else state.get('stage', 'unknown'),
                    manager_alive=False if sources is not None else alive(run / 'evaluation_process.json'), completed=total,
                    expected=len(expected), successes=success, failures=total-success,
                    infrastructure_errors=len(list((run / 'episodes').glob('*/task_*/seed_*/infrastructure_error.json'))),
                    success_rate=success/total if total else None, workers=workers, warnings=warnings,
                    suites=[dict(suite=s, completed=sum(r['suite'] == s for r in records), successes=sum(r['suite'] == s and r['success'] for r in records),
                                 expected=sum(k[0] == s for k in expected)) for s in SUITES], datasets=entries,
                    records=sorted(records, key=lambda e: e.get('finished_at') or '', reverse=True),
                    model=dict(checkpoint_step=request.get('checkpoint_step'), state_enabled=request.get('state_enabled'),
                               controller=request.get('controller'), client_location=request.get('client_location'))))
                datasets.extend(entries)
            self.cached = dict(datasets=datasets, runs=runs, refreshed_at=time.time())
            self.cached_at = time.monotonic()
            return self.cached

    def locate(self, dataset, episode=None):
        match = re.fullmatch(r'([^/]+)/tasks/(libero_spatial|libero_object|libero_goal|libero_10)/task_(0[0-9])', dataset)
        if match is None:
            raise FileNotFoundError('Unknown benchmark task')
        name, suite, task_id = match.groups()
        run, _, _, expected, sources, label, _ = self.view(name)
        entry = dict(eval_run=name, suite=suite, task_id=int(task_id), label=label)
        folder = confined(run, f"episodes/{suite}/task_{int(task_id):02d}")
        if episode is not None:
            if not re.fullmatch(r'seed_(0|[1-9][0-9]*)', episode):
                raise FileNotFoundError('Episode is not a completed benchmark record')
            key = (suite, int(task_id), int(episode[5:]))
            if key not in expected:
                raise FileNotFoundError('Episode is not selected in this evaluation')
            folder = sources[key].parent if sources is not None else confined(run, str(folder.relative_to(run) / episode))
            result = self.cached_json(folder / 'result.json', {})
            if not result.get('complete') or (result.get('suite'), result.get('task_id'), result.get('seed')) != key:
                raise FileNotFoundError('Episode is not a completed benchmark record')
        return entry, folder

    def episode(self, dataset, episode):
        entry, folder = self.locate(dataset, episode)
        trace_path, result_path = folder / 'trajectory.json', folder / 'result.json'
        key = (dataset, str(folder), trace_path.stat().st_mtime_ns, result_path.stat().st_mtime_ns)
        with self.episode_lock:
            if key in self.episode_cache:
                self.episode_cache.move_to_end(key)
                return self.episode_cache[key]
        result = read_json(result_path)
        rows = read_json(trace_path)
        if not isinstance(rows, list):
            raise ValueError('Missing committed trajectory')
        def pose(value):
            return dict(position_m=value['position_m'], rotation_vector_base_rad=value['rotation_vector_rad'])
        frames = []
        for i, row in enumerate(rows):
            before = row['pre_action']
            fingers = before['finger_qpos_m']
            frames.append(dict(index=i, time_s=i/20, phase=row.get('phase', 'APP'), action=row['action'], ee_state=pose(before),
                semantic_mask=row.get('semantic_mask', '000'), effective_mask=row.get('effective_mask', '000'),
                semantic_subtask_label=row.get('semantic_subtask_label'),
                semantic_tolerance_deg=row.get('semantic_tolerance_deg'), semantic_bounds_deg=row.get('semantic_bounds_deg'),
                rotation_release_rad=row.get('rotation_release_rad'), bounds_rad=row.get('bounds_rad'),
                predicted_tolerance_violation_rad=row.get('predicted_tolerance_violation_rad'),
                rejected_rotation_release_rad=row.get('rejected_rotation_release_rad'),
                held_reference_phase=row.get('held_reference_phase'),
                held_reference_bounds_rad=row.get('held_reference_bounds_rad'),
                heading_guarded=row.get('heading_guarded', False),
                nominal_target_pose=pose(row['nominal_target']), post_action_pose=pose(row['post_action']),
                joint_state=dict(position_rad=before['joint_position_rad']), gripper_state=dict(aperture_m=fingers[0]-fingers[1]),
                optimizer=row['optimizer'], aborted=row['aborted'], success=row['success'],
                first_native_success_frame=row.get('first_native_success_frame'), post_success_step=row.get('post_success_step'),
                solver_failed=row.get('solver_failed', not row['optimizer'].get('feasible', True)),
                reference_held=row.get('reference_held', False), consecutive_solver_failures=row.get('consecutive_solver_failures', 0),
                max_consecutive_solver_failures=row.get('max_consecutive_solver_failures', 20)))
        payload = dict(episode=dict(seed=result['seed'], instruction=result['language'], controller=result['controller'],
            result=dict(success=result['success'], termination_reason=result['termination']), benchmark_result=result,
            benchmark_source=dict(run=folder.parents[3].name, result_path=str(result_path))),
            trajectory=frames, stage_annotations=[], tolerance_label_status='disabled',
            dataset=dict(source_type='benchmark', relative_path=dataset, metadata=dict(frequency_hz=20), instructions=[result['language']]))
        with self.episode_lock:
            self.episode_cache[key] = payload
            while len(self.episode_cache) > 24:
                self.episode_cache.popitem(last=False)
        return payload


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def json(self, value, head=False, status=200):
        data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if not head:
            self.wfile.write(data)

    def file(self, path, head=False, immutable=False):
        if not path.is_file():
            raise FileNotFoundError("File unavailable")
        size = path.stat().st_size
        try:
            requested = byte_range(self.headers.get("Range"), size)
        except ValueError:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        start, end = requested or (0, size-1)
        self.send_response(206 if requested else 200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(end-start+1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "private, max-age=86400, immutable" if immutable else "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        if requested:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if not head:
            with path.open("rb") as f:
                f.seek(start)
                remaining = end-start+1
                while remaining:
                    data = f.read(min(65536, remaining))
                    if not data:
                        break
                    self.wfile.write(data)
                    remaining -= len(data)

    def serve(self, head=False):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        one = lambda key, default="": query.get(key, [default])[0]
        catalog = self.server.catalog
        try:
            if parsed.path in ("/api/datasets", "/api/live"):
                snap = catalog.snapshot()
                self.json(dict(datasets=snap["datasets"]) if parsed.path.endswith("datasets") else snap, head)
            elif parsed.path == "/api/episode":
                self.json(catalog.episode(one("dataset"), one("episode")), head)
            elif parsed.path == "/api/log":
                run = next((r for r in catalog.snapshot()["runs"] if r["name"] == one("run")), None)
                if run is None or one("seed") not in tuple(str(w['seed']) for w in run['workers']):
                    raise ValueError("Unknown run or seed")
                p = confined(catalog.root, f"{run['name']}/client_{one('seed')}.log")
                with p.open("rb") as f:
                    f.seek(max(0, p.stat().st_size-12000))
                    text = f.read().decode(errors="replace")
                self.json(dict(text=text), head)
            elif parsed.path == "/media":
                if one("stream") not in ("main_rgb.mp4", "wrist_rgb.mp4", "trajectory.json", "result.json"):
                    raise ValueError("Unknown media stream")
                _, folder = catalog.locate(one("dataset"), one("episode"))
                name = "rollout.mp4" if one("stream").endswith(".mp4") else one("stream")
                self.file(folder / name, head, immutable=True)
            else:
                self.file(confined(STATIC, parsed.path.lstrip("/") or "index.html"), head)
        except (FileNotFoundError, ValueError, KeyError) as error:
            self.json(dict(error=str(error)), head, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        self.serve()

    def do_HEAD(self):
        self.serve(True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data/libero_evaluation")
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args(argv)
    install_exit_signals()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    server.catalog = Catalog(args.root)
    print(f"LIBERO Benchmark Studio: http://127.0.0.1:{server.server_port} · 前台运行，Ctrl+C结束", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
