import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from examples.libero.serve_benchmark_viewer import Catalog, Handler, SUITES, ThreadingHTTPServer


@pytest.fixture
def viewer(tmp_path):
    run = tmp_path / "example_run"
    run.mkdir()
    (run / "evaluation_request.json").write_text(json.dumps(dict(expected_episodes=240, seeds=list(range(6)))))
    (run / "official_tasks.json").write_text(json.dumps([
        dict(suite=s, task_id=i, language=f"task {i}") for s in SUITES for i in range(10)
    ]))
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.catalog = Catalog(tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield run, server.catalog, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join()


def commit(run, seed=0, success=True):
    folder = run / f"episodes/libero_spatial/task_00/seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    pose = dict(position_m=[0.2, 0.1, 0.5], rotation_vector_rad=[0, 0, 0], joint_position_rad=[0]*7, finger_qpos_m=[0.02, -0.02])
    (folder / "trajectory.json").write_text(json.dumps([
        dict(pre_action=pose, post_action=pose, nominal_target=pose, action=[0]*6+[-1], optimizer=dict(feasible=success), aborted=not success, success=success)
    ]))
    (folder / "rollout.mp4").write_bytes(b"0123456789")
    (folder / "result.json").write_text(json.dumps(dict(complete=True, suite="libero_spatial", task_id=0, seed=seed, steps=1, success=success, termination="task_success" if success else "ik_infeasible", language="pick up bowl", controller="shared", finished_at="2026-11-02T20:00:00+08:00")))
    return folder


def get_json(url):
    with urlopen(url) as response:
        return json.load(response)


def test_live_catalog_reads_new_commits_and_counts_failed_rollouts(viewer):
    run, catalog, url = viewer
    first = get_json(url + "/api/live")
    assert len(first["datasets"]) == 40
    assert first["runs"][0]["completed"] == 0
    commit(run, 0, True)
    commit(run, 1, False)
    (run / "episodes/libero_spatial/task_00/seed_2").mkdir()
    catalog.cached_at = 0
    live = get_json(url + "/api/live")["runs"][0]
    assert (live["completed"], live["successes"], live["failures"]) == (2, 1, 1)
    assert live["success_rate"] == 0.5
    assert [e["seed"] for e in live["datasets"][0]["episodes"]] == [0, 1]


def test_episode_preserves_physical_actions_and_aborted_frame(viewer):
    run, _, url = viewer
    commit(run, 0, False)
    payload = get_json(url + "/api/episode?dataset=example_run/tasks/libero_spatial/task_00&episode=seed_0")
    frame = payload["trajectory"][0]
    assert frame["action"] == [0]*6+[-1]
    assert frame["aborted"] is True
    assert frame["gripper_state"]["aperture_m"] == 0.04
    assert payload["episode"]["result"]["termination_reason"] == "ik_infeasible"


def test_release_telemetry_preserves_accepted_and_rejected_values_separately(viewer):
    run, _, url = viewer
    folder = commit(run)
    rows = json.loads((folder / 'trajectory.json').read_text())
    rows[0].update(phase='EXEC', semantic_mask='010', effective_mask='010',
                   rotation_release_rad=[.1, -.2, .03], bounds_rad=[.5235987756]*6,
                   rejected_rotation_release_rad=[2, 3, 4], reference_held=True,
                   held_reference_phase='APP', heading_guarded=False,
                   predicted_tolerance_violation_rad=[0, 0, 0])
    (folder / 'trajectory.json').write_text(json.dumps(rows))
    payload = get_json(url + '/api/episode?dataset=example_run/tasks/libero_spatial/task_00&episode=seed_0')
    frame = payload['trajectory'][0]
    assert frame['phase'] == 'EXEC'
    assert frame['rotation_release_rad'] == [.1, -.2, .03]
    assert frame['rejected_rotation_release_rad'] == [2, 3, 4]
    assert frame['effective_mask'] == '010' and frame['reference_held']
    for asset in ['release_values.js', 'release.js']:
        with urlopen(url + '/' + asset) as response:
            assert response.status == 200


def test_switching_episode_does_not_wait_for_catalog_scan(viewer, monkeypatch):
    run, catalog, url = viewer
    commit(run)
    def unavailable_catalog():
        raise AssertionError("Episode and media requests must not scan the full catalog")
    monkeypatch.setattr(catalog, "snapshot", unavailable_catalog)
    query = "?dataset=example_run/tasks/libero_spatial/task_00&episode=seed_0"
    assert get_json(url + "/api/episode" + query)["episode"]["seed"] == 0
    with urlopen(url + "/media" + query + "&stream=main_rgb.mp4") as response:
        assert response.read() == b"0123456789"
        assert "immutable" in response.headers["Cache-Control"]


def test_per_axis_tolerances_are_labeled_and_sent_with_accepted_hold_limits(viewer):
    run, catalog, url = viewer
    request = dict(expected_episodes=400,seeds=list(range(10)),tolerance_enabled=True,
                   semantic_mask_policy='user_edited_task_phase_angles',tolerance_bounds_policy='per_task_subtask_phase',
                   tolerance_bounds_deg=None)
    (run/'evaluation_request.json').write_text(json.dumps(request))
    folder = commit(run)
    rows = json.loads((folder/'trajectory.json').read_text())
    rows[0].update(semantic_tolerance_deg=[0,5,0],semantic_bounds_deg=[0,0,5,5,0,0],
                   reference_held=True,held_reference_bounds_rad=[.1,.1,.2,.2,.3,.3])
    (folder/'trajectory.json').write_text(json.dumps(rows))
    catalog.cached_at = 0
    live = get_json(url+'/api/live')['runs'][0]
    assert '分轴角度' in live['label']
    frame = get_json(url+'/api/episode?dataset=example_run/tasks/libero_spatial/task_00&episode=seed_0')['trajectory'][0]
    assert frame['semantic_tolerance_deg'] == [0,5,0]
    assert frame['semantic_bounds_deg'] == [0,0,5,5,0,0]
    assert frame['held_reference_bounds_rad'] == [.1,.1,.2,.2,.3,.3]


def test_media_range_seeking_and_head_use_same_committed_pair(viewer):
    run, _, url = viewer
    commit(run)
    media = url + "/media?dataset=example_run/tasks/libero_spatial/task_00&episode=seed_0&stream="
    for stream in ("main_rgb.mp4", "wrist_rgb.mp4"):
        with urlopen(Request(media + stream, headers={"Range": "bytes=2-5"})) as response:
            assert response.status == 206
            assert response.headers["Content-Range"] == "bytes 2-5/10"
            assert response.read() == b"2345"
    with urlopen(Request(media + "main_rgb.mp4", method="HEAD")) as response:
        assert response.headers["Content-Length"] == "10"
        assert response.read() == b""


@pytest.mark.parametrize("path", [
    "/media?dataset=example_run/tasks/libero_spatial/task_00&episode=seed_4&stream=main_rgb.mp4",
    "/media?dataset=example_run/tasks/libero_spatial/task_00&episode=../../secret&stream=main_rgb.mp4",
    "/api/log?run=../../secret&seed=0",
    "/%2e%2e/serve_benchmark_viewer.py",
])
def test_uncommitted_records_and_paths_outside_run_are_unavailable(viewer, path):
    run, _, url = viewer
    commit(run)
    with pytest.raises(HTTPError) as error:
        urlopen(url + path)
    assert error.value.code == 404


def test_supplement_seeds6to9_are_visible_and_serve_media(viewer):
    run, catalog, url = viewer
    (run / 'evaluation_request.json').write_text(json.dumps(dict(expected_episodes=160, seeds=[6, 7, 8, 9])))
    commit(run, 8, True)
    commit(run, 9, False)
    catalog.cached_at = 0
    view = get_json(url + '/api/live')['runs'][0]
    assert (view['expected'], view['completed']) == (160, 2)
    assert [e['seed'] for e in view['datasets'][0]['episodes']] == [8, 9]
    query = '?dataset=example_run/tasks/libero_spatial/task_00&episode=seed_9'
    assert get_json(url + '/api/episode' + query)['episode']['seed'] == 9
    with urlopen(url + '/media' + query + '&stream=main_rgb.mp4') as response:
        assert response.read() == b'0123456789'
    with pytest.raises(HTTPError):
        urlopen(url + '/api/episode?dataset=example_run/tasks/libero_spatial/task_00&episode=seed_0')


def combined_manifest(run):
    source = run.parent / 'original_results'
    retake = run.parent / 'failure_retests'
    rows = []
    for suite in SUITES:
        for task in range(10):
            for seed in range(10):
                parent = run if seed >= 6 else source
                if (suite, task, seed) == ('libero_spatial', 0, 0):
                    parent = retake
                folder = parent / f'episodes/{suite}/task_{task:02d}/seed_{seed}'
                folder.mkdir(parents=True, exist_ok=True)
                p = folder / 'result.json'
                p.write_text(json.dumps(dict(complete=True, suite=suite, task_id=task, seed=seed,
                    steps=1, success=True, termination='task_success', language='test', controller='shared')))
                rows.append(dict(suite=suite, task_id=task, seed=seed, source_result=str(p)))
    commit(retake, 0, False)
    commit(run, 9, True)
    (run / 'evaluation_request.json').write_text(json.dumps(dict(expected_episodes=160, seeds=[6,7,8,9])))
    (run / 'combined_latest_400.json').write_text(json.dumps(dict(complete=True, results=rows)))
    return rows


def test_combined_400_uses_retest_sources_and_new_seed_media_without_copying(viewer):
    run, catalog, url = viewer
    combined_manifest(run)
    catalog.cached_at = 0
    views = get_json(url + '/api/live')['runs']
    combined = next(v for v in views if v['name'] == 'example_run--latest400')
    assert (combined['completed'], combined['expected'], len(combined['datasets'])) == (400, 400, 40)
    assert all(len(d['episodes']) == 10 for d in combined['datasets'])
    query = '?dataset=example_run--latest400/tasks/libero_spatial/task_00&episode=seed_0'
    payload = get_json(url + '/api/episode' + query)
    assert payload['episode']['result']['success'] is False
    assert payload['episode']['benchmark_source']['run'] == 'failure_retests'
    with urlopen(url + '/media' + query + '&stream=main_rgb.mp4') as response:
        assert response.read() == b'0123456789'
    direct = get_json(url + '/api/episode?dataset=example_run/tasks/libero_spatial/task_00&episode=seed_9')
    latest = get_json(url + '/api/episode?dataset=example_run--latest400/tasks/libero_spatial/task_00&episode=seed_9')
    assert direct['dataset']['relative_path'] != latest['dataset']['relative_path']
    # Direct access keeps working without any catalog scan.
    catalog.snapshot = lambda: (_ for _ in ()).throw(AssertionError('unexpected catalog scan'))
    assert get_json(url + '/api/episode' + query)['episode']['seed'] == 0


def test_combined_manifest_cannot_expose_files_outside_catalog(viewer):
    run, catalog, url = viewer
    rows = combined_manifest(run)
    rows[0]['source_result'] = str(run.parent.parent / 'secret/result.json')
    (run / 'combined_latest_400.json').write_text(json.dumps(dict(complete=True, results=rows)))
    with pytest.raises(HTTPError) as error:
        urlopen(url + '/api/episode?dataset=example_run--latest400/tasks/libero_spatial/task_00&episode=seed_0')
    assert error.value.code == 404


def test_formal400_uses_formal_manifest_and_label(viewer):
    run, catalog, url = viewer
    combined_manifest(run)
    (run/'combined_latest_400.json').rename(run/'combined_formal_400.json')
    request=json.loads((run/'evaluation_request.json').read_text())
    request['formal_baseline']=True
    (run/'evaluation_request.json').write_text(json.dumps(request))
    catalog.cached_at=0
    combined=next(r for r in get_json(url+'/api/live')['runs'] if r['name']=='example_run--latest400')
    assert combined['label']=='正式无容差 · 40任务×10次'
    assert combined['completed']==400 and len(combined['datasets'])==40


def test_partial_retest_only_exposes_selected_task_seed_pairs(viewer):
    run, catalog, url = viewer
    request = dict(expected_episodes=1, seeds=list(range(6)),
                   selected_episodes=[dict(suite='libero_spatial', task_id=0, seed=2)])
    (run / 'evaluation_request.json').write_text(json.dumps(request))
    commit(run, 2)
    commit(run, 0)
    catalog.cached_at = 0
    view = get_json(url + '/api/live')['runs'][0]
    assert (view['completed'], view['expected'], len(view['datasets'])) == (1, 1, 1)
    with pytest.raises(HTTPError):
        urlopen(url + '/api/episode?dataset=example_run/tasks/libero_spatial/task_00&episode=seed_0')
