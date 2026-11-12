import h5py
import numpy as np
import pytest

from experiments.libero_tasktol_compat import complete_joint_servo_dataset as full


def test_full_completion_requires_failures_and_successes_to_cover_all_sources(tmp_path):
    jobs = [dict(demo=f"demo_{i}", source_input_sha256=str(i)) for i in range(12)]
    entry = dict(case=0, task=dict(suite="libero_spatial", task_id=0, task_name="task"), candidates=jobs)
    plan = dict(tasks=[entry], inherited_sources=10, inherited_successes=10)
    for i in range(10):
        full.runner.dump(full.record_path(tmp_path, 0, f"demo_{i}"),
                         dict(case=0, demo=f"demo_{i}", source_input_sha256=str(i),
                              success=True, report=dict(executed_frames=5)))
    summary = full.collect_summary(tmp_path, plan)
    assert not summary["complete"] and summary["total_pending"] == 2
    full.runner.dump(full.record_path(tmp_path, 0, "demo_10"),
                     dict(case=0, demo="demo_10", source_input_sha256="10", success=False))
    full.runner.dump(full.record_path(tmp_path, 0, "demo_11"),
                     dict(case=0, demo="demo_11", success=False, error="GPU unavailable"))
    summary = full.collect_summary(tmp_path, plan)
    assert not summary["complete"] and summary["total_pending"] == 1
    assert summary["total_failed"] == 1 and summary["process_errors"] == 1
    full.runner.dump(full.record_path(tmp_path, 0, "demo_11"),
                     dict(case=0, demo="demo_11", source_input_sha256="11", success=False))
    summary = full.collect_summary(tmp_path, plan)
    assert summary["complete"] and summary["total_attempted"] == 12
    assert summary["total_success"] == 10 and summary["total_failed"] == 2
    assert summary["new_successes"] == 0


def test_changed_source_cannot_be_reused(tmp_path):
    full.runner.dump(full.record_path(tmp_path, 0, "demo_0"),
                     dict(case=0, demo="demo_0", source_input_sha256="old", success=False))
    with pytest.raises(AssertionError):
        full.load_records(tmp_path, dict(case=0, candidates=[dict(demo="demo_0", source_input_sha256="new")]))


def test_gzip_recording_preserves_every_camera_channel(tmp_path):
    image = np.random.default_rng(0).integers(0, 256, (16, 16, 3), dtype=np.uint8)
    path = tmp_path / "recording.hdf5"
    writer = full.LosslessObservationWriter(path, "demo_0")
    writer.append(writer.obs, "agentview_rgb", image)
    writer.count += 1
    writer.append(writer.obs, "agentview_rgb", image[::-1])
    writer.close()
    with h5py.File(path) as file:
        data = file["data/demo_0/obs/agentview_rgb"]
        assert data.compression == "gzip" and data.compression_opts == 6
        np.testing.assert_array_equal(data[:], np.stack([image, image[::-1]]))
