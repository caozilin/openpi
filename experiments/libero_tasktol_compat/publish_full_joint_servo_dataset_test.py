from pathlib import Path

import pytest

from experiments.libero_tasktol_compat import publish_full_joint_servo_dataset as publisher


def test_incomplete_dataset_cannot_pass_upload_gate(tmp_path):
    for name in ("summary", "manifest", "validation", "plan"):
        publisher.dump(tmp_path / (name + ".json"), {"complete": False})
    with pytest.raises(AssertionError):
        publisher.local_audit(tmp_path)


def test_transfer_keeps_link_dependencies_and_excludes_transient_media(tmp_path):
    run = tmp_path / "full"
    baseline = tmp_path / "baseline"
    for relative in ("full/episodes/suite/00/demo_0/recording.hdf5",
                     "baseline/episodes/suite/00/demo_1/recording.hdf5",
                     "full/datasets/suite/task.hdf5", "full/.scratch/trial.mp4",
                     "baseline/viewer/video.mp4", "full/viewer/video.mp4"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    publisher.dump(run / "plan.json", {"baseline_run": str(baseline)})
    publisher.dump(run / "manifest.json", {"complete": True})
    plan = publisher.transfer_plan(run)
    paths = {item["path"] for item in plan["files"]}
    assert "baseline/episodes/suite/00/demo_1/recording.hdf5" in paths
    assert "full/episodes/suite/00/demo_0/recording.hdf5" in paths
    assert "full/datasets/suite/task.hdf5" in paths
    assert not any("viewer" in path or ".scratch" in path for path in paths)
    assert all(not Path(path).is_absolute() for path in paths)
    for item in plan["files"]:
        assert publisher.sha(tmp_path / item["path"]) == item["sha256"]
