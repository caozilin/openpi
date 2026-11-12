import json
import os

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from examples.libero import convert_reconstructed_hdf5_to_lerobot as exporter


def fixture_raw(tmp_path):
    raw = tmp_path / "raw"
    folder = raw / "episodes/libero_spatial/00/demo_7"
    folder.mkdir(parents=True)
    recording = folder / "recording.hdf5"
    image = np.random.default_rng(5).integers(0, 256, (3, 256, 256, 3), dtype=np.uint8)
    with h5py.File(recording, "w") as file:
        group = file.create_group("data/demo_7")
        group.attrs.update(success=True, num_samples=3)
        group.create_dataset("dones", data=[0, 0, 1])
        group.create_dataset("actions", data=np.arange(21).reshape(3, 7) * .001)
        group.create_dataset("obs/ee_states", data=np.arange(18).reshape(3, 6) * .01)
        group.create_dataset("obs/gripper_states", data=np.array([[.01, -.01], [.02, -.02], [.03, -.03]]))
        group.create_dataset("obs/agentview_rgb", data=image)
        group.create_dataset("obs/eye_in_hand_rgb", data=255 - image)
    index = raw / "datasets/libero_spatial/task_demo.hdf5"
    index.parent.mkdir(parents=True)
    with h5py.File(index, "w") as file:
        file.attrs.update(control_hz=20, language="move the bowl")
        file["data/demo_7"] = h5py.ExternalLink(os.path.relpath(recording, index.parent), "/data/demo_7")
    manifest = dict(complete=True, episodes=1, tasks=[dict(case=0, suite="libero_spatial", task_id=0,
                                                          path=str(index), episodes=1, frames=3)])
    exporter.dump(raw / "manifest.json", manifest)
    exporter.dump(raw / "validation.json", dict(complete=True, episodes=1))
    return raw


def test_conversion_resume_full_pixel_audit_and_corruption_detection(tmp_path):
    raw = fixture_raw(tmp_path)
    output = tmp_path / "export"
    result = exporter.convert(raw, output, "tests/libero_reconstruction")
    assert result["complete"] and result["episodes"] == 1 and result["frames"] == 3
    info = json.loads((output / "meta/info.json").read_text())
    assert info["fps"] == 20 and info["features"]["state"]["shape"] == [8]
    path = next(output.glob("data/*/*.parquet"))
    original_hash = exporter.sha(path)
    # Simulate an uncommitted tail; resume keeps the committed episode exactly.
    tail = path.with_name("episode_000001.parquet")
    tail.write_bytes(b"interrupted tail")
    result = exporter.convert(raw, output, "tests/libero_reconstruction", resume=True)
    assert result["complete"] and not tail.exists() and exporter.sha(path) == original_hash
    table = pq.read_table(path)
    actions = table["actions"].to_pylist()
    actions[0][0] += 1.
    table = table.set_column(table.schema.get_field_index("actions"), "actions", pa.array(actions, type=table["actions"].type))
    pq.write_table(table, path)
    with pytest.raises(AssertionError):
        exporter.convert(raw, output, "tests/libero_reconstruction", validate_only=True)


def test_unfinished_raw_dataset_is_rejected(tmp_path):
    raw = fixture_raw(tmp_path)
    manifest = json.loads((raw / "manifest.json").read_text())
    exporter.dump(raw / "manifest.json", dict(manifest, complete=False))
    with pytest.raises(AssertionError, match="not complete"):
        exporter.source_plan(raw)
