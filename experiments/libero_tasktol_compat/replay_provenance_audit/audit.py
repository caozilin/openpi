"""Read-only audit of OpenVLA replay provenance and original/replayed robot traces.

Original LIBERO observations are post-action; regenerated observations are pre-action.
Compare regenerated observation i+1 with original observation source_indices[i].
This is a correspondence comparison, not a counterfactual estimate of engine error.
"""
from collections import Counter
from pathlib import Path
import hashlib
import json

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def main():
    root = PROJECT / "data/libero_openvla"
    catalog = json.loads((PROJECT / "data/libero40_mask_annotation/20261019_viewer_10demos_v2/viewer_catalog.json").read_text())
    selected = {(d["suite"], d["task_name"], e["path"]): e["representative"]
                for d in catalog["datasets"] for e in d["episodes"]}
    rows, tasks, versions = [], [], Counter()
    for sidecar in sorted(root.rglob("*.metainfo.json")):
        meta = json.loads(sidecar.read_text())
        processed = sidecar.with_suffix("").with_suffix(".hdf5")
        with h5py.File(processed, "r") as new, h5py.File(meta["source_file"], "r") as raw:
            config = json.loads(new.attrs["configuration"])
            versions[json.dumps(config["versions"], sort_keys=True)] += 1
            tasks.append({k: meta[k] for k in ("suite", "task_name", "processed_demos", "successful_demos", "failed_demos")})
            for name in new["data"]:
                a, b = new["data"][name], raw["data"][name]
                idx = a["tasktol/source_step_indices"][()]
                raw_actions = b["actions"][()]
                actions_equal = np.array_equal(a["actions"][()], raw_actions[idx])
                q = a["tasktol/pre_action_states"][:, 1:8]
                raw_states = b["states"][()]
                raw_joints = b["obs/joint_states"][()]
                # Confirms post-action convention against the next saved state.
                alignment = float(np.max(np.abs(raw_joints[:-1] - raw_states[1:, 1:8])))
                p = a["obs/ee_pos"][()]
                rp = b["obs/ee_pos"][()]
                mm = np.linalg.norm(p[1:] - rp[idx[:-1]], axis=1) * 1000
                key = (meta["suite"], meta["task_name"], name)
                rows.append(dict(suite=meta["suite"], task_name=meta["task_name"], demo_id=name,
                    viewer_selected=key in selected, representative=selected.get(key, False),
                    retained_steps=len(idx), skipped_noops=len(raw_actions)-len(idx), actions_identical=actions_equal,
                    raw_post_action_alignment_max_joint_difference_rad=alignment,
                    same_source_pre_action_max_joint_difference_rad=float(np.max(np.abs(q-raw_states[idx,1:8]))),
                    aligned_post_action_ee_max_mm=float(mm.max()),
                    aligned_post_action_ee_median_mm=float(np.median(mm)),
                    aligned_post_action_comparisons=len(mm)))
    assert all(r["actions_identical"] for r in rows)
    assert sum(r["viewer_selected"] for r in rows) == 400
    assert sum(r["representative"] for r in rows) == 40
    groups = {}
    for label, subset in [("all_saved", rows), ("viewer_400", [r for r in rows if r["viewer_selected"]]),
                          ("representatives_40", [r for r in rows if r["representative"]])]:
        maxima = np.array([r["aligned_post_action_ee_max_mm"] for r in subset])
        groups[label] = dict(episodes=len(subset), episode_max_ee_difference_mm_min=float(maxima.min()),
            episode_max_ee_difference_mm_median=float(np.median(maxima)), episode_max_ee_difference_mm_max=float(maxima.max()),
            episodes_max_difference_over_5mm=int((maxima>5).sum()), episodes_max_difference_over_10mm=int((maxima>10).sum()))
    totals = {k: sum(t[k] for t in tasks) for k in ("processed_demos", "successful_demos", "failed_demos")}
    result = dict(totals=totals, runtime_versions=[dict(versions=json.loads(k), tasks=v) for k,v in versions.items()],
        observation_alignment="raw obs[j] is post-action; regenerated obs[i] is pre-action. Compare regenerated obs[i+1] to raw obs[source_indices[i]]; omit final action because no regenerated terminal observation is stored.",
        interpretation="Differences include settling, skipped simulation of no-ops, reset/controller history and runtime changes relative to original collection. They are not an isolated measurement of MuJoCo version error.",
        raw_post_action_alignment_max_joint_difference_rad=max(r["raw_post_action_alignment_max_joint_difference_rad"] for r in rows),
        groups=groups, tasks=tasks, episodes=rows,
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    example = next(r for r in rows if r["suite"]=="libero_10" and r["task_name"]=="LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket" and r["demo_id"]=="demo_10")
    worst_rep = max((r for r in rows if r["representative"]), key=lambda r:r["aligned_post_action_ee_max_mm"])
    canvas = Image.new("RGB", (560, 650), "#121821")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    for row_number, row in enumerate((example, worst_rep)):
        path = root / (row["suite"]+"_no_noops") / (row["task_name"]+"_demo.hdf5")
        with h5py.File(path,"r") as new, h5py.File(PROJECT / "third_party/libero/libero/datasets" / row["suite"] / path.name,"r") as raw:
            a,b = new["data"][row["demo_id"]],raw["data"][row["demo_id"]]
            idx = a["tasktol/source_step_indices"][()]
            delta = np.linalg.norm(a["obs/ee_pos"][1:]-b["obs/ee_pos"][()][idx[:-1]],axis=1)*1000
            i = int(np.argmax(delta)); j = int(idx[i]); top = 325*row_number
            draw.text((10,top+4),f"{row['suite']} / {row['demo_id']} | aligned EEF difference {delta[i]:.2f} mm",font=font,fill="white")
            draw.text((10,top+23),f"Original: after source action {j}",font=font,fill="white")
            draw.text((290,top+23),f"Regenerated: before retained action {i+1}",font=font,fill="white")
            for col, pixels in enumerate((b["obs/agentview_rgb"][j],a["obs/agentview_rgb"][i+1])):
                picture = Image.fromarray(pixels[::-1,::-1].copy()).resize((256,256))
                canvas.paste(picture,(10+280*col,top+45))
    canvas.save(OUT / "original_vs_regenerated.png")
    print(json.dumps(dict(totals=totals, runtime_versions=result["runtime_versions"], groups=groups,
        raw_alignment=result["raw_post_action_alignment_max_joint_difference_rad"], example=example), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
