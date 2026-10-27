"""Export pre-action video evidence around phases, solver and tracking events."""

import json
from pathlib import Path

import imageio.v2 as imageio
from PIL import Image


def main():
    root = Path(__file__).resolve().parent
    manifests = []
    for batch in ("native_nominal40", "native_tolerance5_four"):
        data = json.loads((root / "results" / batch / "batch_summary.json").read_text())
        for entry in data["results"]:
            folder = Path(entry["report"]).parent
            frames = json.loads((folder / "trajectory.json").read_text())["frames"]
            selected = {0, len(frames) - 1}
            for i in range(1, len(frames)):
                if (frames[i]["phase"], frames[i]["subtask_index"]) != (
                    frames[i - 1]["phase"],
                    frames[i - 1]["subtask_index"],
                ):
                    selected.update((i - 1, i))
            worst = max(range(len(frames)), key=lambda i: frames[i]["tracking_position_error_m"])
            selected.update((worst, min(worst + 1, len(frames) - 1)))
            selected.add(max(range(len(frames)), key=lambda i: frames[i]["optimizer"]["elapsed_ms"]))
            output = folder / "keyframes"
            output.mkdir(exist_ok=True)
            records = []
            with imageio.get_reader(folder / "main_rgb.mp4") as video:
                for i in sorted(selected):
                    image = video.get_data(i)
                    filename = f"{i:04d}_{frames[i]['phase']}.png"
                    Image.fromarray(image).save(output / filename)
                    records.append(
                        dict(
                            index=i,
                            video_frame_index=i,
                            time_s=i / 20,
                            phase=frames[i]["phase"],
                            subtask_index=frames[i]["subtask_index"],
                            observation_timing="pre_action",
                            image="keyframes/" + filename,
                            source_index=frames[i]["source_index"],
                            solver=frames[i]["optimizer"],
                            effective_mask=frames[i]["effective_mask"],
                        )
                    )
            (folder / "process_images.json").write_text(
                json.dumps(dict(view="agentview", records=records), ensure_ascii=False, indent=2) + "\n"
            )
            manifests.append(
                dict(
                    path=str(folder.relative_to(root)),
                    count=len(records),
                    manifest=str((folder / "process_images.json").relative_to(root)),
                )
            )
    (root / "process_images_manifest.json").write_text(
        json.dumps(
            dict(cases=len(manifests), images=sum(x["count"] for x in manifests), cases_detail=manifests), indent=2
        )
        + "\n"
    )
    print("Exported", sum(x["count"] for x in manifests), "process PNGs for", len(manifests), "native IPOPT cases")


if __name__ == "__main__":
    main()
