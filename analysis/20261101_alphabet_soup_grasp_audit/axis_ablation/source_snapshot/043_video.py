"""Optional synchronized RGB recording for counterfactual replays."""

from __future__ import annotations

from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np

from ..policies.openpi.clientlib import image_tools as openpi_image_tools


VIDEO_VIEWS = ("combined", "main", "wrist")
_DATASET_IMAGE_SIZE = 224


class CounterfactualVideoRecorder:
    """Write one 10 Hz MP4, optionally combining both 4:3 RGB streams."""

    def __init__(
        self,
        model: mujoco.MjModel,
        path: Path,
        *,
        width: int = 640,
        height: int = 480,
        view: str = "combined",
    ) -> None:
        if width <= 0 or height <= 0 or width * 3 != height * 4:
            raise ValueError("Each RGB stream must use a positive 4:3 resolution")
        if width % 2 or height % 2:
            raise ValueError("H.264 yuv420p requires even dimensions")
        if view not in VIDEO_VIEWS:
            raise ValueError(f"Unknown video view: {view}")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.view = view
        self.main_renderer = (
            mujoco.Renderer(model, height=height, width=width)
            if view in ("combined", "main")
            else None
        )
        self.wrist_renderer = (
            mujoco.Renderer(model, height=height, width=width)
            if view in ("combined", "wrist")
            else None
        )
        output_width = width * (2 if view == "combined" else 1)
        self.writer = imageio_ffmpeg.write_frames(
            str(self.path),
            (output_width, height),
            fps=10,
            codec="libx264",
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            quality=8,
            macro_block_size=1,
        )
        self.writer.send(None)
        self.frame_count = 0

    @staticmethod
    def _render(
        renderer: mujoco.Renderer,
        data: mujoco.MjData,
        camera: str,
    ) -> np.ndarray:
        renderer.update_scene(data, camera=camera)
        return np.ascontiguousarray(renderer.render(), dtype=np.uint8)

    def capture(self, data: mujoco.MjData) -> None:
        if self.view == "main":
            assert self.main_renderer is not None
            image = self._render(self.main_renderer, data, "front_view")
        elif self.view == "wrist":
            assert self.wrist_renderer is not None
            image = self._render(self.wrist_renderer, data, "wrist_camera")
        else:
            assert self.main_renderer is not None
            assert self.wrist_renderer is not None
            image = np.concatenate((
                self._render(self.main_renderer, data, "front_view"),
                self._render(self.wrist_renderer, data, "wrist_camera"),
            ), axis=1)
        self.writer.send(image)
        self.frame_count += 1

    def close(self) -> None:
        self.writer.close()
        if self.main_renderer is not None:
            self.main_renderer.close()
        if self.wrist_renderer is not None:
            self.wrist_renderer.close()

    def __enter__(self) -> "CounterfactualVideoRecorder":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class CounterfactualDatasetVideoRecorder:
    """Write the two synchronized 224x224 streams used by collected datasets."""

    def __init__(
        self,
        model: mujoco.MjModel,
        directory: Path,
        *,
        width: int = 640,
        height: int = 480,
        frame_limit: int | None = None,
        source_frame_indices: tuple[int, ...] | None = None,
    ) -> None:
        if width <= 0 or height <= 0 or width * 3 != height * 4:
            raise ValueError("Source RGB streams must use a positive 4:3 resolution")
        if frame_limit is not None and frame_limit <= 0:
            raise ValueError("Dataset video frame limit must be positive")
        if source_frame_indices is not None:
            indices = tuple(int(index) for index in source_frame_indices)
            if (
                not indices
                or any(index < 0 for index in indices)
                or tuple(sorted(set(indices))) != indices
            ):
                raise ValueError(
                    "Dataset source frame indices must be non-empty, sorted, "
                    "unique and non-negative",
                )
            if frame_limit is not None and frame_limit != len(indices):
                raise ValueError(
                    "Dataset frame limit must equal the source-index count",
                )
        else:
            indices = None
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.frame_limit = frame_limit
        self.source_frame_indices = indices
        self._source_frame_index_set = (
            None if indices is None else set(indices)
        )
        self.source_frame_index = 0
        self.main_renderer = mujoco.Renderer(model, height=height, width=width)
        self.wrist_renderer = mujoco.Renderer(model, height=height, width=width)
        self.main_writer = self._writer(self.directory / "main_rgb.mp4")
        self.wrist_writer = self._writer(self.directory / "wrist_rgb.mp4")
        self.frame_count = 0

    @staticmethod
    def _writer(path: Path):
        writer = imageio_ffmpeg.write_frames(
            str(path),
            (_DATASET_IMAGE_SIZE, _DATASET_IMAGE_SIZE),
            fps=10,
            codec="libx264",
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            quality=8,
            macro_block_size=1,
        )
        writer.send(None)
        return writer

    @staticmethod
    def _dataset_frame(image: np.ndarray) -> np.ndarray:
        resized = openpi_image_tools.resize_with_pad(
            np.asarray(image, dtype=np.uint8),
            _DATASET_IMAGE_SIZE,
            _DATASET_IMAGE_SIZE,
        )
        return np.ascontiguousarray(resized, dtype=np.uint8)

    def capture(self, data: mujoco.MjData) -> None:
        source_index = self.source_frame_index
        self.source_frame_index += 1
        if (
            self._source_frame_index_set is not None
            and source_index not in self._source_frame_index_set
        ):
            return
        if self.frame_limit is not None and self.frame_count >= self.frame_limit:
            return
        main = CounterfactualVideoRecorder._render(
            self.main_renderer, data, "front_view",
        )
        wrist = CounterfactualVideoRecorder._render(
            self.wrist_renderer, data, "wrist_camera",
        )
        self.main_writer.send(self._dataset_frame(main))
        self.wrist_writer.send(self._dataset_frame(wrist))
        self.frame_count += 1

    def close(self) -> None:
        self.main_writer.close()
        self.wrist_writer.close()
        self.main_renderer.close()
        self.wrist_renderer.close()
