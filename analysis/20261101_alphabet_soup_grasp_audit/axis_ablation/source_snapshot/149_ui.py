from __future__ import annotations

from dataclasses import dataclass, replace
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

import numpy as np
from PIL import Image, ImageTk

from ..config import DOF_LABELS
from ..control.task_space.objective import DEFAULT_EMA_RELEASE_LOSS_WEIGHT
from ..tasks.phases import PhaseObservation
from ..geometry import (
    canonical_rotation_vector,
    continuous_rotation_vector,
    rotation_matrix,
)
from ..tasks.catalog import BLOCK_TO_PLATE_TASK
from ..tasks.planning.presets import PresetTaskExecutor
from ..benchmark.metrics import MultiRateEvaluationSummary
from ..tolerance.state import DEFAULT_STAGE_LIMITS_DEG, DEFAULT_STAGE_RANGED, ROTATION_STAGE_COUNT
from ..tolerance.frame import (
    TOLERANCE_FRAME_MODE_REALTIME_NOMINAL,
    TOLERANCE_FRAME_MODE_STAGE_ENDPOINT,
)
from ..tolerance.context import (
    ResolvedSubtaskTolerance,
    ResolvedTaskTolerance,
    effective_rotation_specific_axes,
    long_task_subtask_labels,
)
from ..simulation.scenes import (
    GEOMETRY_PLATE_BOX_UPRIGHT_SCENE,
    SceneManager,
)
from ..data.task_sets import (
    EXISTING_SCENE_GROUP,
    TOLERANCE_DISCRIMINATION_GROUP,
    TRAINING_TASK_GROUP,
    runtime_scene_from_choice,
    runtime_task_from_choice,
    scene_choices_for_group,
    scene_groups,
    tasks_for_scene_choice,
)
from ..simulation.robots import ROBOT_LABELS, ROBOT_UID_BY_LABEL, get_robot_spec
from ..simulation.layouts import (
    list_named_custom_layouts,
    named_custom_layout_path,
    validate_custom_layout_name,
)
from .window_layout import (
    CAMERAS_WINDOW_TITLE,
    MAIN_WINDOW_TITLE,
    VIEWER_WINDOW_TITLE,
    apply_saved_layout,
    load_layout,
    place_window_right_of,
)


@dataclass(frozen=True)
class LayoutEditCommand:
    kind: str
    x: float | None = None
    y: float | None = None
    yaw_deg: float | None = None
    name: str | None = None


@dataclass(frozen=True)
class InputSnapshot:
    pressed: set[str]
    gripper_closed: bool
    home_requested: bool
    reset_scene_requested: bool
    toggled_dofs: tuple[int, ...]
    active_dofs: np.ndarray
    active_stage: int
    tolerance_enabled: bool
    tolerance_frame_mode: str
    tolerance_target_position: np.ndarray
    rotation_tolerance_rad: np.ndarray
    rotation_tolerance_target: np.ndarray
    rotation_tolerance_frame: np.ndarray
    capture_tolerance_target_stage: int | None
    capture_tolerance_frame_stage: int | None
    rotation_in_base: bool
    teleop_speed_scale: float
    toggle_evaluation: bool
    toggle_trajectory: bool
    play_trajectory: bool
    objective_weights: np.ndarray
    loss_profile: str
    sqp_loss_weights: np.ndarray
    robot_uid: str
    scene_name: str
    scene_group: str
    scene_choice: str
    preset_task_name: str
    execute_preset_task: bool
    camera_x_offset: float
    camera_z_offset: float
    camera_pitch_offset_deg: float
    main_camera_x_offset: float
    main_camera_z_offset: float
    main_camera_pitch_offset_deg: float
    domain_seed: int
    show_reachability_map: bool
    domain_randomization_enabled: bool
    boundary_unreachable_placement: bool
    workspace_boundary_distance_min: float
    workspace_boundary_distance_max: float
    creation_mode: bool
    layout_edit_commands: tuple[LayoutEditCommand, ...]


class KeyboardPanel:
    """Keyboard input panel for the maintained stage-relative SQP controller."""

    PANEL_WIDTH = 1040
    PANEL_HEIGHT = 1010
    MINIMUM_PANEL_WIDTH = 1000
    MINIMUM_PANEL_HEIGHT = 820
    CHART_WIDTH = PANEL_WIDTH - 40
    CAMERA_PANEL_WIDTH = 840

    # Nominal main-camera calibration used by the UI and the MuJoCo worker.
    # Values are expressed as offsets from the scene's ``front_view`` camera.
    MAIN_CAMERA_DEFAULT_X_OFFSET_M = -0.020
    MAIN_CAMERA_DEFAULT_Z_OFFSET_M = 0.008
    MAIN_CAMERA_DEFAULT_PITCH_OFFSET_DEG = -4.0

    STAGE_NAMES = ("Pre-grasp", "Grasp", "Post-grasp", "Release")
    CONTROL_KEYS = set("wasdikqejuolcvh1237890") | {
        "bracketleft", "bracketright", "semicolon", "apostrophe", "comma", "period",
    }
    _CAMERA_STEPS = {
        "bracketleft": ("x", -0.002), "bracketright": ("x", 0.002),
        "semicolon": ("z", -0.002), "apostrophe": ("z", 0.002),
        "comma": ("pitch", -1.0), "period": ("pitch", 1.0),
    }
    PENALTY_TERMS = ("Pose tracking", "Incremental release", "Motion smoothness", "Manipulability", "Self collision")
    PENALTY_COLORS = ("#5dade2", "#48c9b0", "#f5b041", "#bb8fce", "#ec7063")
    WEIGHT_FIELDS = ((("P", 0), ("R", 1)), (("W", 6),), (("V", 2), ("J", 3)), (("W", 4),), (("W", 5),))
    EVALUATION_METRICS = (
        ("mean_position_error_m", "Pos", "m", 0.010),
        ("mean_rotation_error_rad", "Rot", "rad", 0.100),
        ("mean_joint_acceleration_rad_s2", "Acc", "rad/s²", 1.0),
        ("mean_joint_jerk_rad_s3", "Jerk", "rad/s³", 100.0),
        ("mean_manipulability", "Manip", "", 0.1),
        ("mean_joint_movement_rad", "Joint move", "rad/0.1s", 0.100),
    )
    EVALUATION_SERIES = (
        ("optimized_10hz", "Opt 10 Hz", "#5dade2"),
        ("actual_10hz", "Actual 10 Hz", "#f5b041"),
        ("actual_100hz", "Actual 100 Hz", "#58d68d"),
        ("filtered_100hz", "LP20 -> 100 Hz", "#bb8fce"),
        ("actual_1khz", "Raw actual 1 kHz", "#ec7063"),
    )
    LOSS_CHOICES = {
        "Δq² + EMA release": "delta_q_squared_ema_release",
    }
    TOLERANCE_FRAME_MODE_LABELS = {
        "Stage-end frame": TOLERANCE_FRAME_MODE_STAGE_ENDPOINT,
        "Real-time nominal frame": TOLERANCE_FRAME_MODE_REALTIME_NOMINAL,
    }
    def __init__(
        self,
        initial_weights: np.ndarray,
        camera_adjust_mode: str = "none",
        initial_scene: str = GEOMETRY_PLATE_BOX_UPRIGHT_SCENE,
        initial_scene_group: str = TRAINING_TASK_GROUP,
        initial_scene_choice: str | None = None,
        initial_robot: str = "panda",
        initial_tolerance_frame_rotvec_deg: np.ndarray | None = None,
        initial_target_rotvec_deg: np.ndarray | None = None,
        initial_target_position: np.ndarray | None = None,
        initial_loss_profile: str = "delta_q_squared_ema_release",
        initial_sqp_loss_weights: np.ndarray | None = None,
        initial_playback_scale: float = 5.0,
    ) -> None:
        self._pressed: set[str] = set()
        self._toggled_dofs: list[int] = []
        self._toggle_evaluation = self._toggle_trajectory = self._play_trajectory = False
        self._execute_preset_task = False
        self._reset_scene_requested = False
        self._layout_edit_commands: list[LayoutEditCommand] = []
        self._objective_weights = np.asarray(initial_weights, dtype=float).copy()
        if self._objective_weights.shape != (7,):
            raise ValueError("Expected seven objective weights.")
        if initial_loss_profile not in self.LOSS_CHOICES.values():
            raise ValueError(f"Unknown SQP loss profile: {initial_loss_profile}")
        self._initial_loss_profile = initial_loss_profile
        new_weights = (
            np.array((1.0, DEFAULT_EMA_RELEASE_LOSS_WEIGHT), dtype=float)
            if initial_sqp_loss_weights is None
            else np.asarray(initial_sqp_loss_weights, dtype=float)
        )
        if new_weights.shape != (2,) or not np.all(np.isfinite(new_weights)):
            raise ValueError("Expected two finite maintained SQP loss weights.")
        self._sqp_loss_weights = np.clip(new_weights, 0.0, 1000.0)
        self.active_dofs = np.ones(6, dtype=bool)
        self.active_stage = 0
        self._transition_source_stage = 0
        self._editing_stage_index = 0
        self._stage_ranged = DEFAULT_STAGE_RANGED.copy()
        self._stage_limits_deg = DEFAULT_STAGE_LIMITS_DEG.copy()
        initial_frame = np.degrees((np.pi, 0.0, 0.0)) if initial_tolerance_frame_rotvec_deg is None else np.asarray(initial_tolerance_frame_rotvec_deg, dtype=float)
        if initial_frame.shape != (3,) or not np.all(np.isfinite(initial_frame)):
            raise ValueError("Initial tolerance-frame rotvec must contain three finite values.")
        initial_frame = np.degrees(canonical_rotation_vector(
            rotation_matrix(np.radians(initial_frame)),
        ))
        initial_target = initial_frame.copy() if initial_target_rotvec_deg is None else np.asarray(initial_target_rotvec_deg, dtype=float)
        if initial_target.shape != (3,) or not np.all(np.isfinite(initial_target)):
            raise ValueError("Initial target rotvec must contain three finite values.")
        self._stage_target_rotvec_deg = np.tile(initial_target, (ROTATION_STAGE_COUNT, 1))
        initial_position = (
            np.zeros(3)
            if initial_target_position is None
            else np.asarray(initial_target_position, dtype=float)
        )
        if initial_position.shape != (3,) or not np.all(np.isfinite(initial_position)):
            raise ValueError("Initial stage-target position must contain three finite values.")
        self._stage_target_position_m = np.tile(
            initial_position, (ROTATION_STAGE_COUNT, 1),
        )
        self._stage_frame_rotvec_deg = np.tile(initial_frame, (ROTATION_STAGE_COUNT, 1))
        self._telemetry_rotvec_rad = np.radians(initial_target)
        self._capture_tolerance_target_stage: int | None = None
        self._capture_tolerance_frame_stage: int | None = None
        self.running, self.closed, self.home_requested, self.rotation_in_base = True, False, False, False
        initial_robot_spec = get_robot_spec(initial_robot)
        self.camera_x_offset = float(initial_robot_spec.wrist_camera_trim_pos[0])
        self.camera_z_offset = float(initial_robot_spec.wrist_camera_trim_pos[2])
        self.camera_pitch_offset_deg = float(
            initial_robot_spec.wrist_camera_trim_pitch_deg
        )
        self.main_camera_x_offset = self.MAIN_CAMERA_DEFAULT_X_OFFSET_M
        self.main_camera_z_offset = self.MAIN_CAMERA_DEFAULT_Z_OFFSET_M
        self.main_camera_pitch_offset_deg = self.MAIN_CAMERA_DEFAULT_PITCH_OFFSET_DEG
        if camera_adjust_mode not in {"none", "main", "wrist"}:
            raise ValueError(f"Unknown camera adjustment mode: {camera_adjust_mode}")
        self.camera_adjust_mode = camera_adjust_mode
        self._main_photo: ImageTk.PhotoImage | None = None
        self._wrist_photo: ImageTk.PhotoImage | None = None
        self._side_photo: ImageTk.PhotoImage | None = None

        self.root = tk.Tk()
        self._loss_profile = tk.StringVar(
            master=self.root, value=self._initial_loss_profile,
        )
        self._loss_profile_label = tk.StringVar(
            master=self.root,
            value=next(
                label for label, value in self.LOSS_CHOICES.items()
                if value == self._initial_loss_profile
            ),
        )
        self._tolerance_enabled = tk.BooleanVar(
            master=self.root, value=True,
        )
        self._tolerance_frame_mode = tk.StringVar(
            master=self.root,
            value=TOLERANCE_FRAME_MODE_REALTIME_NOMINAL,
        )
        self.root.title(MAIN_WINDOW_TITLE)
        robot_spec = initial_robot_spec
        if initial_scene not in SceneManager.UI_NAMES:
            raise ValueError(f"Unknown initial scene: {initial_scene}")
        if initial_scene_group not in scene_groups():
            raise ValueError(f"Unknown initial scene group: {initial_scene_group}")
        choices = scene_choices_for_group(initial_scene_group)
        if initial_scene_choice not in choices:
            initial_scene_choice = next(
                (
                    choice for choice in choices
                    if runtime_scene_from_choice(initial_scene_group, choice) == initial_scene
                ),
                None,
            )
        if initial_scene_choice is None:
            initial_scene_group = EXISTING_SCENE_GROUP
            initial_scene_choice = initial_scene
        self._scene_name = tk.StringVar(master=self.root, value=initial_scene)
        self._scene_group = tk.StringVar(master=self.root, value=initial_scene_group)
        self._scene_choice = tk.StringVar(master=self.root, value=initial_scene_choice)
        self._robot_label = tk.StringVar(master=self.root, value=robot_spec.label)
        if initial_scene_group in (
            TRAINING_TASK_GROUP,
            TOLERANCE_DISCRIMINATION_GROUP,
        ):
            initial_tasks = tuple(
                task.task_id
                for task in tasks_for_scene_choice(initial_scene_group, initial_scene_choice)
            )
        elif initial_scene_group == EXISTING_SCENE_GROUP:
            initial_tasks = PresetTaskExecutor.task_names(initial_scene)
        else:
            initial_tasks = ()
        self._preset_task_name = tk.StringVar(
            master=self.root,
            value=initial_tasks[0] if initial_tasks else "(no preset task)",
        )
        self._domain_seed = tk.StringVar(master=self.root, value="-1")
        self._domain_seed_value = -1
        self._show_reachability_map = tk.BooleanVar(master=self.root, value=False)
        self._domain_randomization_enabled = tk.BooleanVar(
            master=self.root, value=False,
        )
        self._boundary_unreachable_placement = tk.BooleanVar(
            master=self.root, value=False,
        )
        self._workspace_boundary_distance_min = tk.StringVar(
            master=self.root, value="1.00",
        )
        self._workspace_boundary_distance_max = tk.StringVar(
            master=self.root, value="1.00",
        )
        self._workspace_boundary_distance_min_value = 1.0
        self._workspace_boundary_distance_max_value = 1.0
        self._reachability_status = tk.StringVar(master=self.root, value="map hidden")
        self._creation_mode = tk.BooleanVar(master=self.root, value=False)
        self._layout_selection = tk.StringVar(master=self.root, value="no selection")
        self._layout_x = tk.StringVar(master=self.root, value="0.0000")
        self._layout_y = tk.StringVar(master=self.root, value="0.0000")
        self._layout_yaw = tk.StringVar(master=self.root, value="0.00")
        self._layout_library_name = tk.StringVar(master=self.root, value="")
        self._teleop_speed_scale = tk.StringVar(
            master=self.root, value=f"{float(initial_playback_scale):g}",
        )
        self._teleop_speed_scale_value = float(initial_playback_scale)
        self._stage_editor_name = tk.StringVar(master=self.root, value=self.STAGE_NAMES[0])
        self._gripper_phase_detail = "waiting for first 10 Hz observation"
        self._task_tolerance_context: ResolvedTaskTolerance | None = None
        self._manual_stage_settings: dict[tuple, dict[str, np.ndarray]] = {}
        self._tolerance_subtask_index = 0
        self._tolerance_subtask_count = 1
        self._tolerance_subtask_label = "Subtask 1"
        self._stage_ranged_vars = [tk.BooleanVar(master=self.root, value=False) for _ in range(3)]
        self._stage_limit_vars = [
            tk.StringVar(master=self.root, value=f"{value:g}")
            for value in self._stage_limits_deg[0]
        ]
        self._stage_target_vars = [
            tk.StringVar(master=self.root, value=f"{value:.2f}")
            for value in self._stage_target_rotvec_deg[0]
        ]
        self._stage_frame_vars = [
            tk.StringVar(master=self.root, value=f"{value:.2f}")
            for value in self._stage_frame_rotvec_deg[0]
        ]
        self.root.geometry(
            f"{self.PANEL_WIDTH}x{self.PANEL_HEIGHT}+10+10"
        )
        self.root.minsize(
            self.MINIMUM_PANEL_WIDTH,
            self.MINIMUM_PANEL_HEIGHT,
        )
        self.root.resizable(True, True)
        self.root.configure(bg="#18202a")
        self.root.columnconfigure(0, weight=1)
        self._title_text = tk.StringVar(master=self.root, value=f"{robot_spec.label} — SQP Teleoperation")
        tk.Label(self.root, textvariable=self._title_text, font=("Segoe UI", 14, "bold"), fg="#85c1e9", bg="#18202a").grid(row=0, column=0, pady=(6, 0))
        tk.Label(self.root, text="Keyboard teleoperation | UI configuration", font=("Segoe UI", 8), fg="#aab7c4", bg="#18202a").grid(row=1, column=0, pady=(0, 4))

        controls = tk.Frame(self.root, bg="#24313f", padx=9, pady=5)
        controls.grid(row=2, column=0, sticky="ew", padx=12)
        tk.Label(controls, text="Move W/S:X  A/D:Y  I/K:Z   |   Rotate Q/E:Rx  U/O:Ry  J/L:Rz   |   C/V gripper  H home", font=("Consolas", 9), fg="#e8edf2", bg="#24313f").grid(row=0, column=0, columnspan=2, sticky="w")
        tk.Label(controls, text="1-3 translation specific/free   |   7 tool/base rotation", font=("Consolas", 8), fg="#b7c8d8", bg="#24313f").grid(row=1, column=0, sticky="w", pady=(3, 0))
        scene_controls = tk.Frame(controls, bg="#24313f")
        # The second-level scene selector makes this row wider than the old
        # single scene control. Give it the full panel width so the speed
        # multiplier remains visible at the minimum window size.
        scene_controls.grid(
            row=2, column=0, columnspan=2,
            sticky="ew", pady=(5, 0),
        )
        tk.Label(scene_controls, text="Scene set:", font=("Segoe UI", 9), fg="#aab7c4", bg="#24313f").pack(side="left")
        self._scene_group_selector = ttk.Combobox(
            scene_controls,
            textvariable=self._scene_group,
            values=scene_groups(),
            state="readonly",
            width=12,
        )
        self._scene_group_selector.pack(side="left", padx=(4, 8))
        self._scene_group_selector.bind("<<ComboboxSelected>>", self._on_scene_group_selected)
        tk.Label(scene_controls, text="Scene:", font=("Segoe UI", 9), fg="#aab7c4", bg="#24313f").pack(side="left")
        self._scene_selector = ttk.Combobox(
            scene_controls,
            textvariable=self._scene_choice,
            values=scene_choices_for_group(initial_scene_group),
            state="readonly",
            width=18,
        )
        self._scene_selector.pack(side="left", padx=(4, 0))
        self._scene_selector.bind("<<ComboboxSelected>>", self._on_scene_selected)
        tk.Label(scene_controls, text="Layout seed (-1=base):", font=("Segoe UI", 9), fg="#aab7c4", bg="#24313f").pack(side="left", padx=(14, 4))
        self._domain_seed_entry = tk.Entry(scene_controls, textvariable=self._domain_seed, width=10, justify="right", font=("Consolas", 9))
        self._domain_seed_entry.pack(side="left")
        self._domain_seed_entry.bind("<KeyPress>", self._weight_entry_keypress)
        self._domain_seed_entry.bind("<Return>", lambda _event: self.root.focus_force())
        tk.Label(
            scene_controls,
            text="Simulation speed ×:",
            font=("Segoe UI", 9),
            fg="#aab7c4",
            bg="#24313f",
        ).pack(side="left", padx=(12, 4))
        self._teleop_speed_entry = tk.Entry(
            scene_controls, textvariable=self._teleop_speed_scale,
            width=6, justify="right", font=("Consolas", 9),
        )
        self._teleop_speed_entry.pack(side="left")
        self._teleop_speed_entry.bind("<KeyPress>", self._weight_entry_keypress)
        self._teleop_speed_entry.bind("<Return>", lambda _event: self.root.focus_force())

        creation_row = tk.Frame(controls, bg="#24313f")
        creation_row.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        self._creation_toggle = tk.Checkbutton(
            creation_row, text="CREATION MODE", variable=self._creation_mode,
            command=self._on_creation_mode_changed,
            font=("Segoe UI", 9, "bold"), fg="#f5b041", bg="#24313f",
            activebackground="#24313f", activeforeground="#f5b041",
            selectcolor="#18202a",
        )
        self._creation_toggle.pack(side="left")
        tk.Label(
            creation_row, textvariable=self._layout_selection,
            font=("Consolas", 8), fg="#85c1e9", bg="#24313f", width=25,
            anchor="w",
        ).pack(side="left", padx=(8, 4))
        self._layout_entries: list[tk.Entry] = []
        for label, variable, width in (
            ("X", self._layout_x, 8), ("Y", self._layout_y, 8),
            ("Yaw°", self._layout_yaw, 7),
        ):
            tk.Label(
                creation_row, text=label, font=("Segoe UI", 8),
                fg="#aab7c4", bg="#24313f",
            ).pack(side="left", padx=(4, 2))
            entry = tk.Entry(
                creation_row, textvariable=variable, width=width,
                justify="right", font=("Consolas", 8), state="disabled",
            )
            entry.pack(side="left")
            entry.bind("<Return>", lambda _event: self._queue_layout_numeric_apply())
            self._layout_entries.append(entry)
        self._layout_apply_button = tk.Button(
            creation_row, text="APPLY", width=6,
            command=self._queue_layout_numeric_apply, state="disabled",
        )
        self._layout_apply_button.pack(side="left", padx=(5, 2))
        tk.Button(
            creation_row, text="RESET", width=7,
            command=lambda: self._queue_layout_command("reset"),
        ).pack(side="left", padx=(2, 0))
        self._layout_library_selector = ttk.Combobox(
            creation_row, textvariable=self._layout_library_name,
            values=(), state="readonly", width=15,
        )
        self._layout_library_selector.pack(side="left", padx=(5, 2))
        tk.Button(
            creation_row, text="IMPORT", width=7,
            command=self._request_layout_import,
        ).pack(side="left", padx=(2, 0))
        tk.Button(
            creation_row, text="EXPORT", width=7,
            command=self._request_layout_export,
        ).pack(side="left", padx=(2, 0))
        self.refresh_layout_names()

        actions = tk.Frame(self.root, bg="#18202a")
        actions.grid(row=3, column=0, sticky="ew", padx=12, pady=4)
        actions.columnconfigure(0, weight=1)
        self.status = tk.StringVar()
        controller_row = tk.Frame(actions, bg="#18202a")
        controller_row.grid(row=0, column=0, sticky="w")
        tk.Label(controller_row, text="Optimizer: IPOPT", font=("Segoe UI", 9), fg="#aab7c4", bg="#18202a").pack(side="left", padx=(7, 14))
        tk.Label(controller_row, text="Loss:", font=("Segoe UI", 9), fg="#aab7c4", bg="#18202a").pack(side="left", padx=(0, 5))
        self._loss_selector = ttk.Combobox(
            controller_row,
            textvariable=self._loss_profile_label,
            values=tuple(self.LOSS_CHOICES),
            state="readonly",
            width=19,
        )
        self._loss_selector.pack(side="left")
        self._loss_selector.bind(
            "<<ComboboxSelected>>", self._on_loss_profile_selected,
        )
        tk.Label(controller_row, text="Robot:", font=("Segoe UI", 9), fg="#aab7c4", bg="#18202a").pack(side="left", padx=(14, 5))
        self._robot_selector = ttk.Combobox(
            controller_row, textvariable=self._robot_label,
            values=ROBOT_LABELS, state="readonly", width=18,
        )
        self._robot_selector.pack(side="left")
        self._robot_selector.bind("<<ComboboxSelected>>", self._on_robot_selected)
        tk.Label(controller_row, text="Task:", font=("Segoe UI", 9), fg="#aab7c4", bg="#18202a").pack(side="left", padx=(14, 5))
        self._preset_task_selector = ttk.Combobox(
            controller_row, textvariable=self._preset_task_name,
            values=initial_tasks if initial_tasks else ("(no preset task)",),
            state="readonly", width=19,
        )
        self._preset_task_selector.pack(side="left")
        self._preset_task_selector.bind("<<ComboboxSelected>>", lambda _event: self.root.focus_force())
        self._preset_task_button = tk.Button(controller_row, text="EXECUTE", width=8, command=self._request_preset_task)
        self._preset_task_button.pack(side="left", padx=(6, 0))
        self._preset_task_status = tk.StringVar(value="idle")
        task_status_row = tk.Frame(actions, bg="#18202a")
        task_status_row.grid(row=1, column=0, sticky="ew", pady=(3, 0))
        tk.Button(
            task_status_row,
            text="RESET SCENE + HOME",
            command=self._request_scene_reset,
            font=("Segoe UI", 8, "bold"),
        ).pack(side="right", padx=(8, 7))
        tk.Label(
            task_status_row, textvariable=self._reachability_status,
            font=("Segoe UI", 8), fg="#aab7c4", bg="#18202a",
        ).pack(side="right", padx=(5, 2))
        tk.Checkbutton(
            task_status_row,
            text="3D boundary surface",
            variable=self._show_reachability_map,
            font=("Segoe UI", 8),
            fg="#85c1e9",
            bg="#18202a",
            activebackground="#18202a",
            activeforeground="#85c1e9",
            selectcolor="#24313f",
        ).pack(side="right", padx=(8, 0))
        self._workspace_boundary_max_entry = tk.Entry(
            task_status_row,
            textvariable=self._workspace_boundary_distance_max,
            width=5, justify="right", font=("Consolas", 8),
        )
        self._workspace_boundary_max_entry.pack(side="right", padx=(2, 0))
        tk.Label(
            task_status_row, text="..", font=("Segoe UI", 8),
            fg="#aab7c4", bg="#18202a",
        ).pack(side="right")
        self._workspace_boundary_min_entry = tk.Entry(
            task_status_row,
            textvariable=self._workspace_boundary_distance_min,
            width=5, justify="right", font=("Consolas", 8),
        )
        self._workspace_boundary_min_entry.pack(side="right", padx=(3, 0))
        tk.Label(
            task_status_row, text="factor", font=("Segoe UI", 8),
            fg="#f5b041", bg="#18202a",
        ).pack(side="right", padx=(7, 0))
        tk.Checkbutton(
            task_status_row,
            text="Workspace boundary",
            variable=self._boundary_unreachable_placement,
            font=("Segoe UI", 8),
            fg="#f5b041",
            bg="#18202a",
            activebackground="#18202a",
            activeforeground="#f5b041",
            selectcolor="#24313f",
        ).pack(side="right", padx=(8, 0))
        tk.Checkbutton(
            task_status_row,
            text="Domain randomization",
            variable=self._domain_randomization_enabled,
            font=("Segoe UI", 8),
            fg="#aab7c4",
            bg="#18202a",
            activebackground="#18202a",
            activeforeground="#aab7c4",
            selectcolor="#24313f",
        ).pack(side="right", padx=(8, 0))
        tk.Label(
            task_status_row, text="Task execution:", font=("Segoe UI", 8, "bold"),
            fg="#85c1e9", bg="#18202a",
        ).pack(side="left", padx=(7, 5))
        tk.Label(
            task_status_row, textvariable=self._preset_task_status,
            font=("Segoe UI", 8), justify="left", anchor="w",
            wraplength=self.PANEL_WIDTH - 130,
            fg="#58d68d", bg="#18202a",
        ).pack(side="left", fill="x", expand=True)

        stage_row = tk.Frame(actions, bg="#24313f", padx=8, pady=3)
        stage_row.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        frame_row = tk.Frame(stage_row, bg="#24313f")
        frame_row.pack(anchor="w")
        tk.Label(frame_row, text="Edit:", font=("Segoe UI", 9, "bold"), fg="#85c1e9", bg="#24313f").pack(side="left")
        self._stage_editor = ttk.Combobox(
            frame_row, textvariable=self._stage_editor_name,
            # Grasp and Release are strict runtime states and therefore have
            # no independently editable ranged-tolerance configuration.
            values=(self.STAGE_NAMES[0], self.STAGE_NAMES[2]), state="readonly", width=14,
        )
        self._stage_editor.pack(side="left", padx=(5, 14))
        self._stage_editor.bind("<<ComboboxSelected>>", self._on_stage_editor_selected)
        tk.Label(frame_row, text="Target RV:", font=("Segoe UI", 8), fg="#aab7c4", bg="#24313f").pack(side="left")
        self._stage_target_entries: list[tk.Entry] = []
        for variable in self._stage_target_vars:
            entry = tk.Entry(frame_row, textvariable=variable, width=6, justify="right", font=("Consolas", 8))
            entry.pack(side="left", padx=(3, 0))
            entry.bind("<Return>", self._on_stage_target_commit)
            entry.bind("<FocusOut>", self._on_stage_target_commit)
            entry.bind("<KeyPress>", self._weight_entry_keypress)
            self._stage_target_entries.append(entry)
        tk.Button(frame_row, text="SET EE", command=self._request_tolerance_target_capture, font=("Segoe UI", 8)).pack(side="left", padx=(4, 10))
        tk.Label(frame_row, text="TolF RV:", font=("Segoe UI", 8), fg="#aab7c4", bg="#24313f").pack(side="left")
        self._stage_frame_entries: list[tk.Entry] = []
        for variable in self._stage_frame_vars:
            entry = tk.Entry(frame_row, textvariable=variable, width=6, justify="right", font=("Consolas", 8))
            entry.pack(side="left", padx=(3, 0))
            entry.bind("<Return>", self._on_stage_frame_commit)
            entry.bind("<FocusOut>", self._on_stage_frame_commit)
            entry.bind("<KeyPress>", self._weight_entry_keypress)
            self._stage_frame_entries.append(entry)
        tk.Button(frame_row, text="SET EE", command=self._request_tolerance_frame_capture, font=("Segoe UI", 8)).pack(side="left", padx=(4, 0))

        limits_row = tk.Frame(stage_row, bg="#24313f")
        limits_row.pack(anchor="w", pady=(3, 0))
        tk.Checkbutton(
            limits_row,
            text="Enable tolerance",
            variable=self._tolerance_enabled,
            command=self._on_tolerance_enabled_changed,
            bg="#24313f",
            fg="#58d68d",
            selectcolor="#18202a",
            activebackground="#24313f",
            activeforeground="#58d68d",
            font=("Segoe UI", 8, "bold"),
        ).pack(side="left", padx=(0, 12))
        tk.Label(limits_row, text="Fixed XYZ/RPY ranges:", font=("Segoe UI", 8), fg="#aab7c4", bg="#24313f").pack(side="left", padx=(0, 8))
        self._stage_limit_entries: list[tk.Entry] = []
        for axis, label in enumerate(("Rx", "Ry", "Rz")):
            tk.Checkbutton(
                limits_row, text=f"{label} ranged", variable=self._stage_ranged_vars[axis],
                command=self._on_stage_config_changed, bg="#24313f", fg="#e8edf2",
                selectcolor="#18202a", activebackground="#24313f", activeforeground="#e8edf2",
            ).pack(side="left", padx=(5, 2))
            entry = tk.Entry(
                limits_row, textvariable=self._stage_limit_vars[axis], width=5,
                justify="right", font=("Consolas", 9),
            )
            entry.pack(side="left")
            self._stage_limit_entries.append(entry)
            entry.bind("<Return>", self._on_stage_limit_commit)
            entry.bind("<FocusOut>", self._on_stage_limit_commit)
            entry.bind("<KeyPress>", self._weight_entry_keypress)
            tk.Label(limits_row, text="deg", font=("Segoe UI", 8), fg="#aab7c4", bg="#24313f").pack(side="left", padx=(2, 8))

        frame_mode_row = tk.Frame(stage_row, bg="#24313f")
        frame_mode_row.pack(anchor="w", pady=(3, 0))
        tk.Label(
            frame_mode_row,
            text="Tolerance frame during release:",
            font=("Segoe UI", 8),
            fg="#aab7c4",
            bg="#24313f",
        ).pack(side="left", padx=(0, 8))
        self._tolerance_frame_mode_buttons: list[tk.Radiobutton] = []
        for label, mode in self.TOLERANCE_FRAME_MODE_LABELS.items():
            button = tk.Radiobutton(
                frame_mode_row,
                text=label,
                variable=self._tolerance_frame_mode,
                value=mode,
                command=self._refresh_task_state,
                bg="#24313f",
                fg="#e8edf2",
                selectcolor="#18202a",
                activebackground="#24313f",
                activeforeground="#e8edf2",
                font=("Segoe UI", 8),
            )
            button.pack(side="left", padx=(0, 8))
            self._tolerance_frame_mode_buttons.append(button)

        summary = tk.Frame(self.root, bg="#203042", padx=9, pady=5)
        summary.grid(row=4, column=0, sticky="ew", padx=12)
        summary.columnconfigure(0, weight=1)
        self.task_state, self.ik_state = tk.StringVar(), tk.StringVar(value="SQP: waiting for simulation")
        tk.Label(summary, textvariable=self.task_state, font=("Consolas", 8), justify="left", fg="#58d68d", bg="#203042").grid(row=0, column=0, sticky="w")
        self._ik_state_label = tk.Label(summary, textvariable=self.ik_state, font=("Consolas", 9), justify="right", fg="#d7dee7", bg="#203042")
        self._ik_state_label.grid(row=0, column=1, sticky="e", padx=(30, 0))
        self.record_state = tk.StringVar(value="Metrics: idle (8)    Trajectory: idle (9 record, 0 replay)")
        tk.Label(summary, textvariable=self.record_state, font=("Consolas", 8), justify="left", fg="#f5b041", bg="#203042").grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 0))

        optimisation = tk.Frame(self.root, bg="#18202a")
        optimisation.grid(row=5, column=0, sticky="ew", padx=12, pady=(5, 1))
        tk.Label(optimisation, text="Optimisation penalties", font=("Segoe UI", 10, "bold"), fg="#85c1e9", bg="#18202a").pack(anchor="w")
        self.chart = tk.Canvas(optimisation, width=self.CHART_WIDTH, height=143, bg="#101820", highlightthickness=0)
        self.chart.pack(pady=(2, 0))
        self._weight_vars = [tk.StringVar(value=f"{value:g}") for value in self._objective_weights]
        self._weight_entries = [tk.Entry(self.chart, textvariable=value, width=5, justify="right", font=("Consolas", 8)) for value in self._weight_vars]
        for entry in self._weight_entries:
            entry.bind("<KeyPress>", self._weight_entry_keypress)
            entry.bind("<Return>", lambda _event: self.root.focus_force())
        loss_weights_row = tk.Frame(optimisation, bg="#18202a")
        loss_weights_row.pack(anchor="w", pady=(3, 0))
        tk.Label(
            loss_weights_row,
            text="New loss weights:",
            font=("Segoe UI", 8, "bold"),
            fg="#f5b041",
            bg="#18202a",
        ).pack(side="left", padx=(7, 5))
        self._sqp_loss_weight_vars = [
            tk.StringVar(value=f"{value:g}") for value in self._sqp_loss_weights
        ]
        self._sqp_loss_weight_entries = []
        for label, variable in zip(
            ("Δq²", "EMA release"),
            self._sqp_loss_weight_vars,
            strict=True,
        ):
            tk.Label(
                loss_weights_row, text=label, font=("Consolas", 8),
                fg="#aab7c4", bg="#18202a",
            ).pack(side="left", padx=(5, 2))
            entry = tk.Entry(
                loss_weights_row, textvariable=variable, width=6,
                justify="right", font=("Consolas", 8),
            )
            entry.pack(side="left", padx=(0, 4))
            entry.bind("<KeyPress>", self._weight_entry_keypress)
            entry.bind("<Return>", lambda _event: self.root.focus_force())
            self._sqp_loss_weight_entries.append(entry)
        self._update_loss_profile_widget_state()
        self._nominal_action_ema = np.full(3, np.nan)
        self._ema_release_state = tk.StringVar(
            value="EMA  Rx --  Ry --  Rz --",
        )
        tk.Label(
            loss_weights_row,
            textvariable=self._ema_release_state,
            font=("Consolas", 8, "bold"),
            fg="#58d68d",
            bg="#18202a",
        ).pack(side="left", padx=(12, 0))
        # Read-only live value of 0.05 + 0.95*a^2.  It is redrawn with the
        # penalty chart every telemetry cycle, but is deliberately not an
        # Entry because it is optimizer output rather than a user setting.
        self._release_loss_multipliers = np.full(3, np.nan)
        self._draw_penalty_chart({})

        telemetry = tk.Frame(self.root, bg="#24313f", padx=9, pady=5)
        telemetry.grid(row=6, column=0, sticky="ew", padx=12, pady=(5, 0))
        self.joints, self.pose = tk.StringVar(value="Waiting for simulation..."), tk.StringVar(value="Waiting for simulation...")
        tk.Label(telemetry, text="Joint angles (deg)", font=("Segoe UI", 9, "bold"), fg="#85c1e9", bg="#24313f").grid(row=0, column=0, sticky="w")
        tk.Label(telemetry, textvariable=self.joints, font=("Consolas", 8), justify="left", fg="#e8edf2", bg="#24313f").grid(row=1, column=0, sticky="w", pady=(2, 0))
        tk.Label(telemetry, text="libfranka O_T_EE (base rotvec)", font=("Segoe UI", 9, "bold"), fg="#85c1e9", bg="#24313f").grid(row=0, column=1, sticky="w", padx=(24, 0))
        tk.Label(telemetry, textvariable=self.pose, font=("Consolas", 8), justify="left", fg="#e8edf2", bg="#24313f").grid(row=1, column=1, sticky="w", padx=(24, 0), pady=(2, 0))
        self.sigma_min_state = tk.StringVar(value="Current σmin (velocity-normalized, full J): --")
        tk.Label(
            telemetry,
            textvariable=self.sigma_min_state,
            font=("Consolas", 9, "bold"),
            fg="#85c1e9",
            bg="#24313f",
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(3, 0))
        tolerance_panel = tk.Frame(telemetry, bg="#24313f")
        tolerance_panel.grid(row=3, column=0, columnspan=2, sticky="w", pady=(3, 0))
        self.rotation_tolerance_state = tk.StringVar(
            value="Tolerance XYZ/RPY uses fixed stage axes: Rz(yaw) Ry(pitch) Rx(roll)",
        )
        tk.Label(
            tolerance_panel,
            textvariable=self.rotation_tolerance_state,
            font=("Consolas", 8),
            justify="left",
            anchor="w",
            wraplength=self.PANEL_WIDTH - 44,
            fg="#f5b041",
            bg="#24313f",
        ).pack(anchor="w")

        errors = tk.Frame(self.root, bg="#18202a")
        errors.grid(row=7, column=0, sticky="ew", padx=12, pady=(4, 0))
        tk.Label(errors, text="Goal error (bias-adjusted optimized target / nominal reference)", font=("Segoe UI", 10, "bold"), fg="#85c1e9", bg="#18202a").pack(anchor="w")
        self.error_state = tk.StringVar(value="Waiting for simulation...")
        tk.Label(errors, textvariable=self.error_state, font=("Consolas", 8), fg="#d7dee7", bg="#18202a").pack(anchor="w", pady=(0, 1))
        self.error_chart = tk.Canvas(errors, width=self.CHART_WIDTH, height=84, bg="#101820", highlightthickness=0); self.error_chart.pack()

        evaluation = tk.Frame(self.root, bg="#24313f", padx=9, pady=5)
        evaluation.grid(row=8, column=0, sticky="ew", padx=12, pady=(5, 0))
        tk.Label(evaluation, text="Evaluation (three aligned trajectory domains / 10 Hz display)", font=("Segoe UI", 9, "bold"), fg="#85c1e9", bg="#24313f").pack(anchor="w")
        self.evaluation_state = tk.StringVar(value="Execute a preset to record automatically; press 8 to stop early")
        tk.Label(
            evaluation,
            textvariable=self.evaluation_state,
            font=("Consolas", 8),
            justify="left",
            anchor="w",
            wraplength=self.PANEL_WIDTH - 44,
            fg="#e8edf2",
            bg="#24313f",
        ).pack(anchor="w", pady=(2, 2))
        self.evaluation_chart = tk.Canvas(
            evaluation,
            width=self.CHART_WIDTH,
            height=154,
            bg="#101820",
            highlightthickness=0,
        )
        self.evaluation_chart.pack(anchor="w")
        self._draw_evaluation_chart(MultiRateEvaluationSummary())

        self.root.bind_all("<KeyPress>", self._on_press); self.root.bind_all("<KeyRelease>", self._on_release)
        self.root.protocol("WM_DELETE_WINDOW", self.close); self._refresh_task_state(); self.root.focus_force()
        # Start beside the operation panel; after mapping, the callback below
        # corrects for the native title-bar width and makes the edges flush.
        camera_x = 10 + self.PANEL_WIDTH
        self.wrist = tk.Toplevel(self.root); self.wrist.title(CAMERAS_WINDOW_TITLE); self.wrist.geometry(f"{self.CAMERA_PANEL_WIDTH}x390+{camera_x}+10"); self.wrist.configure(bg="#18202a")
        camera_grid = tk.Frame(self.wrist, bg="#18202a"); camera_grid.pack(padx=8, pady=(7, 8))
        tk.Label(camera_grid, text="Wrist RGB", font=("Segoe UI", 10, "bold"), fg="#85c1e9", bg="#18202a").grid(row=0, column=0, pady=(0, 3))
        tk.Label(camera_grid, text="Side RGB", font=("Segoe UI", 10, "bold"), fg="#85c1e9", bg="#18202a").grid(row=0, column=1, pady=(0, 0))
        self.wrist_camera_state = tk.StringVar()
        self._refresh_camera_state()
        tk.Label(
            camera_grid, textvariable=self.wrist_camera_state,
            font=("Consolas", 8), justify="center", anchor="center",
            width=48, wraplength=390,
            fg="#aab7c4", bg="#18202a",
        ).grid(row=1, column=0, pady=(0, 3))
        self.side_camera_state = tk.StringVar(value="X +0.400  Y +0.910  Z +0.920 m  P -2°")
        tk.Label(
            camera_grid, textvariable=self.side_camera_state,
            font=("Consolas", 8), justify="center", anchor="center",
            width=48, wraplength=390,
            fg="#aab7c4", bg="#18202a",
        ).grid(row=1, column=1, pady=(0, 3))
        self.wrist_image = tk.Label(camera_grid, bg="#101820"); self.wrist_image.grid(row=2, column=0, padx=(0, 8))
        self.side_image = tk.Label(camera_grid, bg="#101820"); self.side_image.grid(row=2, column=1)
        self.viewer = tk.Toplevel(self.root)
        self.viewer.title(VIEWER_WINDOW_TITLE)
        self.viewer.geometry(f"1200x900+{camera_x}+430")
        self.viewer.configure(bg="#101820")
        self.viewer.protocol("WM_DELETE_WINDOW", self.close)
        self.main_image = tk.Label(self.viewer, bg="#101820")
        self.main_image.pack(fill="both", expand=True)
        self.main_image.bind("<ButtonPress-1>", self._on_layout_pointer_down)
        self.main_image.bind("<B1-Motion>", self._on_layout_pointer_drag)
        self.main_image.bind("<ButtonRelease-1>", self._on_layout_pointer_up)
        self.main_image.bind("<MouseWheel>", self._on_layout_mouse_wheel)
        self.main_image.bind("<Button-4>", lambda event: self._on_layout_wheel_step(event, 1))
        self.main_image.bind("<Button-5>", lambda event: self._on_layout_wheel_step(event, -1))
        self._refresh_camera_state()
        self.root.update_idletasks()
        self.root.after_idle(self._place_initial_windows)

    def _place_initial_windows(self) -> None:
        """Restore a captured layout, or make the camera panel flush to the right."""
        saved_layout = load_layout()
        if "main" in saved_layout or "cameras" in saved_layout or "viewer" in saved_layout:
            apply_saved_layout(
                self.root.winfo_id(),
                "main",
                minimum_width=self.MINIMUM_PANEL_WIDTH,
                minimum_height=self.MINIMUM_PANEL_HEIGHT,
            )
            apply_saved_layout(self.wrist.winfo_id(), "cameras")
            apply_saved_layout(self.viewer.winfo_id(), "viewer")
        else:
            place_window_right_of(self.root.winfo_id(), self.wrist.winfo_id())

    def main_camera_frame(self, main_rgb: np.ndarray) -> None:
        """Update the fixed front-camera preview."""
        self._main_photo = ImageTk.PhotoImage(Image.fromarray(main_rgb))

        self.main_image.configure(image=self._main_photo)

    def _image_coordinates(self, event: tk.Event[tk.Misc]) -> tuple[float, float] | None:
        if self._main_photo is None:
            return None
        width, height = self._main_photo.width(), self._main_photo.height()
        left = 0.5 * (self.main_image.winfo_width() - width)
        top = 0.5 * (self.main_image.winfo_height() - height)
        x, y = float(event.x) - left, float(event.y) - top
        if not (0.0 <= x < width and 0.0 <= y < height):
            return None
        return x / width, y / height

    def _on_layout_pointer_down(self, event: tk.Event[tk.Misc]) -> None:
        if not self._creation_mode.get():
            return
        point = self._image_coordinates(event)
        if point is not None:
            self._layout_edit_commands.append(LayoutEditCommand("select", *point))

    def _on_layout_pointer_drag(self, event: tk.Event[tk.Misc]) -> None:
        if not self._creation_mode.get():
            return
        point = self._image_coordinates(event)
        if point is not None:
            self._layout_edit_commands.append(LayoutEditCommand("drag", *point))

    def _on_layout_pointer_up(self, _event: tk.Event[tk.Misc]) -> None:
        if self._creation_mode.get():
            self._layout_edit_commands.append(LayoutEditCommand("release"))

    def _on_layout_mouse_wheel(self, event: tk.Event[tk.Misc]) -> None:
        self._on_layout_wheel_step(event, 1 if event.delta > 0 else -1)

    def _on_layout_wheel_step(self, _event: tk.Event[tk.Misc], direction: int) -> None:
        if self._creation_mode.get():
            self._layout_edit_commands.append(LayoutEditCommand(
                "rotate", yaw_deg=5.0 * direction,
            ))

    def _queue_layout_command(self, kind: str, *, name: str | None = None) -> None:
        if self._creation_mode.get() or kind in {"import", "export"}:
            self._layout_edit_commands.append(LayoutEditCommand(kind, name=name))

    def refresh_layout_names(self, selected: str | None = None) -> None:
        names = list_named_custom_layouts(self._scene_name.get())
        self._layout_library_selector.configure(values=names)
        choice = selected if selected in names else (
            self._layout_library_name.get()
            if self._layout_library_name.get() in names else ""
        )
        if not choice and names:
            choice = names[0]
        self._layout_library_name.set(choice)

    def _request_layout_import(self) -> None:
        name = self._layout_library_name.get()
        if not name:
            messagebox.showinfo(
                "Import custom layout",
                "No saved layout exists for the current scene and robot.",
                parent=self.root,
            )
            return
        self._queue_layout_command("import", name=name)

    def _request_layout_export(self) -> None:
        proposed = self._layout_library_name.get() or "layout_01"
        name = simpledialog.askstring(
            "Export custom layout",
            "Layout name (letters, numbers, '.', '_' or '-'):",
            initialvalue=proposed,
            parent=self.root,
        )
        if name is None:
            return
        try:
            name = validate_custom_layout_name(name)
        except ValueError as error:
            messagebox.showerror(
                "Invalid layout name", str(error), parent=self.root,
            )
            return
        path = named_custom_layout_path(self._scene_name.get(), name)
        action = "Overwrite" if path.exists() else "Export"
        if not messagebox.askyesno(
            "Confirm layout export",
            f"{action} layout '{name}'?\n\n{path}",
            parent=self.root,
        ):
            return
        self._queue_layout_command("export", name=name)

    def _queue_layout_numeric_apply(self) -> None:
        if not self._creation_mode.get():
            return
        try:
            x, y, yaw = (
                float(self._layout_x.get()), float(self._layout_y.get()),
                float(self._layout_yaw.get()),
            )
            if not np.isfinite((x, y, yaw)).all():
                raise ValueError
        except ValueError:
            for entry in self._layout_entries:
                entry.configure(bg="#ffd6d6")
            return
        for entry in self._layout_entries:
            entry.configure(bg="#ffffff")
        self._layout_edit_commands.append(LayoutEditCommand(
            "set_pose", x=x, y=y, yaw_deg=yaw,
        ))

    def _on_creation_mode_changed(self) -> None:
        enabled = bool(self._creation_mode.get())
        self._domain_seed_entry.configure(state="disabled" if enabled else "normal")
        self._layout_apply_button.configure(state="normal" if enabled else "disabled")
        for entry in self._layout_entries:
            entry.configure(state="normal" if enabled else "disabled")
        self._layout_edit_commands.append(LayoutEditCommand(
            "enter" if enabled else "exit",
        ))
        if enabled:
            self.refresh_layout_names()
        self._preset_task_status.set(
            "creation mode: cyan dots show initializer range; click + drag XY, wheel rotates yaw"
            if enabled else "creation layout retained"
        )

    def set_layout_selection(
        self, name: str | None, x: float = 0.0, y: float = 0.0,
        yaw_deg: float = 0.0,
    ) -> None:
        self._layout_selection.set(name or "no selection")
        if name is not None:
            self._layout_x.set(f"{x:.4f}")
            self._layout_y.set(f"{y:.4f}")
            self._layout_yaw.set(f"{yaw_deg:.2f}")

    def camera_frames(self, wrist_rgb: np.ndarray, side_rgb: np.ndarray) -> None:
        """Update the wrist and fixed tabletop-side camera previews together."""
        self._wrist_photo = ImageTk.PhotoImage(Image.fromarray(wrist_rgb))
        self._side_photo = ImageTk.PhotoImage(Image.fromarray(side_rgb))
        self.wrist_image.configure(image=self._wrist_photo)
        self.side_image.configure(image=self._side_photo)

    def _draw_penalty_chart(self, penalties: dict[str, float]) -> None:
        self.chart.delete("all"); values = tuple(max(0.0, float(penalties.get(n, 0.0))) for n in self.PENALTY_TERMS); total = sum(values)
        bar_left, bar_right = 365, self.CHART_WIDTH - 240
        bar_width = bar_right - bar_left
        for row, (name, value, color) in enumerate(zip(self.PENALTY_TERMS, values, self.PENALTY_COLORS, strict=True)):
            percent, y = (0.0 if total <= 1e-12 else 100.0 * value / total), 4 + row * 23
            self.chart.create_text(8, y + 7, text=name, anchor="w", fill="#e8edf2", font=("Segoe UI", 8))
            for column, (label, weight_index) in enumerate(self.WEIGHT_FIELDS[row]):
                x = 96 + 56 * column
                self.chart.create_text(x, y + 7, text=label, anchor="w", fill="#aab7c4", font=("Consolas", 8)); self.chart.create_window(x + 30, y + 7, window=self._weight_entries[weight_index])
            if name == "Incremental release":
                axis_weights = "  ".join(
                    f"{axis} {value:.3f}" if np.isfinite(value) else f"{axis} --"
                    for axis, value in zip(
                        ("Rx", "Ry", "Rz"),
                        self._release_loss_multipliers,
                        strict=True,
                    )
                )
                self.chart.create_text(
                    158, y + 7,
                    text=f"mult {axis_weights}", anchor="w",
                    fill="#aab7c4", font=("Consolas", 8),
                )
            self.chart.create_rectangle(bar_left, y, bar_right, y + 14, fill="#263746", outline="")
            self.chart.create_rectangle(bar_left, y, bar_left + bar_width * percent / 100.0, y + 14, fill=color, outline=""); self.chart.create_text(bar_right + 20, y + 7, text=f"{percent:5.1f}%  {value:8.2f}", anchor="w", fill="#e8edf2", font=("Consolas", 8))

    def _on_loss_profile_selected(self, _event: tk.Event[tk.Misc] | None = None) -> None:
        self._update_loss_profile_widget_state()
        self.root.focus_force()

    def _update_loss_profile_widget_state(self) -> None:
        profile = self.LOSS_CHOICES.get(
            self._loss_profile_label.get(), self._loss_profile.get(),
        )
        self._loss_profile.set(profile)
        legacy = False
        for entry in self._weight_entries:
            entry.configure(state="normal" if legacy else "disabled")
        for entry in self._sqp_loss_weight_entries:
            entry.configure(state="normal")

    def _draw_goal_error(self, error: np.ndarray) -> None:
        """Draw position and axis-angle errors as two compact 3-row columns."""
        self.error_chart.delete("all")
        labels = ("X", "Y", "Z", "Rx", "Ry", "Rz")
        scales = (0.05, 0.05, 0.05, *([np.deg2rad(15.0)] * 3))
        column_width = self.CHART_WIDTH / 2.0
        for index, (label, value, scale) in enumerate(zip(labels, error, scales, strict=True)):
            column, row = divmod(index, 3)
            column_left, y = 8 + column * column_width, 3 + row * 26
            bar_left, bar_right = column_left + 30, column_left + column_width - 135
            center = 0.5 * (bar_left + bar_right)
            half_bar = 0.5 * (bar_right - bar_left) - 3
            ratio = float(np.clip(value / scale, -1.0, 1.0))
            color = "#58d68d" if ratio >= 0.0 else "#ec7063"
            self.error_chart.create_text(column_left, y + 7, text=label, anchor="w", fill="#e8edf2", font=("Consolas", 8))
            self.error_chart.create_line(bar_left, y + 7, bar_right, y + 7, fill="#314454")
            self.error_chart.create_line(center, y + 1, center, y + 13, fill="#aab7c4")
            self.error_chart.create_rectangle(min(center, center + half_bar * ratio), y + 2, max(center, center + half_bar * ratio), y + 12, fill=color, outline="")
            shown, unit = (value * 1000.0, "mm") if index < 3 else (np.degrees(value), "deg")
            self.error_chart.create_text(bar_right + 10, y + 7, text=f"{shown:+8.2f} {unit}", anchor="w", fill="#e8edf2", font=("Consolas", 8))

    def _draw_evaluation_chart(
        self,
        summary: MultiRateEvaluationSummary,
    ) -> None:
        """Draw aligned optimizer, 10 Hz actual, and raw 1 kHz metrics."""
        self.evaluation_chart.delete("all")
        bar_left = 145
        bar_right = self.CHART_WIDTH - 470
        value_x = bar_right + 12
        legend_x = bar_left
        for _, legend, color in self.EVALUATION_SERIES:
            self.evaluation_chart.create_rectangle(
                legend_x, 4, legend_x + 10, 12, fill=color, outline="",
            )
            self.evaluation_chart.create_text(
                legend_x + 14, 8, text=legend, anchor="w",
                fill="#d7dee7", font=("Segoe UI", 7),
            )
            legend_x += 125
        for row, (field, label, unit, scale) in enumerate(
            self.EVALUATION_METRICS
        ):
            y = 18 + row * 19
            self.evaluation_chart.create_text(
                8, y + 6,
                text=f"{label}{(' ' + unit) if unit else ''}", anchor="w",
                fill="#e8edf2", font=("Segoe UI", 7),
            )
            for series_row, (attribute, _, color) in enumerate(
                self.EVALUATION_SERIES
            ):
                series_summary = getattr(summary, attribute)
                value = max(0.0, float(getattr(series_summary, field)))
                ratio = float(np.clip(value / scale, 0.0, 1.0))
                series_y = y + series_row * 4
                self.evaluation_chart.create_rectangle(
                    bar_left, series_y, bar_right, series_y + 3,
                    fill="#263746", outline="",
                )
                self.evaluation_chart.create_rectangle(
                    bar_left, series_y,
                    bar_left + (bar_right - bar_left) * ratio,
                    series_y + 3,
                    fill=color, outline="",
                )
                self.evaluation_chart.create_text(
                    value_x + series_row * 90, y + 6,
                    text=(
                        f"{value:.4g}"
                        if series_summary.samples else "—"
                    ),
                    anchor="w", fill=color, font=("Consolas", 7),
                )

    def _on_press(self, event: tk.Event[tk.Misc]) -> None:
        # Numeric entry widgets own their editing keys; their dedicated
        # handler below returns focus to the teleop panel for motion keys.
        if isinstance(event.widget, tk.Entry):
            return
        name = event.keysym.lower()
        if name == "escape": self.running = False; return
        if name not in self.CONTROL_KEYS: return
        repeated = name in self._pressed; self._pressed.add(name)
        if name == "c": self.set_gripper(True)
        elif name == "v": self.set_gripper(False)
        elif name == "h": self.home_requested = True
        elif name == "7" and not repeated: self.rotation_in_base = not self.rotation_in_base; self._refresh_task_state()
        elif name == "8" and not repeated: self._toggle_evaluation = True
        elif name == "9" and not repeated: self._toggle_trajectory = True
        elif name == "0" and not repeated: self._play_trajectory = True
        elif name in self._CAMERA_STEPS:
            adjust_target = self.camera_adjust_mode
            if adjust_target == "none":
                return
            variable, increment = self._CAMERA_STEPS[name]
            if adjust_target == "wrist":
                if variable == "x": self.camera_x_offset += increment
                elif variable == "z": self.camera_z_offset += increment
                else: self.camera_pitch_offset_deg += increment
            else:
                if variable == "x": self.main_camera_x_offset += increment
                elif variable == "z": self.main_camera_z_offset += increment
                else: self.main_camera_pitch_offset_deg += increment
            self._refresh_camera_state()
        elif name in "123" and not repeated:
            index = int(name) - 1; self.active_dofs[index] = not self.active_dofs[index]; self._toggled_dofs.append(index); self._refresh_task_state()

    def _weight_entry_keypress(self, event: tk.Event[tk.Misc]) -> str | None:
        """Do not let continuous teleop letters end up in a weight field."""
        name = event.keysym.lower()
        state = int(getattr(event, "state", 0))
        # Let Tk's Entry class binding handle clipboard/edit shortcuts before
        # interpreting C/V and other letters as continuous teleoperation.
        if (
            (state & 0x0004 and name in {"a", "c", "v", "x", "y", "z", "insert"})
            or (state & 0x0001 and name == "insert")
        ):
            return None
        if name not in set("wasdikqejuolcvh"):
            return None
        repeated = name in self._pressed
        self._pressed.add(name)
        if name == "c": self.set_gripper(True)
        elif name == "v": self.set_gripper(False)
        elif name == "h": self.home_requested = True
        self.root.focus_force()
        return "break"

    def _on_release(self, event: tk.Event[tk.Misc]) -> None: self._pressed.discard(event.keysym.lower())
    def _on_scene_group_selected(self, _event: tk.Event[tk.Misc] | None = None) -> None:
        scenes = scene_choices_for_group(self._scene_group.get())
        self._scene_selector.configure(values=scenes)
        if self._scene_choice.get() not in scenes:
            self._scene_choice.set(scenes[0])
        self._on_scene_selected()
        self.root.focus_force()
    def _tasks_for_selected_scene(self) -> tuple[str, ...]:
        if self._scene_group.get() in (
            TRAINING_TASK_GROUP,
            TOLERANCE_DISCRIMINATION_GROUP,
        ):
            return tuple(
                task.task_id for task in tasks_for_scene_choice(
                    self._scene_group.get(), self._scene_choice.get(),
                )
            )
        if self._scene_group.get() == EXISTING_SCENE_GROUP:
            return PresetTaskExecutor.task_names(self._scene_name.get())
        return ()
    def _selected_runtime_task(self) -> str:
        selected = self._preset_task_name.get()
        if selected == "(no preset task)":
            return selected
        if self._scene_group.get() in (
            TRAINING_TASK_GROUP,
            TOLERANCE_DISCRIMINATION_GROUP,
        ):
            return runtime_task_from_choice(
                self._scene_group.get(), self._scene_choice.get(), selected,
            )
        return selected
    def _on_scene_selected(self, _event: tk.Event[tk.Misc] | None = None) -> None:
        self._scene_name.set(runtime_scene_from_choice(
            self._scene_group.get(), self._scene_choice.get(),
        ))
        tasks = self._tasks_for_selected_scene()
        values = tasks if tasks else ("(no preset task)",)
        self._preset_task_selector.configure(values=values)
        self._preset_task_name.set(values[0])
        self._preset_task_status.set("idle")
        self.refresh_layout_names()
        self.root.focus_force()

    def _on_robot_selected(self, _event: tk.Event[tk.Misc] | None = None) -> None:
        self.refresh_layout_names()
        self.root.focus_force()
    def _request_preset_task(self) -> None:
        if self._preset_task_name.get() != "(no preset task)":
            if not self._commit_stage_editor():
                self._preset_task_status.set("Invalid tolerance settings: check the highlighted fields")
                return
            if self._creation_mode.get():
                self._creation_mode.set(False)
                self._on_creation_mode_changed()
            self._execute_preset_task = True
            self._preset_task_status.set("queued")
        self.root.focus_force()
    def _request_scene_reset(self) -> None:
        self._reset_scene_requested = True
        self.closed = False
        self._preset_task_status.set("scene reset queued")
        self._refresh_task_state()
        self.root.focus_force()
    def _load_stage_editor(self) -> None:
        for axis in range(3):
            self._stage_target_vars[axis].set(f"{self._stage_target_rotvec_deg[self._editing_stage_index, axis]:.2f}")
            self._stage_target_entries[axis].configure(bg="#ffffff")
            self._stage_frame_vars[axis].set(f"{self._stage_frame_rotvec_deg[self._editing_stage_index, axis]:.2f}")
            self._stage_frame_entries[axis].configure(bg="#ffffff")
        for axis in range(3):
            self._stage_ranged_vars[axis].set(bool(self._stage_ranged[self._editing_stage_index, axis]))
            self._stage_limit_vars[axis].set(f"{self._stage_limits_deg[self._editing_stage_index, axis]:g}")
            self._stage_limit_entries[axis].configure(bg="#ffffff")
    def _commit_stage_editor(self) -> bool:
        limits = np.empty(3)
        target_rotvec = np.empty(3)
        frame_rotvec = np.empty(3)
        valid = True
        for axis in range(3):
            try:
                value = float(self._stage_target_vars[axis].get())
                if not np.isfinite(value) or abs(value) > 1440.0:
                    raise ValueError
                target_rotvec[axis] = value
                self._stage_target_entries[axis].configure(bg="#ffffff")
            except ValueError:
                valid = False
                self._stage_target_entries[axis].configure(bg="#ffd6d6")
        for axis in range(3):
            try:
                value = float(self._stage_frame_vars[axis].get())
                if not np.isfinite(value) or abs(value) > 1440.0:
                    raise ValueError
                frame_rotvec[axis] = value
                self._stage_frame_entries[axis].configure(bg="#ffffff")
            except ValueError:
                valid = False
                self._stage_frame_entries[axis].configure(bg="#ffd6d6")
        for axis in range(3):
            try:
                value = float(self._stage_limit_vars[axis].get())
                if not np.isfinite(value) or not 0.0 <= value <= 180.0:
                    raise ValueError
                limits[axis] = value
                self._stage_limit_entries[axis].configure(bg="#ffffff")
            except ValueError:
                valid = False
                self._stage_limit_entries[axis].configure(bg="#ffd6d6")
        if valid:
            stage = self._editing_stage_index
            updates: dict[str, np.ndarray] = {}
            ranged = np.array([variable.get() for variable in self._stage_ranged_vars], dtype=bool)
            if stage in (1, 3):
                ranged[:] = False
            if not np.array_equal(ranged, self._stage_ranged[stage]):
                updates["_stage_ranged"] = ranged
            # Loading an automatic descriptor rounds its displayed rotvec to
            # two decimals. That alone must not become a manual override.
            for name, value in (
                ("_stage_target_rotvec_deg", target_rotvec),
                ("_stage_frame_rotvec_deg", frame_rotvec),
            ):
                previous = getattr(self, name)[stage]
                if not (
                    np.array_equal(value, previous)
                    or np.array_equal(value, np.round(previous, 2))
                ):
                    updates[name] = value
            if "_stage_frame_rotvec_deg" in updates:
                canonical_frame = np.degrees(canonical_rotation_vector(
                    rotation_matrix(np.radians(frame_rotvec)),
                ))
                updates["_stage_frame_rotvec_deg"] = canonical_frame
                for axis in range(3):
                    self._stage_frame_vars[axis].set(f"{canonical_frame[axis]:.2f}")
            for name, value in updates.items():
                getattr(self, name)[stage] = value
            if updates:
                key = self._stage_settings_key(stage)
                self._manual_stage_settings.setdefault(key, {}).update(
                    {name: value.copy() for name, value in updates.items()}
                )
            self._stage_limits_deg[stage] = limits
        return valid

    def _stage_settings_key(self, stage: int) -> tuple:
        context = self._task_tolerance_context
        task_key = None if context is None else (context.scene, context.task, context.object_variant)
        return task_key, self._tolerance_subtask_index, stage
    def _apply_active_stage(self) -> None:
        # Gripper motion is always a fully Specific rotational phase. Closing
        # a gripper may never silently release orientation or load another
        # pose target.
        new_active = effective_rotation_specific_axes(
            self._tolerance_enabled.get(),
            self.active_stage,
            self._stage_ranged[self.active_stage] & (self._stage_limits_deg[self.active_stage] > 0.0),
        )
        changed = np.flatnonzero(self.active_dofs[3:] != new_active)
        self.active_dofs[3:] = new_active
        self._toggled_dofs.extend(int(axis + 3) for axis in changed)
        self._refresh_task_state()
    def _on_tolerance_enabled_changed(self) -> None:
        self._commit_stage_editor()
        self._apply_active_stage()
        state = "enabled" if self._tolerance_enabled.get() else "disabled"
        self._preset_task_status.set(f"tolerance {state}")
        self.root.focus_force()
    def _on_stage_editor_selected(self, _event: tk.Event[tk.Misc] | None = None) -> None:
        self._commit_stage_editor()
        self._editing_stage_index = self.STAGE_NAMES.index(self._stage_editor_name.get())
        self._load_stage_editor()
        self.root.focus_force()
    def _on_stage_config_changed(self) -> None:
        self._commit_stage_editor()
        if self._editing_stage_index == self.active_stage:
            self._apply_active_stage()
    def _on_stage_limit_commit(self, _event: tk.Event[tk.Misc] | None = None) -> None:
        self._on_stage_config_changed()
    def _on_stage_frame_commit(self, _event: tk.Event[tk.Misc] | None = None) -> None:
        self._commit_stage_editor()
        self._refresh_task_state()
    def _on_stage_target_commit(self, _event: tk.Event[tk.Misc] | None = None) -> None:
        self._commit_stage_editor()
        self._refresh_task_state()
    def _request_tolerance_target_capture(self) -> None:
        self._capture_tolerance_target_stage = self._editing_stage_index
        self._preset_task_status.set(f"capture target EE for stage {self._editing_stage_index + 1} queued")
        self.root.focus_force()
    def _request_tolerance_frame_capture(self) -> None:
        self._capture_tolerance_frame_stage = self._editing_stage_index
        self._preset_task_status.set(f"capture EE for stage {self._editing_stage_index + 1} queued")
        self.root.focus_force()
    def set_stage_target(
        self,
        stage_index: int,
        rotation: np.ndarray,
        position: np.ndarray | None = None,
    ) -> None:
        """Store the current EE orientation as one stage's target pose."""
        if not 0 <= stage_index < ROTATION_STAGE_COUNT:
            raise IndexError("Tolerance stage is out of range.")
        previous = np.radians(self._stage_target_rotvec_deg[stage_index])
        self._stage_target_rotvec_deg[stage_index] = np.degrees(
            continuous_rotation_vector(np.asarray(rotation, dtype=float), previous)
        )
        if position is not None:
            position = np.asarray(position, dtype=float)
            if position.shape != (3,) or not np.all(np.isfinite(position)):
                raise ValueError("Stage-target position must contain three finite values")
            self._stage_target_position_m[stage_index] = position
        self._manual_stage_settings.setdefault(self._stage_settings_key(stage_index), {}).update({
            "_stage_target_rotvec_deg": self._stage_target_rotvec_deg[stage_index].copy(),
            "_stage_target_position_m": self._stage_target_position_m[stage_index].copy(),
        })
        if stage_index == self._editing_stage_index:
            self._load_stage_editor()
        self._preset_task_status.set(f"stage {stage_index + 1} target recorded")
        self._refresh_task_state()
    def set_stage_tolerance_frame(self, stage_index: int, rotation: np.ndarray) -> None:
        """Store the current EE orientation as one stage's fixed frame."""
        if not 0 <= stage_index < ROTATION_STAGE_COUNT:
            raise IndexError("Tolerance stage is out of range.")
        self._stage_frame_rotvec_deg[stage_index] = np.degrees(
            canonical_rotation_vector(np.asarray(rotation, dtype=float))
        )
        self._manual_stage_settings.setdefault(self._stage_settings_key(stage_index), {})[
            "_stage_frame_rotvec_deg"
        ] = self._stage_frame_rotvec_deg[stage_index].copy()
        if stage_index == self._editing_stage_index:
            self._load_stage_editor()
        self._preset_task_status.set(f"stage {stage_index + 1} frame recorded")
        self._refresh_task_state()
    def apply_task_tolerance_context(
        self,
        context: ResolvedTaskTolerance,
    ) -> None:
        """Store a task context and activate its first pick-place subtask."""
        self._commit_stage_editor()
        first_context = self._task_tolerance_context is None
        self._task_tolerance_context = context
        if first_context:
            # Accept edits made while the first background plan was pending.
            for stage in (0, 2):
                pending = self._manual_stage_settings.pop((None, 0, stage), None)
                if pending is not None:
                    task_key = (context.scene, context.task, context.object_variant)
                    self._manual_stage_settings[(task_key, 0, stage)] = pending
        self._tolerance_subtask_count = max(
            1,
            len(context.subtasks),
            len(long_task_subtask_labels(context.task)),
        )
        self._apply_tolerance_subtask(0)

    def apply_runtime_tolerance_subtask(
        self,
        subtask: ResolvedSubtaskTolerance,
        *,
        total_subtasks: int,
    ) -> np.ndarray:
        """Install a deferred segment before its first preset waypoint.

        Dynamic long tasks do not know later endpoint poses until the previous
        placement is complete.  Merge the newly planned descriptor into the
        cached task context so the subsequent gripper-phase observation cannot
        restore an older subtask's mask.
        """
        context = self._task_tolerance_context
        if context is None:
            raise RuntimeError("Task tolerance context is not initialized")
        subtasks = list(context.subtasks)
        if subtask.index < len(subtasks):
            subtasks[subtask.index] = subtask
        elif subtask.index == len(subtasks):
            subtasks.append(subtask)
        else:
            raise ValueError("Runtime tolerance subtasks must arrive in order")
        self._task_tolerance_context = replace(
            context,
            subtasks=tuple(subtasks),
        )
        self._tolerance_subtask_count = max(
            1,
            int(total_subtasks),
            len(subtasks),
        )
        self.active_stage = 0
        self._transition_source_stage = 0
        self._apply_tolerance_subtask(subtask.index)
        return np.radians(self._stage_limits_deg[0].copy())

    def _apply_tolerance_subtask(self, subtask_index: int) -> None:
        """Replace Pre/Post goal, axes and mask for one detected subtask."""
        context = self._task_tolerance_context
        if context is None:
            return
        resolved_subtask = context.subtask(subtask_index)
        self._commit_stage_editor()
        self._tolerance_subtask_index = resolved_subtask.index
        self._tolerance_subtask_label = resolved_subtask.label
        for stage_index, resolved in (
            (0, resolved_subtask.pregrasp),
            (2, resolved_subtask.postgrasp),
        ):
            previous_target = np.radians(
                self._stage_target_rotvec_deg[stage_index]
            )
            self._stage_target_rotvec_deg[stage_index] = np.degrees(
                continuous_rotation_vector(
                    resolved.target_rotation,
                    previous_target,
                )
            )
            self._stage_target_position_m[stage_index] = (
                resolved.target_position
            )
            self._stage_frame_rotvec_deg[stage_index] = np.degrees(
                canonical_rotation_vector(resolved.tolerance_frame)
            )
            self._stage_ranged[stage_index] = resolved.ranged_axes
            for name, value in self._manual_stage_settings.get(
                self._stage_settings_key(stage_index), {},
            ).items():
                getattr(self, name)[stage_index] = value
        self._stage_ranged[1] = False
        self._stage_ranged[3] = False
        self._load_stage_editor()
        self._apply_active_stage()
        variant = (
            "" if context.object_variant is None
            else f" / {context.object_variant}"
        )
        self._preset_task_status.set(
            f"tolerance auto-configured: {context.task}{variant} | "
            f"subtask {resolved_subtask.index + 1}/{self._tolerance_subtask_count} "
            f"{resolved_subtask.label}"
        )
        self._refresh_task_state()

    def reset_episode_tolerance(self) -> np.ndarray:
        """Start a reset episode at Pre-grasp and retain edited limits."""
        self._commit_stage_editor()
        self.active_stage = 0
        self._transition_source_stage = 0
        self._tolerance_subtask_index = 0
        self._apply_active_stage()
        self._refresh_task_state()
        return np.radians(self._stage_limits_deg[0].copy())

    def reset_tolerance_subtask(self) -> None:
        """Restart at the first PRE while retaining the operator's edits."""
        self._commit_stage_editor()
        self.active_stage = 0
        self._transition_source_stage = 0
        if self._task_tolerance_context is None:
            self._apply_active_stage()
        else:
            self._apply_tolerance_subtask(0)

    def tolerance_snapshot(self, snapshot: InputSnapshot) -> InputSnapshot:
        """Refresh tolerance fields after an automatic UI state transition.

        This does not poll Tk or consume one-shot input events.
        """
        stage = 0 if self.active_stage == 1 else 2 if self.active_stage == 3 else self.active_stage
        return replace(
            snapshot,
            active_stage=self.active_stage,
            active_dofs=self.active_dofs.copy(),
            tolerance_target_position=self._stage_target_position_m[stage].copy(),
            rotation_tolerance_rad=np.radians(self._stage_limits_deg[stage].copy()),
            rotation_tolerance_target=rotation_matrix(np.radians(self._stage_target_rotvec_deg[stage])),
            rotation_tolerance_frame=rotation_matrix(np.radians(self._stage_frame_rotvec_deg[stage])),
        )

    def set_gripper_phase(self, observation: PhaseObservation) -> None:
        """Apply the automatically recognized manipulation phase."""
        if (
            self._task_tolerance_context is not None
            and observation.subtask_changed
            and observation.subtask_index < self._tolerance_subtask_count
        ):
            self._apply_tolerance_subtask(observation.subtask_index)
        stage = int(observation.phase)
        if stage != self.active_stage:
            self._commit_stage_editor()
            if stage == 1:
                self._transition_source_stage = 0
            elif stage == 3:
                self._transition_source_stage = 2
            self.active_stage = stage
            self._apply_active_stage()
        self._gripper_phase_detail = (
            f"subtask {min(observation.subtask_index + 1, self._tolerance_subtask_count)}"
            f"/{self._tolerance_subtask_count} {self._tolerance_subtask_label} | "
            f"{observation.phase.label}: aperture {observation.aperture_m * 1000.0:.2f} mm, "
            f"10 Hz motion {observation.motion_m * 1000.0:.2f} mm"
        )
        self._refresh_task_state()
    def _refresh_camera_state(self) -> None:
        if self.camera_adjust_mode == "main":
            text = (
                f"MAIN ADJUST  X {self.main_camera_x_offset * 1000.0:+.0f}  "
                f"Z {self.main_camera_z_offset * 1000.0:+.0f} mm  "
                f"P {self.main_camera_pitch_offset_deg:+.0f}°"
            )
        elif self.camera_adjust_mode == "wrist":
            text = (
                f"ADJUST  X {self.camera_x_offset * 1000.0:+.0f}  "
                f"Z {self.camera_z_offset * 1000.0:+.0f} mm  "
                f"P {self.camera_pitch_offset_deg:+.0f}°"
            )
        else:
            text = (
                f"FIXED  X {self.camera_x_offset * 1000.0:+.0f}  "
                f"Z {self.camera_z_offset * 1000.0:+.0f} mm  "
                f"P {self.camera_pitch_offset_deg:+.0f}°"
            )
        self.wrist_camera_state.set(text)
    def _refresh_task_state(self) -> None:
        specific = " ".join(l for l, e in zip(DOF_LABELS, self.active_dofs, strict=True) if e) or "none"
        free_translation = " ".join(DOF_LABELS[i] for i in range(3) if not self.active_dofs[i]) or "none"
        ranged_rotation = " ".join(DOF_LABELS[i] for i in range(3, 6) if not self.active_dofs[i]) or "none"
        frame = "base frame" if self.rotation_in_base else "tool frame"
        config_stage = (
            self._transition_source_stage
            if self.active_stage in (1, 3)
            else self.active_stage
        )
        limits = self._stage_limits_deg[config_stage]
        target_rotvec = self._stage_target_rotvec_deg[config_stage]
        target_position = self._stage_target_position_m[config_stage]
        frame_rotvec = self._stage_frame_rotvec_deg[config_stage]
        frame_mode_label = next(
            label
            for label, mode in self.TOLERANCE_FRAME_MODE_LABELS.items()
            if mode == self._tolerance_frame_mode.get()
        )
        self.task_state.set(
            f"Tolerance {'ON' if self._tolerance_enabled.get() else 'OFF'} | "
            f"Frame {frame_mode_label} | "
            f"Subtask {self._tolerance_subtask_index + 1}/{self._tolerance_subtask_count} "
            f"{self._tolerance_subtask_label} | {self.STAGE_NAMES[self.active_stage]} | "
            f"Specific: {specific} | Ranged: {ranged_rotation}\n"
            f"Free XYZ: {free_translation} | GoalP {target_position[0]:+.3f}/{target_position[1]:+.3f}/{target_position[2]:+.3f} | "
            f"GoalRV {target_rotvec[0]:+.1f}/{target_rotvec[1]:+.1f}/{target_rotvec[2]:+.1f} | "
            f"TolFRV {frame_rotvec[0]:+.1f}/{frame_rotvec[1]:+.1f}/{frame_rotvec[2]:+.1f} | "
            f"Tol {limits[0]:g}/{limits[1]:g}/{limits[2]:g} deg | {frame} | Grip {'CLOSED' if self.closed else 'OPEN'}"
            + (" | all rotation axes Specific" if self.active_stage in (1, 3) else "")
        )
        self.status.set(
            f"Gripper: {'CLOSED' if self.closed else 'OPEN'} (C/V)   |   {self._gripper_phase_detail}   |   H Home"
        )
    def set_gripper(self, closed: bool) -> None: self.closed = closed; self._refresh_task_state()
    def configure_automated_run(
        self, *, domain_seed: int, speed_scale: float,
    ) -> None:
        """Queue one preset run while preserving the normal UI execution path."""
        self._domain_seed.set(str(int(domain_seed)))
        self._domain_seed_value = int(domain_seed)
        self._teleop_speed_scale.set(f"{float(speed_scale):g}")
        self._teleop_speed_scale_value = float(speed_scale)
        self._execute_preset_task = True

    def snapshot(self) -> InputSnapshot:
        try: self.root.update_idletasks(); self.root.update()
        except tk.TclError: self.running = False
        self._commit_stage_editor()
        self._apply_active_stage()
        toggles, self._toggled_dofs = tuple(self._toggled_dofs), []; home, self.home_requested = self.home_requested, False
        reset_scene, self._reset_scene_requested = self._reset_scene_requested, False
        evaluation, self._toggle_evaluation = self._toggle_evaluation, False
        trajectory, self._toggle_trajectory = self._toggle_trajectory, False
        playback, self._play_trajectory = self._play_trajectory, False
        execute_preset, self._execute_preset_task = self._execute_preset_task, False
        layout_commands, self._layout_edit_commands = (
            tuple(self._layout_edit_commands), []
        )
        capture_target_stage, self._capture_tolerance_target_stage = self._capture_tolerance_target_stage, None
        capture_frame_stage, self._capture_tolerance_frame_stage = self._capture_tolerance_frame_stage, None
        for index, variable in enumerate(self._weight_vars):
            try:
                value = float(variable.get())
                if np.isfinite(value): self._objective_weights[index] = np.clip(value, 0.0, 1000.0)
            except ValueError:
                pass
        for index, variable in enumerate(self._sqp_loss_weight_vars):
            try:
                value = float(variable.get())
                if np.isfinite(value):
                    self._sqp_loss_weights[index] = np.clip(value, 0.0, 1000.0)
            except ValueError:
                pass
        try:
            seed = int(self._domain_seed.get().strip())
            if not -1 <= seed < (1 << 63):
                raise ValueError
            self._domain_seed_value = seed
            self._domain_seed_entry.configure(bg="#ffffff")
        except ValueError:
            # Keep the last valid pose while the user is editing an incomplete
            # or invalid integer. A valid value takes effect on the next step.
            self._domain_seed_entry.configure(bg="#ffd6d6")
        try:
            speed_scale = float(self._teleop_speed_scale.get().strip())
            if not np.isfinite(speed_scale) or not 0.05 <= speed_scale <= 5.0:
                raise ValueError
            self._teleop_speed_scale_value = speed_scale
            self._teleop_speed_entry.configure(bg="#ffffff")
        except ValueError:
            self._teleop_speed_entry.configure(bg="#ffd6d6")
        try:
            boundary_min = float(
                self._workspace_boundary_distance_min.get().strip(),
            )
            boundary_max = float(
                self._workspace_boundary_distance_max.get().strip(),
            )
            if not (
                np.isfinite(boundary_min)
                and np.isfinite(boundary_max)
                and 0.0 <= boundary_min <= boundary_max <= 2.0
            ):
                raise ValueError
            self._workspace_boundary_distance_min_value = boundary_min
            self._workspace_boundary_distance_max_value = boundary_max
            self._workspace_boundary_min_entry.configure(bg="#ffffff")
            self._workspace_boundary_max_entry.configure(bg="#ffffff")
        except ValueError:
            self._workspace_boundary_min_entry.configure(bg="#ffd6d6")
            self._workspace_boundary_max_entry.configure(bg="#ffd6d6")
        config_stage = 0 if self.active_stage == 1 else 2 if self.active_stage == 3 else self.active_stage
        tolerance_target = rotation_matrix(np.radians(self._stage_target_rotvec_deg[config_stage]))
        tolerance_target_position = self._stage_target_position_m[
            config_stage
        ].copy()
        tolerance_frame = rotation_matrix(np.radians(self._stage_frame_rotvec_deg[config_stage]))
        loss_profile = self._loss_profile.get()
        robot_uid = ROBOT_UID_BY_LABEL[self._robot_label.get()]
        return InputSnapshot(set(self._pressed), self.closed, home, reset_scene, toggles, self.active_dofs.copy(), self.active_stage, bool(self._tolerance_enabled.get()), self._tolerance_frame_mode.get(), tolerance_target_position, np.radians(self._stage_limits_deg[config_stage].copy()), tolerance_target, tolerance_frame, capture_target_stage, capture_frame_stage, self.rotation_in_base, self._teleop_speed_scale_value, evaluation, trajectory, playback, self._objective_weights.copy(), loss_profile, self._sqp_loss_weights.copy(), robot_uid, self._scene_name.get(), self._scene_group.get(), self._scene_choice.get(), self._selected_runtime_task(), execute_preset, self.camera_x_offset, self.camera_z_offset, self.camera_pitch_offset_deg, self.main_camera_x_offset, self.main_camera_z_offset, self.main_camera_pitch_offset_deg, self._domain_seed_value, bool(self._show_reachability_map.get()), bool(self._domain_randomization_enabled.get()), bool(self._boundary_unreachable_placement.get()), self._workspace_boundary_distance_min_value, self._workspace_boundary_distance_max_value, bool(self._creation_mode.get()), layout_commands)

    def preset_task_status(self, status: str) -> None:
        self._preset_task_status.set(status.removeprefix("Preset task: "))

    def set_creation_locked(self, locked: bool) -> None:
        state = "disabled" if locked else "normal"
        self._creation_toggle.configure(state=state)
        if self._creation_mode.get():
            self._layout_apply_button.configure(state=state)
            for entry in self._layout_entries:
                entry.configure(state=state)

    def set_tolerance_frame_mode_locked(self, locked: bool) -> None:
        """Prevent branch changes while one complete preset task is active."""
        state = "disabled" if locked else "normal"
        for button in self._tolerance_frame_mode_buttons:
            button.configure(state=state)

    def reachability_status(self, status: str) -> None:
        self._reachability_status.set(status)

    def recording_state(self, evaluation_recording: bool, trajectory_recording: bool, trajectory_playing: bool, progress: tuple[int, int]) -> None:
        metrics = "RECORDING" if evaluation_recording else "idle"
        trajectory = f"PLAY {progress[0]}/{progress[1]}" if trajectory_playing else (f"RECORDING {progress[1]} steps" if trajectory_recording else "idle")
        self.record_state.set(f"Metrics: {metrics} (8)    Trajectory: {trajectory} (9 record, 0 replay)")

    def evaluation(
        self,
        summary: MultiRateEvaluationSummary,
        recording: bool,
    ) -> None:
        state = "RECORDING" if recording else "LAST RESULT"
        self.evaluation_state.set(
            f"{state} | n=opt:{summary.optimized_10hz.samples} "
            f"a10:{summary.actual_10hz.samples} "
            f"a100:{summary.actual_100hz.samples} "
            f"lp100:{summary.filtered_100hz.samples} "
            f"raw1k:{summary.actual_1khz.samples} | "
            f"raw exceed={summary.actual_1khz.exceed_tolerances} "
            f"(pos>10 mm or rot>0.10 rad) | 8: stop/toggle"
        )
        self._draw_evaluation_chart(summary)

    def telemetry(self, q_deg: np.ndarray, position: np.ndarray, rotation: np.ndarray, goal_error: np.ndarray, diagnostics: object | None, home_hold: bool, tolerance_deg: np.ndarray | None = None, active_rotations: np.ndarray | None = None, intent_preference: np.ndarray | None = None, actual_tolerance_deg: np.ndarray | None = None, release_loss_multipliers: np.ndarray | None = None, nominal_action_ema: np.ndarray | None = None, current_sigma_min: float | None = None) -> None:
        # Actual pose at this refresh, independent of the last optimizer solve.
        sigma_text = (
            f"{current_sigma_min:.6f}"
            if current_sigma_min is not None and np.isfinite(current_sigma_min)
            else "--"
        )
        self.sigma_min_state.set(f"Current σmin (velocity-normalized, full J): {sigma_text}")
        live_multipliers = np.asarray(
            np.full(3, np.nan)
            if release_loss_multipliers is None else release_loss_multipliers,
            dtype=float,
        )
        if live_multipliers.shape != (3,):
            raise ValueError("Release-loss display requires Rx/Ry/Rz multipliers.")
        self._release_loss_multipliers = live_multipliers.copy()
        live_ema = np.asarray(
            np.full(3, np.nan)
            if nominal_action_ema is None else nominal_action_ema,
            dtype=float,
        )
        if live_ema.shape != (3,):
            raise ValueError("EMA display requires Rx/Ry/Rz values.")
        self._nominal_action_ema = live_ema.copy()
        ema_fields = "  ".join(
            f"{axis} {value:.3f}" if np.isfinite(value) else f"{axis} --"
            for axis, value in zip(
                ("Rx", "Ry", "Rz"), live_ema, strict=True,
            )
        )
        self._ema_release_state.set(f"EMA  {ema_fields}")
        self._telemetry_rotvec_rad = continuous_rotation_vector(rotation, self._telemetry_rotvec_rad)
        rotvec_deg = np.degrees(self._telemetry_rotvec_rad)
        joint_fields = [f"q{i + 1}: {value:+7.2f} deg" for i, value in enumerate(q_deg)]
        self.joints.set("  ".join(joint_fields[:4]) + "\n" + "  ".join(joint_fields[4:])); self.pose.set(f"X: {position[0]:+7.4f} m  Y: {position[1]:+7.4f} m  Z: {position[2]:+7.4f} m\nRx: {rotvec_deg[0]:+7.2f} deg  Ry: {rotvec_deg[1]:+7.2f} deg  Rz: {rotvec_deg[2]:+7.2f} deg"); self.error_state.set("Left: XYZ (mm)     Right: axis-angle Rx/Ry/Rz (deg)     green: positive     red: negative"); self._draw_goal_error(goal_error)
        if tolerance_deg is None:
            self.rotation_tolerance_state.set("Rotation tolerance: inactive")
        else:
            active_rotations = np.ones(3, dtype=bool) if active_rotations is None else active_rotations
            intent_preference = np.zeros(3) if intent_preference is None else intent_preference
            config_stage = (
                self._transition_source_stage
                if self.active_stage in (1, 3)
                else self.active_stage
            )
            limits = self._stage_limits_deg[config_stage]
            fields = []
            actual_for_mode = np.zeros(3) if actual_tolerance_deg is None else actual_tolerance_deg
            for label, value, active, limit, preference, actual in zip(("Rx", "Ry", "Rz"), tolerance_deg, active_rotations, limits, intent_preference, actual_for_mode, strict=True):
                if active:
                    mode = "specific"
                elif actual > limit:
                    mode = f"step [{-2.0 * limit:+.1f}, +0.0] deg"
                elif actual < -limit:
                    mode = f"step [+0.0, {2.0 * limit:+.1f}] deg"
                else:
                    step_lower = -limit - actual
                    step_upper = limit - actual
                    mode = f"step [{step_lower:+.1f}, {step_upper:+.1f}] deg"
                fields.append(f"{label} {value:+.2f}° [{mode}, intent {preference:.2f}]")
            actual_fields = []
            if actual_tolerance_deg is not None:
                for label, value, active, limit in zip(("Rx", "Ry", "Rz"), actual_tolerance_deg, active_rotations, limits, strict=True):
                    outside = not active and abs(float(value)) > float(limit)
                    actual_fields.append(f"{label} {value:+.2f}°/±{limit:g}°{' OUT' if outside else ''}")
            self.rotation_tolerance_state.set(
                f"{self.STAGE_NAMES[self.active_stage]} controller-target error | " + "   ".join(fields)
                + ("" if actual_tolerance_deg is None else
                   "\nStage-relative optimized target minus nominal reference | " + "   ".join(actual_fields))
            )
        self._ik_state_label.configure(fg="#d7dee7")
        if home_hold:
            self.ik_state.set("Home joint hold — press an enabled Cartesian key to resume")
            self._draw_penalty_chart({})
        elif diagnostics is not None:
            feasibility = "feasible" if diagnostics.feasible else "INFEASIBLE"
            status_color = "#ec7063" if not diagnostics.feasible else "#d7dee7"
            self._ik_state_label.configure(fg=status_color)
            self.ik_state.set(
                f"IPOPT  {feasibility}  cost={diagnostics.cost:.3f}   "
                f"iter={diagnostics.iterations:02d}   {diagnostics.elapsed_ms:.1f} ms   "
                f"|ep|={diagnostics.position_residual:.2e}   "
                f"|eR|={diagnostics.strict_rotation_residual:.2e}\n"
                f"status={diagnostics.status}   "
                f"dmin={diagnostics.sigma_min_gap_normalized:.3e}\n"
                f"sigma_min,N={diagnostics.sigma_min_normalized:.5f}"
            )
            self._draw_penalty_chart(diagnostics.task_penalties)
    def close(self) -> None:
        self.running = False
        try: self.root.destroy(); self.wrist.destroy(); self.viewer.destroy()
        except tk.TclError: pass
