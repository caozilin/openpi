"""Offline demonstration collection and dataset interchange."""

from .task_sets import (
    DATASET_SCENES,
    CONTEXTUAL_SUBCATEGORY,
    GEOMETRIC_BASIS_SUBCATEGORY,
    MULTI_SUBTASK_SUBCATEGORY,
    TRAINING_DATASET_SCENES,
    TOLERANCE_DISCRIMINATION_DATASET_SCENES,
    UNSEEN_DATASET_SCENES,
    runtime_scenes_for_group,
    resolve_training_task,
    seeded_task_selection,
    task_instruction_record,
    training_tasks_for_subcategory,
    weighted_training_plan,
)

__all__ = (
    "CollectionConfig",
    "DATASET_SCENES",
    "CONTEXTUAL_SUBCATEGORY",
    "GEOMETRIC_BASIS_SUBCATEGORY",
    "MULTI_SUBTASK_SUBCATEGORY",
    "TRAINING_DATASET_SCENES",
    "TOLERANCE_DISCRIMINATION_DATASET_SCENES",
    "UNSEEN_DATASET_SCENES",
    "collect_batch",
    "runtime_scenes_for_group",
    "resolve_training_task",
    "seeded_task_selection",
    "task_instruction_record",
    "training_tasks_for_subcategory",
    "weighted_training_plan",
)


def __getattr__(name: str):
    if name in {"CollectionConfig", "collect_batch"}:
        from .collector import CollectionConfig, collect_batch

        globals()["CollectionConfig"] = CollectionConfig
        globals()["collect_batch"] = collect_batch
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
