"""Stable scene/task identifiers shared across project subsystems."""

BLOCK_TO_PLATE_TASK = "Blue block -> plate"
BLOCK_TO_PLATE_STABLE_TASK = "Rectangular block -> tray without tilting"
CUBE_TO_PLATE_TASK = "Cube -> plate"
BALL_TO_PLATE_TASK = "Ball -> plate"
BALL_PLATE_THEN_BLOCK_PAPER_TASK = "Ball -> plate, then cube -> paper"
CYLINDER_TO_PLATE_TASK = "Cylinder -> plate"
CYLINDER_TO_PLATE_STABLE_TASK = "Cylinder -> tray without tilting"
CYLINDER_TO_NARROW_BOX_TASK = "Place the lying cylinder in the narrow box"
BOX_TO_NARROW_BOX_TASK = "Place the lying rectangular block in the narrow box"
LYING_BOX_TO_PLATE_TASK = "Lying rectangular block -> plate"
LYING_BOX_TO_REGION_TASK = "Lying rectangular block -> white region"
ADJUST_BOTTLE_TASK = "Stand up the fallen bottle"
CLICK_BELL_TASK = "Click the bell"
PRESS_BUTTON_TASK = "Press the button"
PRESS_POWER_STRIP_TASK = "Press the power strip button"
INSERT_OBJECT_TASK = "Insert the object"
BANANA_TO_PLATE_TASK = "Banana -> plate"
BANANA_TO_TRAY_TASK = "Banana -> tray"
STRAWBERRY_TO_TRAY_TASK = "Strawberry -> tray"
BANANA_PLATE_THEN_PEAR_BOWL_TASK = "Banana -> plate, then pear -> bowl"
PEAR_TO_PLATE_TASK = "Pear -> plate"
PEAR_TO_BOWL_TASK = "Pear -> bowl"
PEAR_TO_TRAY_TASK = "Pear -> tray"
BANANA_THEN_STRAWBERRY_TO_TRAY_TASK = (
    "Banana -> tray, then strawberry -> tray"
)
APPLE_THEN_BANANA_TO_TRAY_TASK = "Apple -> tray, then banana -> tray"
APPLE_TO_PLATE_TASK = "Apple -> plate"
APPLE_TO_BOWL_TASK = "Apple -> bowl"
LEMON_TO_PLATE_TASK = "Lemon -> plate"
LEMON_TO_BOWL_TASK = "Lemon -> bowl"
SPORTS_BALL_TO_PLATE_TASK = "Sports ball -> plate"
UPRIGHT_BOX_TO_PLATE_TASK = "Upright box -> plate"
UPRIGHT_BOX_TO_REGION_TASK = "Upright box -> white region"
FRUIT_RECEPTACLE_TASKS = (
    BANANA_TO_TRAY_TASK,
    STRAWBERRY_TO_TRAY_TASK,
    BANANA_THEN_STRAWBERRY_TO_TRAY_TASK,
)
FRUIT_TRAY_SEQUENCE_TASK_SPECS = {
    BANANA_THEN_STRAWBERRY_TO_TRAY_TASK: ("banana", "strawberry"),
    APPLE_THEN_BANANA_TO_TRAY_TASK: ("apple", "banana"),
}
PLACE_OBJECT_ON_PAPER_TASK = "Place the object on the paper region"
STACK_COLORED_BLOCKS_TASK = "Stack the red, green, and purple blocks"
YELLOW_RED_PURPLE_STACK_TASK = (
    "Stack the yellow, red, and purple blocks from bottom to top"
)
COLOR_BLOCK_STACK_TASK_SPECS = {
    YELLOW_RED_PURPLE_STACK_TASK: ("yellow", "red", "purple"),
}
COLOR_BLOCK_STACK_TASKS = tuple(COLOR_BLOCK_STACK_TASK_SPECS)
ROTATE_KNOB_TASK = "Rotate the knob clockwise 90 degrees"
WIPE_WHITEBOARD_TASK = "Wipe the whiteboard"
OPEN_CYLINDRICAL_POT_LID_TASK = "Open the cylindrical pot lid"
OPEN_HANDLE_POT_LID_TASK = "Open the handle pot lid"
CLOSE_CYLINDRICAL_POT_LID_TASK = "Close the cylindrical pot lid"
CLOSE_HANDLE_POT_LID_TASK = "Close the handle pot lid"
ROUND_HANDLE_POT_PEAR_SEQUENCE_TASK = (
    "Place the round-handle lid down, put the pear in the pot, then close it"
)
POT_LID_TASK_SPECS = {
    OPEN_CYLINDRICAL_POT_LID_TASK: ("cylindrical", False),
    OPEN_HANDLE_POT_LID_TASK: ("handle", False),
    CLOSE_CYLINDRICAL_POT_LID_TASK: ("cylindrical", True),
    CLOSE_HANDLE_POT_LID_TASK: ("handle", True),
}
POT_LID_TASKS = tuple(POT_LID_TASK_SPECS)


def pot_lid_task_spec(task: str) -> tuple[str, bool]:
    """Return ``(lid kind, closes pot)`` for one fixed-variant task."""
    try:
        return POT_LID_TASK_SPECS[task]
    except KeyError as error:
        raise ValueError(f"Not a fixed-variant pot-lid task: {task!r}") from error


def pot_lid_task_variant_index(task: str) -> int | None:
    """Return the scene variant index, or ``None`` for non-pot-lid tasks."""
    spec = POT_LID_TASK_SPECS.get(task)
    if spec is None:
        return None
    return 0 if spec[0] == "cylindrical" else 1

__all__ = (
    "ADJUST_BOTTLE_TASK",
    "APPLE_TO_BOWL_TASK",
    "APPLE_TO_PLATE_TASK",
    "BALL_TO_PLATE_TASK",
    "BALL_PLATE_THEN_BLOCK_PAPER_TASK",
    "BANANA_PLATE_THEN_PEAR_BOWL_TASK",
    "BANANA_TO_PLATE_TASK",
    "BANANA_TO_TRAY_TASK",
    "BANANA_THEN_STRAWBERRY_TO_TRAY_TASK",
    "APPLE_THEN_BANANA_TO_TRAY_TASK",
    "BLOCK_TO_PLATE_TASK",
    "BLOCK_TO_PLATE_STABLE_TASK",
    "BOX_TO_NARROW_BOX_TASK",
    "CUBE_TO_PLATE_TASK",
    "CLICK_BELL_TASK",
    "CYLINDER_TO_PLATE_TASK",
    "CYLINDER_TO_PLATE_STABLE_TASK",
    "CYLINDER_TO_NARROW_BOX_TASK",
    "INSERT_OBJECT_TASK",
    "LEMON_TO_BOWL_TASK",
    "LEMON_TO_PLATE_TASK",
    "LYING_BOX_TO_PLATE_TASK",
    "LYING_BOX_TO_REGION_TASK",
    "FRUIT_RECEPTACLE_TASKS",
    "FRUIT_TRAY_SEQUENCE_TASK_SPECS",
    "PEAR_TO_BOWL_TASK",
    "PEAR_TO_PLATE_TASK",
    "PEAR_TO_TRAY_TASK",
    "STRAWBERRY_TO_TRAY_TASK",
    "PLACE_OBJECT_ON_PAPER_TASK",
    "STACK_COLORED_BLOCKS_TASK",
    "COLOR_BLOCK_STACK_TASK_SPECS",
    "COLOR_BLOCK_STACK_TASKS",
    "YELLOW_RED_PURPLE_STACK_TASK",
    "ROTATE_KNOB_TASK",
    "WIPE_WHITEBOARD_TASK",
    "SPORTS_BALL_TO_PLATE_TASK",
    "OPEN_CYLINDRICAL_POT_LID_TASK",
    "OPEN_HANDLE_POT_LID_TASK",
    "CLOSE_CYLINDRICAL_POT_LID_TASK",
    "CLOSE_HANDLE_POT_LID_TASK",
    "ROUND_HANDLE_POT_PEAR_SEQUENCE_TASK",
    "POT_LID_TASK_SPECS",
    "POT_LID_TASKS",
    "pot_lid_task_spec",
    "pot_lid_task_variant_index",
    "PRESS_BUTTON_TASK",
    "PRESS_POWER_STRIP_TASK",
    "UPRIGHT_BOX_TO_PLATE_TASK",
    "UPRIGHT_BOX_TO_REGION_TASK",
)
