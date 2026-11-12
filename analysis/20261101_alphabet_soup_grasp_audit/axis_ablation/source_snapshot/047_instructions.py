"""Task-level natural-language instruction sets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..tasks.catalog import (
    ADJUST_BOTTLE_TASK,
    BALL_TO_PLATE_TASK,
    BALL_PLATE_THEN_BLOCK_PAPER_TASK,
    APPLE_THEN_BANANA_TO_TRAY_TASK,
    BLOCK_TO_PLATE_TASK,
    BLOCK_TO_PLATE_STABLE_TASK,
    BOX_TO_NARROW_BOX_TASK,
    CUBE_TO_PLATE_TASK,
    BANANA_PLATE_THEN_PEAR_BOWL_TASK,
    BANANA_TO_PLATE_TASK,
    BANANA_TO_TRAY_TASK,
    BANANA_THEN_STRAWBERRY_TO_TRAY_TASK,
    CLICK_BELL_TASK,
    CYLINDER_TO_PLATE_TASK,
    CYLINDER_TO_PLATE_STABLE_TASK,
    CYLINDER_TO_NARROW_BOX_TASK,
    INSERT_OBJECT_TASK,
    LYING_BOX_TO_PLATE_TASK,
    LYING_BOX_TO_REGION_TASK,
    PEAR_TO_BOWL_TASK,
    PEAR_TO_PLATE_TASK,
    PEAR_TO_TRAY_TASK,
    STRAWBERRY_TO_TRAY_TASK,
    PLACE_OBJECT_ON_PAPER_TASK,
    ROTATE_KNOB_TASK,
    WIPE_WHITEBOARD_TASK,
    STACK_COLORED_BLOCKS_TASK,
    COLOR_BLOCK_STACK_TASK_SPECS,
    FRUIT_TRAY_SEQUENCE_TASK_SPECS,
    CLOSE_CYLINDRICAL_POT_LID_TASK,
    CLOSE_HANDLE_POT_LID_TASK,
    OPEN_CYLINDRICAL_POT_LID_TASK,
    OPEN_HANDLE_POT_LID_TASK,
    ROUND_HANDLE_POT_PEAR_SEQUENCE_TASK,
    PRESS_BUTTON_TASK,
    PRESS_POWER_STRIP_TASK,
)


TASK_LANGUAGE = {
    BLOCK_TO_PLATE_TASK: "Place the rectangular block on the tray",
    BLOCK_TO_PLATE_STABLE_TASK: (
        "Place the rectangular block on the tray without tilting it"
    ),
    CUBE_TO_PLATE_TASK: "Place the cube in the tray",
    BALL_TO_PLATE_TASK: "Place the ball on the tray",
    BALL_PLATE_THEN_BLOCK_PAPER_TASK: "Place the ball on the plate, then place the cube on the matching white region",
    CYLINDER_TO_PLATE_TASK: "Place the cylinder on the tray",
    CYLINDER_TO_PLATE_STABLE_TASK: (
        "Place the cylinder on the tray without tilting it"
    ),
    CYLINDER_TO_NARROW_BOX_TASK: "Place the cylinder in the box",
    BOX_TO_NARROW_BOX_TASK: "Place the rectangular block in the box",
    ADJUST_BOTTLE_TASK: "Stand the bottle upright",
    CLICK_BELL_TASK: "Press the bell",
    PRESS_BUTTON_TASK: "Press the button",
    PRESS_POWER_STRIP_TASK: "Press the power strip switch",
    INSERT_OBJECT_TASK: "Insert the object into the matching hole",
    LYING_BOX_TO_PLATE_TASK: "Place the lying rectangular block in the tray",
    LYING_BOX_TO_REGION_TASK: "Align the lying rectangular block with the white region",
    BANANA_TO_PLATE_TASK: "Place the banana on the plate",
    BANANA_TO_TRAY_TASK: "Place the banana in the tray",
    BANANA_THEN_STRAWBERRY_TO_TRAY_TASK: "Place the banana in the tray, then place the strawberry in the tray",
    APPLE_THEN_BANANA_TO_TRAY_TASK: (
        "Place the apple on the tray, then place the banana on the tray"
    ),
    BANANA_PLATE_THEN_PEAR_BOWL_TASK: "Place the banana on the plate, then place the pear in the bowl",
    PEAR_TO_PLATE_TASK: "Place the pear on the plate",
    PEAR_TO_TRAY_TASK: "Place the pear in the tray",
    STRAWBERRY_TO_TRAY_TASK: "Place the strawberry in the tray",
    PEAR_TO_BOWL_TASK: "Place the pear in the bowl",
    PLACE_OBJECT_ON_PAPER_TASK: "Place the object in the white target area",
    ROTATE_KNOB_TASK: "Rotate the knob 90 degrees clockwise",
    WIPE_WHITEBOARD_TASK: "Wipe the whiteboard",
    STACK_COLORED_BLOCKS_TASK: "Stack the red, green, and purple blocks in order on the white paper target",
    OPEN_CYLINDRICAL_POT_LID_TASK: "Remove the cylindrical lid from the pot and place it on the table beside the pot",
    OPEN_HANDLE_POT_LID_TASK: "Remove the handle lid from the pot and place it on the table beside the pot",
    CLOSE_CYLINDRICAL_POT_LID_TASK: "Replace the lid on the pot",
    CLOSE_HANDLE_POT_LID_TASK: "Replace the lid on the pot",
    ROUND_HANDLE_POT_PEAR_SEQUENCE_TASK: (
        "Remove the lid from the pot and place it on the table, then place "
        "the pear in the pot, and finally replace the lid on the pot"
    ),
}
TASK_LANGUAGE.update({
    task: (
        "Place the red block squarely on the yellow block, then place the "
        "purple block squarely on the red block"
    )
    for task in COLOR_BLOCK_STACK_TASK_SPECS
})

TASK_INSTRUCTION_VARIANTS = {
    BLOCK_TO_PLATE_TASK: (
        "Place the rectangular block in the tray",
        "Put the upright rectangular block into the tray",
        "Move the rectangular block to the tray",
        "Pick up the upright block and place it in the tray",
        "Transfer the rectangular block into the tray",
        "Place the tall rectangular block inside the tray",
        "Put the tall block into the tray",
        "Move the upright rectangular block into the tray",
        "Pick up the tall rectangular block and set it in the tray",
        "Transfer the upright block to the tray",
    ),
    CUBE_TO_PLATE_TASK: (
        "Place the cube in the tray",
        "Pick up the cube and put it into the tray",
        "Move the cube to the tray",
    ),
    CLICK_BELL_TASK: (
        "Press the bell",
        "Click the bell",
        "Push down the bell button",
    ),
    PRESS_BUTTON_TASK: (
        "Press the button",
        "Push down the industrial button",
        "Depress the mushroom button",
    ),
    PRESS_POWER_STRIP_TASK: (
        "Press the power strip switch",
        "Push the button on the power strip",
        "Depress the power strip power button",
    ),
    INSERT_OBJECT_TASK: (
        "Insert the object into the matching hole",
        "Pick up the object and place it into the matching container opening",
        "Align the object with the matching slot and insert it",
    ),
    CYLINDER_TO_NARROW_BOX_TASK: (
        "Place the lying cylinder in the narrow wooden box",
        "Put the horizontal cylinder into the narrow wooden box",
        "Pick up the cylinder and place it inside the matching wood box",
    ),
    BOX_TO_NARROW_BOX_TASK: (
        "Place the lying rectangular block in the narrow wooden box",
        "Put the horizontal rectangular block into the narrow wooden box",
        "Pick up the rectangular block and place it inside the matching wood box",
    ),
    LYING_BOX_TO_PLATE_TASK: (
        "Place the lying rectangular block in the tray",
        "Put the horizontal rectangular block into the tray",
        "Move the lying block to the tray",
    ),
    LYING_BOX_TO_REGION_TASK: (
        "Align the lying rectangular block with the white region",
        "Place the horizontal rectangular block inside the white target",
        "Move the lying block completely into the matching white region",
    ),
    BANANA_TO_PLATE_TASK: (
        "Place the banana on the plate",
        "Pick up the banana and put it on the plate",
    ),
    BANANA_TO_TRAY_TASK: (
        "Place the banana in the tray",
        "Pick up the banana and put it into the tray",
    ),
    BANANA_PLATE_THEN_PEAR_BOWL_TASK: (
        "Place the banana on the plate, then place the pear in the bowl",
        "First put the banana on the plate, then put the pear into the bowl",
        "Move the banana to the plate and afterward place the pear in the bowl",
    ),
    PEAR_TO_PLATE_TASK: (
        "Place the pear on the plate",
        "Pick up the pear and put it on the plate",
    ),
    PEAR_TO_TRAY_TASK: (
        "Place the pear in the tray",
        "Pick up the pear and put it into the tray",
    ),
    PEAR_TO_BOWL_TASK: (
        "Place the pear in the bowl",
        "Pick up the pear and put it into the bowl",
    ),
    PLACE_OBJECT_ON_PAPER_TASK: (
        "Place the object inside the matching white paper region",
        "Pick up the colored object and align it with the white shape",
        "Move the object completely within the matching paper outline",
    ),
    WIPE_WHITEBOARD_TASK: (
        "Wipe the whiteboard",
        "Erase the black scribbles from the whiteboard",
        "Pick up the eraser and wipe the whiteboard clean",
    ),
    ROTATE_KNOB_TASK: (
        "Rotate the knob clockwise by 90 degrees",
        "Turn the tabletop knob 90 degrees clockwise",
        "Grip the knob and rotate it one quarter turn clockwise",
    ),
    STACK_COLORED_BLOCKS_TASK: (
        "Stack the red, green, and purple blocks in order on the white paper target",
        "Place the red block first, the green block second, and the purple block on top",
        "Build a red-green-purple tower centered on the white paper square",
    ),
    BALL_PLATE_THEN_BLOCK_PAPER_TASK: (
        "Place the ball on the plate, then place the cube on the matching white region",
        "First put the ball on the plate, then align the cube with the white target",
        "Move the ball to the plate and afterward place the cube inside the matching white region",
    ),
    OPEN_CYLINDRICAL_POT_LID_TASK: (
        "Remove the cylindrical lid from the pot and place it on the table beside the pot",
        "Open the cylindrical pot by setting its lid down beside it",
        "Lift the cylindrical pot lid and place it on the tabletop next to the pot",
    ),
    OPEN_HANDLE_POT_LID_TASK: (
        "Remove the handle lid from the pot and place it on the table beside the pot",
        "Open the handle-lid pot by setting its lid down beside it",
        "Lift the pot lid by its bar handle and place it next to the pot",
    ),
    CLOSE_CYLINDRICAL_POT_LID_TASK: (
        "Pick up the cylindrical lid beside the pot and place it on the pot",
        "Close the pot with the cylindrical lid on the table",
        "Lift the cylindrical lid from the tabletop and cover the pot",
    ),
    CLOSE_HANDLE_POT_LID_TASK: (
        "Pick up the handle lid beside the pot and place it on the pot",
        "Close the pot with the handle lid on the table",
        "Lift the lid by its bar handle and cover the pot",
    ),
}
TASK_INSTRUCTION_VARIANTS.update({
    BLOCK_TO_PLATE_STABLE_TASK: (
        "Place the rectangular block on the tray without tilting it",
    ),
    CYLINDER_TO_PLATE_STABLE_TASK: (
        "Place the cylinder on the tray without tilting it",
    ),
    STRAWBERRY_TO_TRAY_TASK: (
        "Place the strawberry in the tray",
        "Pick up the strawberry and put it into the tray",
    ),
    BANANA_THEN_STRAWBERRY_TO_TRAY_TASK: (
        "Place the banana in the tray, then place the strawberry in the tray",
        "First put the banana into the tray, then put the strawberry into the tray",
    ),
    APPLE_THEN_BANANA_TO_TRAY_TASK: (
        "Place the apple on the tray, then place the banana on the tray",
    ),
    ROUND_HANDLE_POT_PEAR_SEQUENCE_TASK: (
        "Remove the lid from the pot and place it on the table, then place "
        "the pear in the pot, and finally replace the lid on the pot",
    ),
})
TASK_INSTRUCTION_VARIANTS.update({
    task: (
        "Place the red block squarely on the yellow block, then place the "
        "purple block squarely on the red block",
    )
    for task, order in COLOR_BLOCK_STACK_TASK_SPECS.items()
})


_FRUIT_TASK_SPECS = {
    BANANA_TO_TRAY_TASK: ("banana", "yellow", "tray", "in"),
    STRAWBERRY_TO_TRAY_TASK: ("strawberry", "red", "tray", "in"),
    PEAR_TO_TRAY_TASK: ("pear", "green", "tray", "in"),
    BANANA_TO_PLATE_TASK: ("banana", "yellow", "plate", "on"),
    PEAR_TO_PLATE_TASK: ("pear", "green", "plate", "on"),
    PEAR_TO_BOWL_TASK: ("pear", "green", "bowl", "in"),
}


def _include_color(seed: int, task_id: str, entity: str) -> bool:
    digest = hashlib.sha256(
        f"tasktol-instruction-color:{seed}:{task_id}:{entity}".encode("utf-8")
    ).digest()
    return bool(int.from_bytes(digest[:8], "little") & 1)


def _colored_noun(
    noun: str,
    color: str,
    *,
    seed: int,
    task_id: str,
    entity: str,
) -> str:
    return (
        f"{color} {noun}"
        if _include_color(seed, task_id, entity)
        else noun
    )


def resolve_episode_instruction(
    base_instruction: str,
    *,
    task_id: str,
    scene: str,
    seed: int,
    active_body_names: tuple[str, ...] = (),
    geometry_color: str | None = None,
) -> str:
    """Apply seeded, image-consistent language rules to one episode."""
    if task_id == ADJUST_BOTTLE_TASK:
        return "Stand the bottle upright"

    if task_id in (
        LYING_BOX_TO_PLATE_TASK,
        LYING_BOX_TO_REGION_TASK,
    ):
        if geometry_color is None:
            return base_instruction
        geometry = _colored_noun(
            "lying rectangular block", geometry_color,
            seed=seed, task_id=task_id, entity="geometry",
        )
        if task_id == LYING_BOX_TO_PLATE_TASK:
            return f"Place the {geometry} in the tray"
        return f"Align the {geometry} with the white region"

    if scene.startswith(("Geometry Plate -", "Geometry Region -")) or (
        scene in ("Cylinder to Narrow Box", "Lying Box to Narrow Box")
    ):
        descriptor = {
            "Geometry Plate - Upright Box": "rectangular block",
            "Geometry Plate - Lying Box": "rectangular block",
            "Geometry Plate - Cube": "cube",
            "Geometry Plate - Ball": "ball",
            "Geometry Plate - Upright Cylinder": "cylinder",
            "Geometry Plate - Lying Cylinder": "cylinder",
            "Geometry Region - Upright Box": "rectangular block",
            "Geometry Region - Lying Box": "rectangular block",
            "Geometry Region - Cube": "cube",
            "Geometry Region - Upright Cylinder": "cylinder",
            "Geometry Region - Lying Cylinder": "cylinder",
            "Cylinder to Narrow Box": "cylinder",
            "Lying Box to Narrow Box": "rectangular block",
        }.get(scene)
        if descriptor is None or geometry_color is None:
            return base_instruction
        geometry = _colored_noun(
            descriptor, geometry_color,
            seed=seed, task_id=task_id, entity="geometry",
        )
        if scene.startswith("Geometry Plate -"):
            suffix = (
                " on the tray without tilting it"
                if task_id in (
                    BLOCK_TO_PLATE_STABLE_TASK,
                    CYLINDER_TO_PLATE_STABLE_TASK,
                )
                else " on the tray"
            )
            return f"Place the {geometry}{suffix}"
        if scene in ("Cylinder to Narrow Box", "Lying Box to Narrow Box"):
            return f"Place the {geometry} in the box"
        return f"Place the {geometry} in the white target area"

    if task_id == BANANA_PLATE_THEN_PEAR_BOWL_TASK:
        active = tuple(active_body_names)
        plate_body = next((
            name for name in active if name.startswith("fruit_task_plate_")
        ), None)
        bowl_body = next((
            name for name in active if name.startswith("fruit_task_bowl_")
        ), None)
        if plate_body is None or bowl_body is None:
            return base_instruction
        banana = _colored_noun(
            "banana", "yellow",
            seed=seed, task_id=task_id, entity="banana",
        )
        pear = _colored_noun(
            "pear", "green",
            seed=seed, task_id=task_id, entity="pear",
        )
        plate = _colored_noun(
            "plate", plate_body.rsplit("_", 1)[-1],
            seed=seed, task_id=task_id, entity="plate",
        )
        bowl = _colored_noun(
            "bowl", bowl_body.rsplit("_", 1)[-1],
            seed=seed, task_id=task_id, entity="bowl",
        )
        return (
            f"Place the {banana} on the {plate}, then place the {pear} "
            f"in the {bowl}"
        )

    fruit_sequence = FRUIT_TRAY_SEQUENCE_TASK_SPECS.get(task_id)
    if fruit_sequence is not None:
        first, second = fruit_sequence
        colors = {"banana": "yellow", "strawberry": "red", "apple": "red"}
        first_noun = _colored_noun(
            first, colors[first], seed=seed, task_id=task_id, entity=first,
        )
        second_noun = _colored_noun(
            second, colors[second], seed=seed, task_id=task_id, entity=second,
        )
        return (
            f"Place the {first_noun} on the tray, then place the "
            f"{second_noun} on the tray"
        )

    if task_id == ROUND_HANDLE_POT_PEAR_SEQUENCE_TASK:
        pear = _colored_noun(
            "pear", "green",
            seed=seed, task_id=task_id, entity="pear",
        )
        return (
            "Remove the lid from the pot and place it on the table, then "
            f"place the {pear} in the pot, and finally replace the lid on "
            "the pot"
        )

    fruit_spec = _FRUIT_TASK_SPECS.get(task_id)
    if fruit_spec is not None:
        fruit_name, fruit_color, receptacle_name, preposition = fruit_spec
        fruit = _colored_noun(
            fruit_name, fruit_color,
            seed=seed, task_id=task_id, entity="fruit",
        )
        receptacle_prefix = (
            "tray_khaki_"
            if receptacle_name == "tray"
            else f"fruit_task_{receptacle_name}_"
        )
        receptacle_body = next((
            name for name in active_body_names
            if name.startswith(receptacle_prefix)
        ), None)
        if receptacle_body is None:
            return base_instruction
        receptacle = (
            "tray"
            if receptacle_name == "tray"
            else _colored_noun(
                receptacle_name, receptacle_body.rsplit("_", 1)[-1],
                seed=seed, task_id=task_id, entity=receptacle_name,
            )
        )
        return f"Place the {fruit} {preposition} the {receptacle}"

    return base_instruction


def validate_instructions(task_id: str, instructions: list[str] | tuple[str, ...]) -> list[str]:
    cleaned = [str(instruction).strip() for instruction in instructions]
    if not cleaned:
        raise ValueError(f"Task {task_id!r} requires at least one instruction")
    for instruction in cleaned:
        if not instruction:
            raise ValueError(f"Task {task_id!r} contains an empty instruction")
        if not instruction.isascii():
            raise ValueError(f"Task {task_id!r} instructions must be English ASCII text")
        if instruction.endswith((".", "?", "!")):
            raise ValueError(f"Task {task_id!r} instructions must not use terminal punctuation")
    return cleaned


def load_or_create_instructions(output_dir: Path, task_id: str) -> tuple[Path, list[str]]:
    path = output_dir / "instructions.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("task_id") != task_id:
            raise ValueError("Instruction file belongs to a different task")
        instructions = validate_instructions(task_id, payload.get("instructions", ()))
    else:
        defaults = TASK_INSTRUCTION_VARIANTS.get(task_id, (TASK_LANGUAGE[task_id],))
        instructions = validate_instructions(task_id, defaults)
        path.write_text(
            json.dumps({"task_id": task_id, "instructions": instructions}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return path, instructions


for _task_id, _instruction in TASK_LANGUAGE.items():
    validate_instructions(_task_id, (_instruction,))
