import yaml
from pathlib import Path


def load_command_spec(spec_path: str | Path) -> dict:
    """
    Load the command specification that defines the label space.

    Args:
        spec_path: Path to commands.yaml

    Returns:
        Parsed specification dictionary
    """
    return yaml.safe_load(open(spec_path))


def slot_values(spec: dict, slot_name: str) -> list:
    """
    List the values a slot can take, expanding numeric slots into their range.

    Args:
        spec: Parsed command specification
        slot_name: Name of the slot

    Returns:
        Ordered list of slot values
    """
    slot = spec["slots"][slot_name]

    if slot.get("type") == "number":
        start, stop = slot["range"]
        return list(range(start, stop + 1, slot.get("step", 1)))

    return slot["values"]


def digit_slots(spec: dict) -> dict[str, tuple[str, str]]:
    """
    Map each digit-decomposed slot to the pair of heads that represent it.

    Args:
        spec: Parsed command specification

    Returns:
        Mapping of slot name to (tens head, ones head)
    """
    return {
        name: (f"{name}_tens", f"{name}_ones")
        for name, slot in spec["slots"].items()
        if slot.get("decompose") == "digits"
    }


def build_label_space(spec: dict) -> tuple[list[str], dict[str, list]]:
    """
    Build the ordered class list for every head.

    Every slot head carries an extra N/A class for utterances where the slot
    does not apply, so the model has a fixed output shape across all intents.

    Args:
        spec: Parsed command specification

    Returns:
        Tuple of (intent names, mapping of slot name to its class list)
    """
    intent_names = list(spec["intents"].keys())
    decomposed = digit_slots(spec)

    slot_classes = {}
    for name in spec["slots"]:
        if name in decomposed:
            tens, ones = decomposed[name]
            highest = max(slot_values(spec, name))
            slot_classes[tens] = list(range(highest // 10 + 1)) + ["N/A"]
            slot_classes[ones] = list(range(10)) + ["N/A"]
        else:
            slot_classes[name] = slot_values(spec, name) + ["N/A"]

    return intent_names, slot_classes


def active_slots(spec: dict, intent_name: str) -> list[str]:
    """
    List the slot heads an intent actually uses.

    Used both to mask the training loss and to decide which heads to read at
    inference time, so a light command never reports a timer number.

    Args:
        spec: Parsed command specification
        intent_name: Name of the intent

    Returns:
        Slot names belonging to that intent
    """
    decomposed = digit_slots(spec)

    names = []
    for name in spec["intents"][intent_name].get("slots") or []:
        names.extend(decomposed[name] if name in decomposed else [name])

    return names
