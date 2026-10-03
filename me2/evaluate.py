"""Score a trained voice command model on a held-out manifest.

Reports exact command accuracy - the intent and every slot that intent uses,
all correct at once - because that is what the device actually has to get
right. Intent-only accuracy flatters: five heads at 0.95 each is 0.77 overall.

Results are split by whether the audio is a recording of a person or
synthetic - text-to-speech and voice clones, the manifest's is_synthetic -
since most of the class dataset is synthetic and the real subset is the honest
read on how the model behaves in front of a person.

The loaded checkpoint is printed on every run. An earlier round of results was
wrong for a week because a glob sorted alphabetically and quietly returned
step_8000 in preference to step_16000; --step exists so a checkpoint can be
pinned rather than inferred.

Usage:
    python evaluate.py modelstore/vcm_hf --step 8000
    python evaluate.py modelstore/vcm_hf_s2 --step 7500 --split c_val
"""
import argparse
import collections
import json
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from dataloaders.vcm_dataloader import VoiceCommandDataset
from utils import find_latest_checkpoint


def load_run(run_dir: str | Path, step: int | None = None, device: str = "cpu"):
    """
    Load a trained module and the config it was trained under.

    Args:
        run_dir: Directory holding config.yaml and checkpoints/
        step: Pin a specific checkpoint step, newest by write time if None (default: None)
        device: Device to run on (default: 'cpu')

    Returns:
        Tuple of (module in eval mode, resolved config, checkpoint path)
    """
    run_dir = Path(run_dir)
    cfg = OmegaConf.load(run_dir / "config.yaml")
    # main.py saves the spec a run was trained against beside it; reading that
    # copy means a renamed or edited spec in the repo cannot change the labels
    if (run_dir / "commands.yaml").exists():
        cfg.module.spec_path = str(run_dir / "commands.yaml")
    OmegaConf.resolve(cfg.module)

    if step is None:
        ckpt_path = find_latest_checkpoint(run_dir / "checkpoints")
    else:
        ckpt_path = run_dir / "checkpoints" / f"checkpoint_step_{step}.ckpt"

    module = instantiate(cfg.module.module, cfg.module)
    state = torch.load(ckpt_path, weights_only=False, map_location="cpu")["state_dict"]
    module.load_state_dict(state)

    return module.to(device).eval(), cfg, ckpt_path


def spec_slots(module, intent: str) -> list[str]:
    """
    Slots an intent uses, named as the spec names them.

    The model carries digit-decomposed slots as two heads, but decode recombines
    them, so scoring happens on the single value the spec describes.

    Args:
        module: Trained module
        intent: Intent name

    Returns:
        Slot names at spec level, with tens/ones folded back into their base
    """
    digits = module.backbone.digit_slots
    parts = {name for pair in digits.values() for name in pair}
    # A label the model was never given (a test set from a wider spec) has no slots to read
    slots = module.backbone.intent_slots.get(intent, [])

    used = [name for name in slots if name not in parts]
    used += [base for base, pair in digits.items() if pair[0] in slots]

    return used


def evaluate(module, dataset, batch_size: int = 128, device: str = "cpu") -> list[dict]:
    """
    Decode every record in a dataset and pair it with its label.

    Args:
        module: Trained module
        dataset: Dataset to score, un-augmented and in manifest order
        batch_size: Items per forward pass (default: 128)
        device: Device to run on (default: 'cpu')

    Returns:
        One row per record holding the truth, the prediction, the source and
        whether the audio is synthetic
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    rows = []
    with torch.no_grad():
        for batch in loader:
            logits = module(batch["waveform"].to(device))
            commands = module.backbone.decode(logits)

            for command in commands:
                record = dataset.records[len(rows)]
                # A slot the row cannot give a value for is not graded, as in training
                truth = {"intent": record["intent"]}
                truth.update({
                    name: record["slots"].get(name)
                    for name in spec_slots(module, record["intent"])
                    if name not in record.get("unsupervise", ())
                })

                rows.append({
                    "model": record.get("model", "?"),
                    "synthetic": bool(record.get("is_synthetic", 0)),
                    "truth": truth,
                    "pred": command,
                })

    return rows


def summarise(rows: list[dict]) -> dict:
    """
    Reduce scored rows to the headline numbers.

    Args:
        rows: Output of evaluate

    Returns:
        Intent accuracy, exact command accuracy, none recall and per-slot counts
    """
    slot_hits = collections.Counter()
    slot_total = collections.Counter()
    intent_ok = exact_ok = none_total = none_ok = 0

    for row in rows:
        truth, pred = row["truth"], row["pred"]
        correct_intent = pred["intent"] == truth["intent"]
        intent_ok += correct_intent

        if truth["intent"] == "none":
            none_total += 1
            none_ok += correct_intent

        slots_ok = True
        for name, value in truth.items():
            if name == "intent":
                continue
            slot_total[name] += 1
            hit = correct_intent and pred.get(name) == value
            slot_hits[name] += hit
            slots_ok &= hit

        exact_ok += correct_intent and slots_ok

    total = max(len(rows), 1)

    return {
        "n": len(rows),
        "intent": intent_ok / total,
        "exact": exact_ok / total,
        "none_recall": none_ok / max(none_total, 1),
        "slots": {
            name: slot_hits[name] / slot_total[name]
            for name in sorted(slot_total) if slot_total[name]
        },
    }


def slot_errors(rows: list[dict], slot: str) -> collections.Counter:
    """
    Split a slot's errors into dropped and confused.

    A dropped slot means the model answered N/A when a value was present, which
    is a prior problem and responds to class weighting. A confused slot means it
    answered a different value, which is acoustic and does not.

    Args:
        rows: Output of evaluate
        slot: Slot name to break down

    Returns:
        Counts keyed by 'dropped', 'confused' and 'wrong_intent'
    """
    counts = collections.Counter()

    for row in rows:
        truth, pred = row["truth"], row["pred"]
        if slot not in truth or truth[slot] is None:
            continue

        if pred["intent"] != truth["intent"]:
            counts["wrong_intent"] += 1
        elif pred.get(slot) == truth[slot]:
            continue
        elif pred.get(slot) is None:
            counts["dropped"] += 1
        else:
            counts["confused"] += 1

    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run directories under modelstore/")
    ap.add_argument("--split", default="c_test")
    ap.add_argument("--step", type=int, default=None, help="pin a checkpoint step")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    for run in args.runs:
        module, cfg, ckpt_path = load_run(run, args.step, args.device)

        # Runs made before the spec was snapshotted fall back to the repo copy,
        # which is only safe while the label space has not moved under them
        spec_path = Path(run) / "commands.yaml"
        if not spec_path.exists():
            spec_path = Path("commands.yaml")
            print(f"  note: {run} has no spec snapshot, using {spec_path}")

        dataset = VoiceCommandDataset(
            data_dir=cfg.dataloader.dataset.data_dir,
            spec_path=str(spec_path),
            split=args.split,
            random_offset=False,
            augment=None,
        )
        rows = evaluate(module, dataset, args.batch_size, args.device)
        real = [r for r in rows if not r["synthetic"]]
        synth = [r for r in rows if r["synthetic"]]

        print(f"\n=== {Path(run).name}   {ckpt_path.name}")
        print(f"{'subset':10s} {'n':>6s} {'intent':>8s} {'exact':>8s} {'none':>8s}")
        for label, subset in (("all", rows), ("real", real), ("synthetic", synth)):
            s = summarise(subset)
            print(f"{label:10s} {s['n']:6d} {s['intent']:8.3f} {s['exact']:8.3f} {s['none_recall']:8.3f}")

        slots = summarise(real)["slots"]
        print("  slots (real):", "  ".join(f"{k}={v:.3f}" for k, v in slots.items()))
        for slot in slots:
            counts = slot_errors(real, slot)
            if counts:
                print(f"  {slot:8s} errors (real):", dict(counts))


if __name__ == "__main__":
    main()
