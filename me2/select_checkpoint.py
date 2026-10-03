"""Pick each run's checkpoint by its validation accuracy.

A run trained on a small set can pass its best before its last step: in 8,000
steps the training set is seen about 70 times. This scores every saved
checkpoint of each run on a validation split - exact command accuracy over
commands and out-of-scope rows alike, at argmax - and records the best, which
evaluation and export then pin as `run:step`. For fallback_hf that was
vcm_hf:8000 and vcm_hf_s2:7500.

Usage:
    python select_checkpoint.py modelstore/vcm_hf modelstore/vcm_hf_s2
"""
import argparse
import json
import re
from pathlib import Path

from dataloaders.vcm_dataloader import VoiceCommandDataset
from evaluate import evaluate, load_run, summarise

REPO = Path(__file__).resolve().parent


def saved_steps(run_dir: Path) -> list[int]:
    """Steps of a run's saved checkpoints, in order."""
    found = (re.search(r"step_(\d+)", p.name) for p in (run_dir / "checkpoints").glob("checkpoint_step_*.ckpt"))

    return sorted(int(m.group(1)) for m in found if m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run directories")
    ap.add_argument("--split", default="c_val")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="logs/selected_checkpoints.json")
    args = ap.parse_args()

    out = REPO / args.out
    selected = json.load(open(out)) if out.exists() else {}
    for run in args.runs:
        run_dir = Path(run)
        dataset, scores = None, {}
        for step in saved_steps(run_dir):
            module, cfg, _ = load_run(run_dir, step, args.device)
            if dataset is None:
                dataset = VoiceCommandDataset(data_dir=cfg.dataloader.dataset.data_dir,
                                              spec_path=str(run_dir / "commands.yaml"), split=args.split,
                                              random_offset=False, augment=None)
            summary = summarise(evaluate(module, dataset, 128, args.device))
            scores[step] = round(summary["exact"], 4)
            print(f"  {run_dir.name:16s} step {step:6,d}   {args.split} exact {summary['exact']:.4f}   "
                  f"intent {summary['intent']:.4f}   none recall {summary['none_recall']:.4f}", flush=True)

        # Ties go to the later checkpoint, trained longer for the same score
        best = max(scores, key=lambda step: (scores[step], step))
        selected[run_dir.name] = {"step": best, "split": args.split, "exact": scores[best], "by_step": scores}
        print(f"{run_dir.name}: step {best} ({args.split} exact {scores[best]:.4f})\n")

    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(selected, open(out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
