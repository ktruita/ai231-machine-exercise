"""Score models on Fluent Speech Commands' test split - a corpus they never saw.

Every figure so far came from held-out parts of corpora the models trained on:
other speakers, same recording set-up. FSC's test split is 10 speakers, a
corpus, a microphone set-up and a phrasing style no run has touched, so this is
the first cross-corpus measure of generality.

The split is never ingested (see add_fsc.py), so it stays cross-corpus even
after FSC's train split is added, and before/after numbers compare directly.

Reported per model:
    capability   exact command on the utterances that map to a spec command
    rejection    share of heating, language and fetching commands left alone -
                 command-shaped speech the device should not act on
    each at argmax and at the demo threshold of 0.6

Streams the parquet 128 rows at a time, so memory stays small.

Usage:
    python evaluate_fsc.py modelstore/vcm_stop modelstore/vcm_mined
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from add_fsc import LABELS, decode_audio, iter_rows, map_label
from evaluate import load_run, spec_slots

REPO = Path(__file__).resolve().parent
DEMO_THRESHOLD = 0.6
NUM_SAMPLES = 96000


def fit(audio: np.ndarray) -> np.ndarray:
    """Left-align and zero-pad to the model window, as the test split is scored."""
    audio = audio[:NUM_SAMPLES]

    return np.pad(audio, (0, NUM_SAMPLES - len(audio))).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--fsc", default="data/fsc")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    models = {Path(r).name: load_run(r, None, args.device)[0] for r in args.runs}
    results = {name: [] for name in models}
    skipped = Counter()

    batch, meta = [], []
    rows = iter_rows(str(REPO / args.fsc / "test-*.parquet"), LABELS + ["audio"])
    for row in rows:
        label = map_label(row["action"], row["object"], row["location"], row["transcription"])
        if label is None:
            skipped["unmapped"] += 1
            continue
        audio = decode_audio(row["audio"])
        if audio is None:
            skipped["unreadable"] += 1
            continue
        batch.append(fit(audio))
        meta.append((label, row))
        if len(batch) == 128:
            score(models, batch, meta, results, args.device)
            batch, meta = [], []
    if batch:
        score(models, batch, meta, results, args.device)

    n = len(next(iter(results.values())))
    print(f"FSC test: {n:,} utterances scored, 10 speakers never heard in training   skipped {dict(skipped)}\n")

    for name, scored in results.items():
        pos = [s for s in scored if s["truth"]["intent"] != "none"]
        neg = [s for s in scored if s["truth"]["intent"] == "none"]
        acts = lambda s, t: s["pred"]["intent"] != "none" and s["p"] >= t

        print(f"=== {name}")
        for label, t in (("argmax", 0.0), (f"threshold {DEMO_THRESHOLD}", DEMO_THRESHOLD)):
            cap = np.mean([s["exact"] and acts(s, t) for s in pos])
            intent = np.mean([s["pred"]["intent"] == s["truth"]["intent"] and acts(s, t) for s in pos])
            wrong = np.mean([acts(s, t) and not s["exact"] for s in pos])
            reject = np.mean([not acts(s, t) for s in neg])
            print(f"  {label:14s} capability {cap:.3f}  intent {intent:.3f}  wrong action {wrong:.3f}   "
                  f"hard negatives left alone {reject:.3f}")

        by_combo = defaultdict(list)
        for s in pos:
            by_combo[s["combo"]].append(s["exact"])
        print("  capability by FSC command (argmax):")
        for combo, hits in sorted(by_combo.items(), key=lambda x: np.mean(x[1])):
            print(f"    {combo:34s} {np.mean(hits):.3f}  (n={len(hits)})")

        rooms = defaultdict(list)
        for s in pos:
            if s["truth"]["intent"] == "light.set" and s["truth"].get("room"):
                said = "washroom" if "washroom" in s["text"].lower() else s["truth"]["room"]
                rooms[said].append(s["pred"].get("room") == s["truth"]["room"])
        print("  room read correctly, by the word spoken:",
              {k: f"{np.mean(v):.3f} (n={len(v)})" for k, v in sorted(rooms.items())})

        fa = Counter(s["pred"]["intent"] for s in neg if s["pred"]["intent"] != "none")
        kind = defaultdict(list)
        for s in neg:
            kind[s["combo"].split(" ")[0] if "heat" not in s["combo"] else "heat"].append(
                s["pred"]["intent"] == "none")
        print(f"  hard negatives left alone by kind: {{{', '.join(f'{k}: {np.mean(v):.3f}' for k, v in sorted(kind.items()))}}}")
        print(f"  false accepts go to: {dict(fa.most_common())}\n")


def score(models: dict, batch: list, meta: list, results: dict, device: str) -> None:
    """Run every model on one batch and record prediction, confidence and exactness."""
    x = torch.from_numpy(np.stack(batch)).to(device)
    with torch.inference_mode():
        for name, module in models.items():
            logits = module(x)
            commands = module.backbone.decode(logits)
            p = torch.softmax(logits["intent"].double(), dim=-1).max(dim=-1).values.cpu().tolist()
            for k, ((intent, slots), row) in enumerate(meta):
                truth = {"intent": intent}
                truth.update({n: slots.get(n) for n in spec_slots(module, intent)})
                pred = commands[k]
                exact = pred["intent"] == intent and all(
                    pred.get(n) == v for n, v in truth.items() if n != "intent")
                results[name].append({"truth": truth, "pred": pred, "p": p[k], "exact": exact,
                                      "text": row["transcription"],
                                      "combo": f"{row['action']} {row['object']} {row['location']}"})


if __name__ == "__main__":
    main()
