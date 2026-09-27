"""Per-number accuracy of the timer reading on real speech, and the prior that helps it.

A timer number is read by two heads, tens and ones, and the misreadings people
notice sit in the teens: "fifteen" and "fifty" differ only in their ending, and
a teen's ones digit is a clipped stem ("fif-", "thir-") rather than the digit
word. This scores every held-out real timer command number by number and sorts
the misreadings by kind:

    ending    a teen read as its -ty twin, or the reverse (15 <-> 50)
    ones      right tens digit, wrong ones digit (15 -> 16)
    tens      right ones digit, wrong tens digit (25 -> 35)
    both      both digits wrong, and not a teen/-ty pair
    none      a digit head chose N/A, so no number was read

It also checks how often the right number is the runner-up, which is what a
"15 min (or 50?)" prompt in the demo would recover.

The number heads are read directly, whatever intent the model chose, so this
measures hearing the number rather than routing the command. The test and
validation splits are pooled - rare numbers have a handful of examples in
either - and neither was trained on.

With --prior it also tunes number_prior.py: every valid number is scored from
both heads together plus weight x log-prior, the weight that reads the
validation split best is chosen, and the test split - which the choice never
saw - reports it. --write-prior saves the prior for the demo.

Usage:
    python evaluate_numbers.py modelstore/vcm_mined_fsc modelstore/vcm_mined_fsc_s2
    python evaluate_numbers.py "modelstore/vcm_mined_fsc+modelstore/vcm_mined_fsc_s2@-0.2"
    python evaluate_numbers.py <runs> --prior --write-prior
"""
import argparse
import json
import wave
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import number_prior
from dataloaders.vcm_dataloader import VoiceCommandDataset
from eval_history import CLIPS
from evaluate import spec_slots
from evaluate_benchmark import load_spec
from evaluate_fsc import fit

REPO = Path(__file__).resolve().parent
TEEN_TY = {13: 30, 14: 40, 15: 50, 16: 60}
WEIGHTS = np.round(np.arange(0.0, 2.01, 0.1), 2)
SMOOTHING = 1.0


def kind(truth: int, read: int | None) -> str:
    """Name the kind of misreading, as listed in the module docstring."""
    if read is None:
        return "none"
    if TEEN_TY.get(truth) == read or TEEN_TY.get(read) == truth:
        return "ending"
    if truth // 10 == read // 10:
        return "ones"
    if truth % 10 == read % 10:
        return "tens"

    return "both"


def dataset_batches(dataset, indices: list, batch: int = 128):
    """Yield (waveforms, records) from a VoiceCommandDataset."""
    for start in range(0, len(indices), batch):
        idx = indices[start:start + batch]
        yield [dataset[i]["waveform"].numpy() for i in idx], [dataset.records[i] for i in idx]


def clip_batches():
    """Yield Ken's timer recordings as one batch, labelled like manifest records, if they are here."""
    waves, records = [], []
    for stem, want in CLIPS.items():
        path = REPO / "my_audio_samples/wav" / f"{stem}.wav"
        if want["intent"] != "timer.set" or not path.exists():
            continue
        with wave.open(str(path)) as handle:
            audio = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(np.float32) / 32768
        waves.append(fit(audio))
        records.append({"file": stem, "split": "clip", "intent": "timer.set",
                        "slots": {"number": want["number"], "unit": want["unit"]}})
    if waves:
        yield waves, records


def read_batches(module, batches, device: str) -> list[dict]:
    """
    Read the number, the unit and the whole command for each utterance.

    Args:
        module: Trained module or ensemble
        batches: Iterable of (waveforms, records)
        device: Device to run on

    Returns:
        One dict per utterance: the number said and read, the top two
        numbers with their probabilities, whether the unit and the whole
        command were right, and every number's joint log-probability
    """
    backbone = module.backbone
    tens, ones = backbone.digit_slots["number"]
    out = []

    with torch.inference_mode():
        for waves, records in batches:
            z = module(torch.from_numpy(np.stack(waves).astype(np.float32)).to(device))
            heads = {h: torch.log_softmax(v.double(), dim=-1).cpu().numpy() for h, v in z.items()}
            commands = backbone.decode({h: torch.from_numpy(v) for h, v in heads.items()})
            scores = number_prior.joint(heads[tens], heads[ones], backbone.slot_classes[tens],
                                        backbone.slot_classes[ones])                            # (B, 60)
            order = np.argsort(-scores, axis=1)

            for k, record in enumerate(records):
                t = backbone.slot_classes[tens][heads[tens][k].argmax()]
                o = backbone.slot_classes[ones][heads[ones][k].argmax()]
                truth = {"intent": record["intent"]}
                truth.update({n: record["slots"].get(n) for n in spec_slots(module, record["intent"])})
                command = commands[k]

                out.append({
                    "file": record["file"],
                    "split": record["split"],
                    "number": record["slots"]["number"],
                    "read": None if "N/A" in (t, o) else t * 10 + o,
                    "top2": number_prior.NUMBERS[order[k, :2]].tolist(),
                    "p_top2": np.exp(scores[k, order[k, :2]]).tolist(),
                    "unit": backbone.slot_classes["unit"][heads["unit"][k].argmax()] == record["slots"]["unit"],
                    "exact": all(command.get(n) == v for n, v in truth.items()),
                    "rest": command["intent"] == "timer.set" and command.get("unit") == record["slots"]["unit"],
                    "scores": scores[k],
                })

    return out


def training_counts() -> tuple[Counter, Counter]:
    """Timer commands per number in the training manifest: real speech, and synthetic."""
    real, synthetic = Counter(), Counter()
    with open(REPO / "data/dataset/manifest_train.jsonl") as handle:
        for line in handle:
            row = json.loads(line)
            if row["intent"] == "timer.set":
                (real if row.get("model") in number_prior.REAL_SOURCES else synthetic)[row["slots"]["number"]] += 1

    return real, synthetic


def report(name: str, rows: list[dict], clips: list[dict], real: Counter, synthetic: Counter) -> None:
    """Print the summary, the teen/-ty pairs, the per-number table and Ken's clips."""
    wrong = [r for r in rows if r["read"] != r["number"]]
    kinds = Counter(kind(r["number"], r["read"]) for r in wrong)

    print(f"=== {name}   (n={len(rows):,}, no threshold)")
    print(f"  number read right {1 - len(wrong) / len(rows):.3f}   "
          f"right or runner-up {np.mean([r['number'] in r['top2'] for r in rows]):.3f}   "
          f"unit right {np.mean([r['unit'] for r in rows]):.3f}   "
          f"whole command right {np.mean([r['exact'] for r in rows]):.3f}")
    print(f"  {len(wrong)} misreadings: " + ", ".join(f"{k} {v} ({v / len(wrong):.0%})" for k, v in kinds.most_common()))
    print(f"  the right number is the runner-up in {np.mean([r['number'] in r['top2'] for r in wrong]):.0%} of them\n")

    print("  teen / -ty pairs, said -> read as its twin:")
    for teen, ty in TEEN_TY.items():
        for said, twin in ((teen, ty), (ty, teen)):
            these = [r for r in rows if r["number"] == said]
            if these:
                print(f"    {said:2d} -> {twin:2d}   {sum(r['read'] == twin for r in these)} of {len(these)}")

    print(f"\n  {'number':>6s} {'n':>5s} {'right':>6s} {'top 2':>6s} {'real train':>11s} {'synthetic':>10s}   most common misreading")
    for number in sorted({r["number"] for r in rows}):
        these = [r for r in rows if r["number"] == number]
        misread = Counter(r["read"] for r in these if r["read"] != number).most_common(1)
        print(f"  {number:6d} {len(these):5d} {np.mean([r['read'] == number for r in these]):6.2f} "
              f"{np.mean([number in r['top2'] for r in these]):6.2f} {real[number]:11d} {synthetic[number]:10d}   "
              f"{f'{misread[0][0]} ({misread[0][1]})' if misread else ''}")

    print("\n  Ken's recordings:")
    for c in clips:
        print(f"    {c['file']:22s} said {c['number']:2d}, read {c['read']}   "
              f"top two {c['top2'][0]} (p {c['p_top2'][0]:.2f}), {c['top2'][1]} (p {c['p_top2'][1]:.2f})")
    print()


def tune(results: dict, prior: np.ndarray) -> float:
    """
    Choose the prior's weight on the validation split, averaged over the runs.

    Returns:
        The smallest weight with the best mean validation number accuracy
    """
    curves = {}
    for name, result in results.items():
        val = [r for r in result["rows"] if r["split"] == "val"]
        scores, truth = np.stack([r["scores"] for r in val]), np.array([r["number"] for r in val])
        curves[name] = np.array([(number_prior.read(scores, prior, w) == truth).mean() for w in WEIGHTS])
    mean = np.mean(list(curves.values()), axis=0)
    best = float(WEIGHTS[int(np.argmax(mean))])

    print("=== number prior: validation number accuracy by weight")
    print(f"  {'weight':>6s}" + "".join(f"{name[:34]:>36s}" for name in curves) + f"{'mean':>8s}")
    for i, w in enumerate(WEIGHTS):
        mark = "  <- chosen" if w == best else ""
        print(f"  {w:6.1f}" + "".join(f"{c[i]:36.3f}" for c in curves.values()) + f"{mean[i]:8.3f}{mark}")
    print()

    return best


def test_report(name: str, result: dict, prior: np.ndarray, weight: float) -> None:
    """Report the chosen prior on the test split and on Ken's recordings."""
    test = [r for r in result["rows"] if r["split"] == "test_human"]
    scores, truth = np.stack([r["scores"] for r in test]), np.array([r["number"] for r in test])
    heads = np.array([-1 if r["read"] is None else r["read"] for r in test])
    joint = number_prior.read(scores, prior, 0.0)
    primed = number_prior.read(scores, prior, weight)
    rest = np.array([r["rest"] for r in test])

    print(f"=== {name}: test, n={len(test)}")
    for label, read in (("heads read separately (now)", heads), ("both heads together", joint),
                        (f"together + prior, weight {weight:.1f}", primed)):
        print(f"  {label:34s} number right {(read == truth).mean():.3f}   "
              f"whole command right {(rest & (read == truth)).mean():.3f}")

    changed = Counter()
    for number in sorted(set(truth)):
        mask = truth == number
        before, after = int((heads[mask] == number).sum()), int((primed[mask] == number).sum())
        if before != after:
            changed[number] = after - before
    print("  per number, test commands gained (+) or lost (-) with the prior:",
          ", ".join(f"{n}: {d:+d} of {int((truth == n).sum())}" for n, d in sorted(changed.items())))

    for c in result["clips"]:
        after = int(number_prior.read(c["scores"][None], prior, weight)[0])
        print(f"  Ken's {c['file']:22s} said {c['number']:2d}: read {c['read']} now, {after} with the prior")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+",
                    help="run directories; join with '+' for an ensemble, end with '@<bias>' to bias `none`")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--prior", action="store_true", help="tune the number prior on val and report it on test")
    ap.add_argument("--write-prior", action="store_true", help=f"save the tuned prior to {number_prior.PRIOR_FILE}")
    ap.add_argument("--out", default="logs/numbers.json")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    data, spec = str(REPO / "data/dataset"), str(REPO / "commands.yaml")
    sets = []
    for split in ("test_human", "val"):
        ds = VoiceCommandDataset(data_dir=data, spec_path=spec, split=split)
        keep = [i for i, r in enumerate(ds.records) if r["intent"] == "timer.set"
                and r.get("model") in number_prior.REAL_SOURCES and r["slots"].get("number")]
        for i in keep:
            ds.records[i]["split"] = split
        sets.append((ds, keep))
    real, synthetic = training_counts()
    print(f"real timer commands: {sum(len(k) for _, k in sets):,} "
          f"(test_human {len(sets[0][1]):,}, val {len(sets[1][1]):,})\n")

    results = {}
    for run in args.runs:
        module, name = load_spec(run, args.device)
        rows = []
        for ds, keep in sets:
            rows += read_batches(module, dataset_batches(ds, keep), args.device)
        clips = read_batches(module, clip_batches(), args.device)
        report(name, rows, clips, real, synthetic)
        results[name] = {"rows": rows, "clips": clips}

        del module
        torch.cuda.empty_cache()

    if args.prior or args.write_prior:
        counts = number_prior.real_counts()
        prior = number_prior.log_prior(counts, SMOOTHING)
        weight = tune(results, prior)
        for name, result in results.items():
            test_report(name, result, prior, weight)

        if args.write_prior:
            number_prior.PRIOR_FILE.write_text(json.dumps({
                "about": "Real timer commands per number in the training set (STOP, Timers and Such), "
                         "add-one smoothed. The demo scores every number 1-60 from both digit heads "
                         "together and adds weight x log-prior. Weight tuned on the validation split "
                         "by evaluate_numbers.py --prior.",
                "smoothing": SMOOTHING,
                "weight": weight,
                "counts": {str(n): c for n, c in counts.items()},
            }, indent=1))
            print(f"wrote {number_prior.PRIOR_FILE}")

    strip = lambda r: {k: v for k, v in r.items() if k != "scores"}
    json.dump({n: {"rows": [strip(r) for r in res["rows"]], "clips": [strip(c) for c in res["clips"]]}
               for n, res in results.items()}, open(REPO / args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
