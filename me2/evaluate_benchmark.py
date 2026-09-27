"""Measure the benchmark's headline: capability, restraint and robustness.

`benchmark.yaml` defines deployment readiness as three numbers, and until now
none was reported. This computes the first two, and the robustness curve:

    restraint    false accepts per hour on test_negatives (3.3 h of real
                 non-command speech), swept against the confidence threshold
    capability   exact command accuracy on test_human (3,327 real commands,
                 all five intents) at the operating point the restraint fixes
    robustness   capability under noise and room reverberation the models
                 never trained on (--robustness)

The operating point. The spec says "highest confidence threshold whose
false_accepts_per_hour <= target", but false accepts fall as the threshold
rises, so the highest qualifying threshold is always ~1.0 and rejects
everything. The meaningful reading is the lowest threshold that meets the
target - the most permissive setting that still honours restraint - and that
is what is computed here.

Confidence is the intent head's max softmax, as the spec asks. These models
output p = 1.000 on many utterances, where floating-point confidences tie, so
utterances are ranked by the residual mass S = sum over the losing intents of
exp(z_i - z_max). p_max = 1 / (1 + S): the same order, without the ties.

Ensembles. Runs joined by '+' are decoded as one model, the way the demo
runtime does it: probabilities averaged per head, and an optional '@<bias>'
added to the `none` intent. The degradations are seeded, so every run - single
model or ensemble - hears the same noisy and reverberant audio.

Usage:
    python evaluate_benchmark.py modelstore/vcm_mined_fsc modelstore/vcm_mined_fsc_s2
    python evaluate_benchmark.py modelstore/vcm_mined_fsc --robustness
    python evaluate_benchmark.py "modelstore/vcm_mined_fsc+modelstore/vcm_mined_fsc_s2@-0.2" --robustness --number-prior
"""
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

import number_prior
from dataloaders.augment import AudioAugment
from dataloaders.vcm_dataloader import VoiceCommandDataset
from evaluate import load_run, spec_slots

REPO = Path(__file__).resolve().parent
TARGET_FA_PER_HOUR = 1.0
DEMO_THRESHOLD = 0.6
WAKE_FA_PER_HOUR = 0.5          # 3-of-5 windows over 0.99, from vcm_demo/demo.py
SNR_DB = (20, 15, 10, 5, 0)
HELD_OUT_NOISE = "data/augment/RIRS_NOISES/pointsource_noises"


class Ensemble(nn.Module):
    """
    Several trained modules decoded as one, as vcm_demo/vcm/runtime.py does.

    Members' probabilities are averaged per head - not their logits - and a
    constant is added to the averaged `none` intent, so an ensemble is scored
    exactly as the demo runs it.

    Flow: (B, num_samples) -> {head: (B, C)} log-probabilities
    """

    def __init__(
        self,
        members: list,
        none_bias: float = 0.0,
    ):
        """
        Args:
            members: Trained modules with the same heads and label order
            none_bias: Added to the `none` intent log-probability (default: 0.0)
        """
        super().__init__()

        # Validation
        reference = (members[0].backbone.intent_names, members[0].backbone.slot_classes)
        for member in members[1:]:
            if (member.backbone.intent_names, member.backbone.slot_classes) != reference:
                raise ValueError(f"ensemble members must share labels, got intents {member.backbone.intent_names}")

        self.members = nn.ModuleList(members)
        self.backbone = members[0].backbone
        self.none_index = self.backbone.intent_names.index("none")
        self.none_bias = none_bias

    def forward(self, waveform: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = [member(waveform) for member in self.members]

        heads = {}
        for head in outputs[0]:
            logp = torch.stack([torch.log_softmax(o[head].double(), dim=-1) for o in outputs])  # (M, B, C)
            heads[head] = torch.logsumexp(logp, dim=0) - math.log(len(outputs))               # (B, C)
        heads["intent"][:, self.none_index] += self.none_bias
        heads["intent"] = torch.log_softmax(heads["intent"], dim=-1)

        return heads


def load_spec(spec: str, device: str) -> tuple:
    """
    Load a run, or several runs decoded as one ensemble.

    Args:
        spec: A run directory, or several joined by '+', optionally followed by
            '@<bias>' added to the `none` intent - for example
            modelstore/vcm_mined_fsc+modelstore/vcm_mined_fsc_s2@-0.2
        device: Device to run on

    Returns:
        Tuple of (module in eval mode, name used in the results)
    """
    runs, _, bias = spec.partition("@")
    dirs = runs.split("+")
    members = [load_run(d, None, device)[0] for d in dirs]
    name = "+".join(Path(d).name for d in dirs) + (f"@{bias}" if bias else "")

    if len(members) == 1 and not bias:
        return members[0], name

    return Ensemble(members, float(bias or 0.0)).to(device).eval(), name


def score_set(module, dataset, device: str, transform=None, batch: int = 256, prior: tuple | None = None) -> dict:
    """
    Decode a dataset and keep what the threshold sweep needs.

    Args:
        module: Trained module
        dataset: VoiceCommandDataset over test_human or test_negatives
        device: Device to run on
        transform: Optional function applied to each waveform, for degradation
        batch: Items per forward pass (default: 256)
        prior: (log-prior, weight) to read timer numbers with, as the demo
            does with number_prior.json; None reads each digit head alone
            (default: None)

    Returns:
        Arrays of truth intent, predicted intent, residual mass S, exact-match
        flag, source corpus and duration, one entry per utterance
    """
    records = dataset.records
    out = {k: [] for k in ("truth", "pred", "residual", "exact", "source", "seconds")}

    with torch.inference_mode():
        for start in range(0, len(dataset), batch):
            idx = range(start, min(start + batch, len(dataset)))
            waves = [dataset[i]["waveform"].numpy() for i in idx]
            if transform is not None:
                waves = [transform(w) for w in waves]
            logits = module(torch.from_numpy(np.stack(waves).astype(np.float32)).to(device))
            commands = module.backbone.decode(logits)
            if prior is not None:
                tens, ones = module.backbone.digit_slots["number"]
                scores = number_prior.joint(
                    *(torch.log_softmax(logits[h].double(), dim=-1).cpu().numpy() for h in (tens, ones)),
                    module.backbone.slot_classes[tens], module.backbone.slot_classes[ones])
                for command, number in zip(commands, number_prior.read(scores, *prior)):
                    if "number" in command:
                        command["number"] = int(number)

            z = logits["intent"].double().cpu()
            top = z.max(dim=-1, keepdim=True).values
            residual = (torch.exp(z - top).sum(dim=-1) - 1.0).tolist()   # (B,) mass off the winner

            for k, i in enumerate(idx):
                record = records[i]
                truth = {"intent": record["intent"]}
                truth.update({n: record["slots"].get(n) for n in spec_slots(module, record["intent"])})
                pred = commands[k]
                exact = pred["intent"] == truth["intent"] and all(
                    pred.get(n) == v for n, v in truth.items() if n != "intent")

                out["truth"].append(record["intent"])
                out["pred"].append(pred["intent"])
                out["residual"].append(max(residual[k], 0.0))
                out["exact"].append(exact)
                out["source"].append(record.get("model", "?"))
                out["seconds"].append(record.get("duration_s") or 0.0)

    return {k: np.array(v) for k, v in out.items()}


def at_threshold(human: dict, negatives: dict, max_residual: float) -> dict:
    """
    Score one threshold, expressed as the largest residual mass accepted.

    An utterance is acted on only if its intent is not `none` and its residual
    mass is at most max_residual; otherwise the device says "didn't catch that".

    Returns:
        Capability (micro and macro), false-reject rate, false accepts and
        their hourly rate, coverage and selective accuracy
    """
    acts_h = (human["pred"] != "none") & (human["residual"] <= max_residual)
    acts_n = (negatives["pred"] != "none") & (negatives["residual"] <= max_residual)
    hours = negatives["seconds"].sum() / 3600

    correct = human["exact"] & acts_h
    per_intent = {i: float(correct[human["truth"] == i].mean()) for i in sorted(set(human["truth"]))}

    return {
        "threshold": 1.0 / (1.0 + max_residual) if math.isfinite(max_residual) else 0.0,
        "capability": float(correct.mean()),
        "capability_macro": float(np.mean(list(per_intent.values()))),
        "per_intent": per_intent,
        "false_reject_rate": float((~acts_h).mean()),
        "false_accepts": int(acts_n.sum()),
        "fa_per_hour": float(acts_n.sum() / hours),
        "fa_rate": float(acts_n.mean()),
        "coverage": float(acts_h.mean()),
        "selective_accuracy": float(correct.sum() / max(acts_h.sum(), 1)),
    }


def operating_point(negatives: dict, target: float) -> float:
    """
    The most permissive threshold whose false accepts per hour meet the target.

    Returns:
        The largest accepted residual mass, or -1.0 when even the most
        confident false accepts cannot be separated from each other
    """
    hours = negatives["seconds"].sum() / 3600
    allowed = int(math.floor(target * hours))
    accepted = np.sort(negatives["residual"][negatives["pred"] != "none"])

    if len(accepted) <= allowed:
        return math.inf
    # Accept strictly more confident than the (allowed + 1)-th most confident
    # false accept, so at most `allowed` negatives pass
    return float(np.nextafter(accepted[allowed], -1.0))


def noise_transform(noise_files: list, snr_db: float, seed: int):
    """Mix a random held-out noise file into each waveform at a fixed SNR."""
    import soundfile as sf

    rng = random.Random(seed)

    def apply(waveform: np.ndarray) -> np.ndarray:
        noise, _ = sf.read(str(rng.choice(noise_files)), dtype="float32", always_2d=False)
        if noise.ndim > 1:
            noise = noise.mean(axis=1)
        reps = int(np.ceil(len(waveform) / max(len(noise), 1)))
        noise = np.tile(noise, reps)[:len(waveform)]
        speech = waveform[np.abs(waveform) > 1e-4]
        p_speech = float((speech ** 2).mean()) if speech.size else 1e-8
        p_noise = float((noise ** 2).mean()) + 1e-12
        noise = noise * math.sqrt(p_speech / (p_noise * 10 ** (snr_db / 10)))

        return (waveform + noise).astype(np.float32)

    return apply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+",
                    help="run directories; join with '+' for an ensemble, end with '@<bias>' to bias `none`")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--robustness", action="store_true")
    ap.add_argument("--number-prior", action="store_true",
                    help="read timer numbers with vcm_demo/vcm/number_prior.json, as the demo does")
    ap.add_argument("--out", default="logs/benchmark_core.json")
    args = ap.parse_args()
    prior = number_prior.load() if args.number_prior else None

    data, spec = str(REPO / "data/dataset"), str(REPO / "commands.yaml")
    human_set = VoiceCommandDataset(data_dir=data, spec_path=spec, split="test_human")
    negative_set = VoiceCommandDataset(data_dir=data, spec_path=spec, split="test_negatives")
    hours = sum(r.get("duration_s") or 0 for r in negative_set.records) / 3600
    print(f"test_human {len(human_set):,} commands   test_negatives {len(negative_set):,} "
          f"utterances, {hours:.2f} h   target {TARGET_FA_PER_HOUR} FA/h "
          f"(at most {int(hours * TARGET_FA_PER_HOUR)} false accepts)\n")

    results = {}
    for run in args.runs:
        module, name = load_spec(run, args.device)
        human = score_set(module, human_set, args.device, prior=prior)
        negatives = score_set(module, negative_set, args.device, prior=prior)

        argmax = at_threshold(human, negatives, math.inf)
        demo = at_threshold(human, negatives, 1 / DEMO_THRESHOLD - 1)
        op_residual = operating_point(negatives, TARGET_FA_PER_HOUR)
        op = at_threshold(human, negatives, op_residual) if op_residual >= 0 else None

        # Cross-check against the rejection measured earlier on the same three
        # sources: SLURP, STOP and Speech Commands negatives, 2,017 utterances
        subset = np.isin(negatives["source"], ["slurp", "stop", "speech_commands"])
        rejection = float((negatives["pred"][subset] == "none").mean())

        sweep = []
        for p in np.concatenate([np.linspace(0, 0.99, 100), 1 - np.logspace(-2, -8, 61)]):
            sweep.append(at_threshold(human, negatives, math.inf if p <= 0 else 1 / p - 1))

        fa_by_source = {}
        for source in sorted(set(negatives["source"])):
            mask = negatives["source"] == source
            fa_by_source[source] = int(((negatives["pred"] != "none") & mask).sum())

        print(f"=== {name}")
        print(f"  rejection on SLURP+STOP+Speech Commands negatives: {rejection:.3f} (n={int(subset.sum()):,})")
        for label, r in (("argmax (no threshold)", argmax), (f"demo threshold {DEMO_THRESHOLD}", demo),
                         ("operating point", op)):
            if r is None:
                print(f"  {label:24s} unreachable - most confident false accepts are tied")
                continue
            print(f"  {label:24s} p>={r['threshold']:.6f}  capability {r['capability']:.3f} "
                  f"(macro {r['capability_macro']:.3f})  FRR {r['false_reject_rate']:.3f}  "
                  f"FA/h {r['fa_per_hour']:7.2f}  system FA/h {WAKE_FA_PER_HOUR * r['fa_rate']:.3f}  "
                  f"coverage {r['coverage']:.3f}  selective acc {r['selective_accuracy']:.3f}")
        if op is not None:
            print("  capability by intent at the operating point:",
                  {k: round(v, 3) for k, v in op["per_intent"].items()})
        print(f"  false accepts at argmax by source: {fa_by_source}\n")

        results[name] = {"argmax": argmax, "demo": demo, "operating_point": op,
                         "rejection_check": rejection, "fa_by_source": fa_by_source, "sweep": sweep}

        if args.robustness:
            conditions = {"clean": None}
            noise = sorted((REPO / HELD_OUT_NOISE).rglob("*.wav"))
            for snr in SNR_DB:
                conditions[f"noise {snr} dB"] = noise_transform(noise, snr, seed=snr)
            far = AudioAugment(musan_dir=str(REPO / "data/augment/musan"),
                               rir_dir=str(REPO / "data/augment/RIRS_NOISES"), split="test", seed=7)
            conditions["far-field (held-out RIRs)"] = lambda w: far.apply_reverb(w.copy()).astype(np.float32)

            curve = {}
            for label, transform in conditions.items():
                degraded = score_set(module, human_set, args.device, transform, prior=prior)
                r = at_threshold(degraded, negatives, math.inf)
                curve[label] = {"capability": r["capability"], "capability_macro": r["capability_macro"]}
                print(f"  robustness  {label:26s} capability {r['capability']:.3f}  macro {r['capability_macro']:.3f}")

            native = {}
            for flag in ("Yes", "No"):
                idx = [i for i, rec in enumerate(human_set.records) if rec.get("native") == flag]
                if idx:
                    native[flag] = float(human["exact"][idx].mean())
            print(f"  native vs L2 (STOP native column, argmax): {native}\n")
            results[name]["robustness"] = curve
            results[name]["native"] = native

        del module
        torch.cuda.empty_cache()

    json.dump(results, open(REPO / args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
