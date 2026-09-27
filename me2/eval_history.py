"""Score every trained model against one frozen test set, in training order.

The test manifest grew over the project - LibriSpeech numbers and STOP
commands were added along the way - so figures quoted at the time are not all
comparable with each other. This rescored every run against the current test
set, so the table reads as progress rather than as a series of moving targets.

Columns:
    real exact    evaluate.py's headline: SLURP, Speech Commands and Timers and
                  Such only, frozen so early runs stay comparable
    real intent   intent accuracy on the same rows
    live intent   held-out real commands with a real "Marvin" before them, the
                  way the live loop captured audio until the trim fix
    stop timer    number accuracy on 695 held-out STOP timer commands
    num slice     number accuracy on 993 mined LibriSpeech number words
    clips         Ken's 10 recordings, exact match

Runs from before the tens/ones number heads (vcm_digits) cannot be read for
the two number columns, and are marked n/a rather than forced.

Usage:
    python eval_history.py
"""
import argparse
import json
import random
import time
import wave
from pathlib import Path

import numpy as np
import torch

from dataloaders.vcm_dataloader import VoiceCommandDataset
from evaluate import REAL_MODELS, load_run, spec_slots, summarise
from evaluate_live import PAUSE_RANGE_S, conditions, held_out_wake_words

REPO = Path(__file__).resolve().parent
LIVE_SOURCES = REAL_MODELS | {"stop"}

CLIPS = {
    "Dim_the_living_room_light_to_70": {"intent": "light.dim", "room": "living room", "percent": 70},
    "Louder": {"intent": "media.control", "action": "volume_up"},
    "Set_timer_for_15_min": {"intent": "timer.set", "unit": "minutes", "number": 15},
    "Set_timer_for_50min": {"intent": "timer.set", "unit": "minutes", "number": 50},
    "Set_timer_to_35min": {"intent": "timer.set", "unit": "minutes", "number": 35},
    "Turn_of_kitchen_lights": {"intent": "light.set", "state": "off", "room": "kitchen"},
    "Turn_of_music": {"intent": "media.control", "action": "stop"},
    "Turn_on_lights": {"intent": "light.set", "state": "on", "room": None},
    "Unrelated": {"intent": "none"},
    "What_time_is_it": {"intent": "query.info", "kind": "time"},
}


def runs_in_order(store: Path) -> list[Path]:
    """Command-model runs sorted by when their final checkpoint was written."""
    runs = []
    for run in store.glob("vcm_*"):
        ckpts = sorted((run / "checkpoints").glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
        if ckpts:
            runs.append((ckpts[-1].stat().st_mtime, run))

    return [run for _, run in sorted(runs)]


def load_clip(path: Path, num_samples: int) -> torch.Tensor:
    """Read one of Ken's clips and pad it to the model window."""
    with wave.open(str(path)) as handle:
        audio = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(np.float32) / 32768
    audio = audio[:num_samples]

    return torch.from_numpy(np.pad(audio, (0, num_samples - len(audio))))


def decode(module, waves: torch.Tensor, device: str, batch: int = 256) -> tuple[list[dict], dict]:
    """
    Decode a stack of waveforms.

    Returns:
        (decoded commands, raw digit-head class indices keyed by head name)
    """
    commands, heads = [], {}
    with torch.inference_mode():
        for start in range(0, len(waves), batch):
            logits = module(waves[start:start + batch].to(device))
            commands += module.backbone.decode(logits)
            for name in ("number_tens", "number_ones"):
                if name in logits:
                    heads.setdefault(name, []).extend(logits[name].argmax(-1).tolist())

    return commands, heads


def number_hits(heads: dict, targets: list[tuple[int, int]]) -> float | None:
    """Share of utterances whose tens and ones heads both match, or None if absent."""
    if "number_tens" not in heads:
        return None
    hits = sum(t == tens and o == ones
               for (t, o), tens, ones in zip(targets, heads["number_tens"], heads["number_ones"]))

    return hits / max(len(targets), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="logs/history.json")
    args = ap.parse_args()

    t0 = time.time()
    dataset = VoiceCommandDataset(data_dir=str(REPO / "data/dataset"),
                                  spec_path=str(REPO / "commands.yaml"), split="test")
    records = dataset.records

    real = [i for i, r in enumerate(records) if r.get("model") in REAL_MODELS]
    live = [i for i, r in enumerate(records) if r.get("model") in LIVE_SOURCES and r["intent"] != "none"]
    timer = [i for i, r in enumerate(records) if r.get("model") == "stop" and r["intent"] == "timer.set"]
    numbers = [i for i, r in enumerate(records) if r.get("model") == "librispeech_num"]

    # Every item is read from NFS exactly once and reused for all runs
    wanted = sorted(set(real) | set(live) | set(timer) | set(numbers))
    items = {i: dataset[i] for i in wanted}
    wave_of = lambda idx: torch.stack([items[i]["waveform"] for i in idx])
    digit = lambda idx: [(int(items[i]["target_number_tens"]), int(items[i]["target_number_ones"])) for i in idx]

    rng = random.Random(11)
    marvins = held_out_wake_words(REPO / "data/dataset", REPO / "data/real_speech/speech_commands/marvin")
    live_waves = torch.stack([torch.from_numpy(
        conditions(items[i]["waveform"].numpy(), rng.choice(marvins), rng.uniform(*PAUSE_RANGE_S))["wake_word"])
        for i in live])

    clips = {stem: load_clip(REPO / "my_audio_samples/wav" / f"{stem}.wav", dataset.num_samples) for stem in CLIPS}
    print(f"cached {len(items):,} test items and {len(live):,} live variants in {time.time() - t0:.0f}s\n")

    header = f"{'trained':12s} {'run':22s} {'step':>6s} {'real ex':>8s} {'real int':>9s} " \
             f"{'live int':>9s} {'stop tmr':>9s} {'num slc':>8s} {'clips':>6s}"
    print(header)
    print("-" * len(header))

    table = []
    for run in runs_in_order(REPO / "modelstore"):
        ckpt = sorted((run / "checkpoints").glob("*.ckpt"), key=lambda p: p.stat().st_mtime)[-1]
        step = int("".join(c for c in ckpt.stem if c.isdigit()))
        try:
            module, _, _ = load_run(run, step, args.device)
        except Exception as error:
            print(f"{'':12s} {run.name:22s} could not load: {type(error).__name__}")
            continue

        commands, _ = decode(module, wave_of(real), args.device)
        rows = []
        for i, command in zip(real, commands):
            truth = {"intent": records[i]["intent"]}
            truth.update({n: records[i]["slots"].get(n) for n in spec_slots(module, records[i]["intent"])})
            rows.append({"truth": truth, "pred": command})
        headline = summarise(rows)

        live_cmds, _ = decode(module, live_waves, args.device)
        live_intent = sum(c["intent"] == records[i]["intent"] for i, c in zip(live, live_cmds)) / len(live)

        _, timer_heads = decode(module, wave_of(timer), args.device)
        _, number_heads = decode(module, wave_of(numbers), args.device)
        stop_timer = number_hits(timer_heads, digit(timer))
        num_slice = number_hits(number_heads, digit(numbers))

        clip_cmds, _ = decode(module, torch.stack(list(clips.values())), args.device)
        clip_ok = sum(all(str(c.get(k)) == str(v) for k, v in want.items())
                      for c, want in zip(clip_cmds, CLIPS.values()))

        when = time.strftime("%m-%d %H:%M", time.localtime(ckpt.stat().st_mtime))
        fmt = lambda v: f"{v:.3f}" if v is not None else "n/a"
        print(f"{when:12s} {run.name:22s} {step:6d} {headline['exact']:8.3f} {headline['intent']:9.3f} "
              f"{live_intent:9.3f} {fmt(stop_timer):>9s} {fmt(num_slice):>8s} {clip_ok:>4d}/10", flush=True)

        table.append({"run": run.name, "trained": when, "step": step,
                      "real_exact": headline["exact"], "real_intent": headline["intent"],
                      "live_intent": live_intent, "stop_timer": stop_timer,
                      "num_slice": num_slice, "clips": clip_ok})
        del module
        torch.cuda.empty_cache()

    json.dump(table, open(REPO / args.out, "w"), indent=1)
    print(f"\nwrote {args.out}   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
