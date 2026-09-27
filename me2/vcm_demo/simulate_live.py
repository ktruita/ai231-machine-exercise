"""Replay "Marvin" + a pause + a command through the live loop, before and after.

The live loop cannot be tested with a microphone at scale, but everything it
decides happens in CommandListener, one 100 ms block at a time - so this
builds the audio a microphone would deliver and feeds it the same way:

    room noise | "marvin" | pause | real command | room noise

and checks whether the listener hands the recogniser the command. `right`
means the live decode matches what the recogniser makes of the clean command
on its own, so the model's own mistakes do not count against the loop.

    different  a decode came back, but not the one the clean command gives
    early      a result came back before the command started - the capture
               ended on the tail of "marvin"
    timeout    the listener gave up waiting for a command
    missed     no decode at all

`before` is the loop as it was: no tail rule and no timeout. `after` is
CommandListener's defaults. Streams where the wake word never fired are left
out of both.

Three more changes were tried here and dropped, over the same 1,656 streams
(right, of streams where the wake word fired, against 74.4% for the two rules
kept): capturing from 0.3 s before the command's onset instead of from the
wake word, 72.4%, clipping quiet openings at a far mic; setting the speech
level to 3x the room's background at start-up, 72.4%, costing far-mic speech
in a noisy room 9 points; the same at 2x, 74.1%, no gain.

Torch-free like the demo, but it reads the dataset, so it runs from the repo,
not the Pi.

Usage:
    python simulate_live.py
    python simulate_live.py --model deploy/vcm_mined_fsc --none-bias 0
"""
import argparse
import json
import random
import wave
from collections import Counter
from pathlib import Path

import numpy as np

from demo import CommandListener
from vcm.runtime import CommandRecogniser, WakeWordDetector

REPO = Path(__file__).resolve().parent
ME2 = REPO.parent
SR = 16000
HOP = 1600
PAUSES_S = (0.0, 0.3, 0.6, 1.0, 1.5, 2.5)
LEAD_S = 1.5                                   # room before "marvin"
TAIL_S = 3.0
LEVELS = {"close": 0.08, "far": 0.03}          # loudest 100 ms block of speech, RMS
ROOMS = {"quiet": 0.002, "noisy": 0.008}       # background RMS
KEN_CLIPS = ME2 / "my_audio_samples/wav"
CONFIGS = {
    "before": dict(skip_s=0, wait_s=None),
    "after": dict(),
}


def load(path: Path) -> np.ndarray:
    """Read a 16-bit mono wav as float32 in [-1, 1]."""
    with wave.open(str(path)) as handle:
        audio = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")

    return audio.astype(np.float32) / 32768


def trim(audio: np.ndarray, threshold: float = 0.02, margin: int = 800) -> np.ndarray:
    """Cut leading and trailing silence, keeping 50 ms either side."""
    loud = np.where(np.abs(audio) > threshold)[0]

    return audio if loud.size == 0 else audio[max(loud[0] - margin, 0): loud[-1] + margin]


def to_level(audio: np.ndarray, peak: float) -> np.ndarray:
    """Scale so the loudest 100 ms block has RMS `peak`."""
    blocks = [audio[i:i + HOP] for i in range(0, max(len(audio) - HOP, 0) + 1, HOP // 4)]
    top = max(float(np.sqrt((b ** 2).mean())) for b in blocks)

    return (audio * (peak / max(top, 1e-6))).astype(np.float32)


def held_out_marvins() -> list[np.ndarray]:
    """Real "marvin" recordings from Speech Commands speakers the models never trained on."""
    seen = set()
    for line in open(ME2 / "data/dataset/manifest_train.jsonl"):
        record = json.loads(line)
        if record.get("model") in {"speech_commands", "speech_commands_slot"}:
            seen.add(str(record.get("speaker_id")))
    paths = sorted((ME2 / "data/real_speech/speech_commands/marvin").glob("*.wav"))

    return [trim(load(p)) for p in paths if p.name.split("_")[0] not in seen]


def commands(per_intent: int, rng: random.Random) -> list[tuple[str, np.ndarray]]:
    """Held-out real commands, the same number per intent, plus Ken's recordings."""
    by_intent = {}
    for line in open(ME2 / "data/dataset/manifest_test_human.jsonl"):
        record = json.loads(line)
        if record["intent"] != "none":
            by_intent.setdefault(record["intent"], []).append(record)

    picked = []
    for intent, records in sorted(by_intent.items()):
        for record in rng.sample(records, per_intent):
            picked.append(("test", trim(load(ME2 / "data/dataset" / record["file"]))))
    for path in sorted(KEN_CLIPS.glob("*.wav")):
        if path.stem != "Unrelated":
            picked.append(("ken", trim(load(path))))

    return picked


def room_noise(kind: str, length: int, rms: float, noises: list[Path], rng: random.Random) -> np.ndarray:
    """Background: faint white noise for a quiet room, a held-out real recording for a noisy one."""
    if kind == "quiet":
        noise = np.random.default_rng(rng.randrange(1 << 30)).standard_normal(length)
    else:
        clip = load(rng.choice(noises))
        noise = np.tile(clip, int(np.ceil(length / max(len(clip), 1))))[:length]

    return (noise * (rms / max(float(np.sqrt((noise ** 2).mean())), 1e-9))).astype(np.float32)


def replay(listener: CommandListener, stream: np.ndarray, command_at: int) -> tuple[str, dict | None]:
    """
    Feed a stream block by block and name the outcome.

    Returns:
        (outcome, the first result event or None)
    """
    woke = False
    for i in range(0, len(stream) - HOP + 1, HOP):
        for event in listener.feed(stream[i:i + HOP]):
            if event["type"] == "wake":
                woke = True
            elif event["type"] == "timeout":
                return "timeout", None
            elif event["type"] == "result":
                return ("early" if listener.clock <= command_at + HOP else "decoded"), event

    return ("missed" if woke else "no wake"), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=["deploy/vcm_mined_fsc", "deploy/vcm_mined_fsc_s2"])
    ap.add_argument("--none-bias", type=float, default=-0.2)
    ap.add_argument("--per-intent", type=int, default=12)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    recogniser = CommandRecogniser([REPO / m for m in args.model], none_bias=args.none_bias, number_prior=None)
    detector = WakeWordDetector(REPO / "deploy/wakeword_marvin")
    marvins = held_out_marvins()
    clips = commands(args.per_intent, rng)
    noises = sorted((ME2 / "data/augment/RIRS_NOISES/pointsource_noises").glob("*.wav"))
    print(f"{len(clips)} commands ({sum(s == 'ken' for s, _ in clips)} of them Ken's), "
          f"{len(marvins)} held-out 'marvin' clips, pauses {PAUSES_S} s\n")

    tally = {}                                  # (config, level, room, pause) or (config, source) -> outcomes
    for source, clip in clips:
        for level, peak in LEVELS.items():
            command = to_level(clip, peak)
            want = recogniser(recogniser.fit_length(command))["command"]
            for room, noise_rms in ROOMS.items():
                for pause in PAUSES_S:
                    marvin = to_level(rng.choice(marvins), peak)
                    start = int(LEAD_S * SR) + rng.randrange(HOP)
                    command_at = start + len(marvin) + int(pause * SR)
                    speech = np.zeros(command_at + len(command) + int(TAIL_S * SR), np.float32)
                    speech[start:start + len(marvin)] = marvin
                    speech[command_at:command_at + len(command)] = command
                    stream = speech + room_noise(room, len(speech), noise_rms, noises, rng)

                    for name, config in CONFIGS.items():
                        outcome, event = replay(CommandListener(recogniser, detector, **config), stream, command_at)
                        if outcome == "decoded":
                            outcome = "right" if event["command"] == want else "different"
                        tally.setdefault((name, level, room, pause), Counter())[outcome] += 1
                        tally.setdefault((name, source), Counter())[outcome] += 1

    def outcomes(name: str, match=lambda key: True) -> Counter:
        """Outcomes of one config, summed over the conditions `match` accepts."""
        return sum((c for k, c in tally.items() if k[0] == name and len(k) == 4 and match(k)), Counter())

    def right(counter: Counter) -> str:
        """Decoded right, as a share of the streams where the wake word fired."""
        heard = sum(v for k, v in counter.items() if k != "no wake")
        return f"{counter['right'] / heard:.0%}" if heard else "-"

    columns = [(level, room) for level in LEVELS for room in ROOMS]
    print("decoded right, of streams where the wake word fired")
    print(f"{'':8s}" + "".join(f"{l + '/' + r:>13s}" for l, r in columns)
          + "".join(f"{p:>6.1f}s" for p in PAUSES_S) + f"{'all':>6s}{'test':>6s}{'Ken':>6s}")
    for name in CONFIGS:
        by_room = [right(outcomes(name, lambda k, l=l, r=r: k[1:3] == (l, r))) for l, r in columns]
        by_pause = [right(outcomes(name, lambda k, p=p: k[3] == p)) for p in PAUSES_S]
        by_source = [right(tally.get((name, s), Counter())) for s in ("test", "ken")]
        print(f"{name:8s}" + "".join(f"{c:>13s}" for c in by_room) + "".join(f"{c:>7s}" for c in by_pause)
              + f"{right(outcomes(name)):>6s}" + "".join(f"{c:>6s}" for c in by_source))

    print("\noutcomes, all conditions")
    for name in CONFIGS:
        total = outcomes(name)
        heard = sum(v for k, v in total.items() if k != "no wake")
        print(f"  {name:8s}" + "  ".join(f"{k} {total[k] / heard:5.1%}" for k in
                                          ("right", "different", "early", "timeout", "missed")))

    print("\nbefore / after, by pause and condition")
    print(f"{'pause':>7s}" + "".join(f"{l + ' mic, ' + r + ' room':>26s}" for l, r in columns))
    for pause in PAUSES_S:
        cells = [f"{right(tally[('before', l, r, pause)]):>4s} / {right(tally[('after', l, r, pause)]):>4s}"
                 for l, r in columns]
        print(f"{pause:6.1f}s" + "".join(f"{c:>26s}" for c in cells))


if __name__ == "__main__":
    main()
