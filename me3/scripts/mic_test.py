"""Mic check for the live agent: opens every input device exactly like the agent does
(PyAudio, 16 kHz, mono, 16-bit), records a few seconds while you talk, and prints the level.
It then tells you which number to put in .env as ME3_INPUT_DEVICE.

Run on the Pi (agent stopped, so the mic is free):
    .venv/bin/python scripts/mic_test.py            # test all input devices
    .venv/bin/python scripts/mic_test.py --device 1 # test one device
"""

from __future__ import annotations

import argparse
import array
import math
import os
import sys

RATE, CHUNK = 16000, 320  # what the agent uses: 16 kHz mono, 20 ms chunks


def dbfs(samples: array.array) -> float:
    if not samples:
        return -120.0
    return 20 * math.log10(max(1.0, math.sqrt(sum(x * x for x in samples) / len(samples))) / 32768)


def record_device(p, index: int, seconds: float) -> float | None:
    import pyaudio

    name = p.get_device_info_by_index(index)["name"]
    try:
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True,
                        input_device_index=index, frames_per_buffer=CHUNK)
    except Exception as e:  # noqa: BLE001
        print(f"  [{index}] {name}: CANNOT OPEN at 16 kHz mono ({e})")
        return None
    print(f"  [{index}] {name}: recording {seconds:.0f} s, talk normally now ...", flush=True)
    frames = array.array("h")
    loudest = -120.0
    for _ in range(int(RATE / CHUNK * seconds)):
        chunk = array.array("h", stream.read(CHUNK, exception_on_overflow=False))
        frames.extend(chunk)
        loudest = max(loudest, dbfs(chunk))
    stream.stop_stream()
    stream.close()
    avg = dbfs(frames)
    verdict = ("GOOD" if loudest > -30 else "QUIET: raise mic gain (alsamixer, F4)" if loudest > -50
               else "SILENT: wrong device, muted, or not plugged in")
    print(f"       average {avg:.1f} dBFS, loudest moment {loudest:.1f} dBFS -> {verdict}")
    return loudest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, help="only test this device index")
    ap.add_argument("--seconds", type=float, default=4)
    a = ap.parse_args()

    # silence ALSA's noisy probing messages
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    os.dup2(devnull, 2)
    try:
        import pyaudio
        p = pyaudio.PyAudio()
    finally:
        os.dup2(saved, 2)  # always restore, so real errors are visible

    inputs = [i for i in range(p.get_device_count())
              if p.get_device_info_by_index(i)["maxInputChannels"] > 0]
    current = os.environ.get("ME3_INPUT_DEVICE")
    try:
        from dotenv import dotenv_values
        current = current or dotenv_values(".env").get("ME3_INPUT_DEVICE")
    except Exception:  # noqa: BLE001
        pass
    print(f"Input devices: {inputs}   (.env ME3_INPUT_DEVICE={current or 'not set = system default'})")
    for i in inputs:
        print(f"  [{i}] {p.get_device_info_by_index(i)['name']}")
    print()

    results = {}
    for i in ([a.device] if a.device is not None else inputs):
        results[i] = record_device(p, i, a.seconds)
        print()
    p.terminate()

    good = [i for i, lvl in results.items() if lvl is not None and lvl > -30]
    if good:
        print(f"Use: ME3_INPUT_DEVICE={good[0]}   (put this in ~/me3/.env, then restart the agent)")
    else:
        print("No device opened at 16 kHz mono with a good level. If every device says CANNOT OPEN,\n"
              "create ~/.asoundrc with a 'plug' default (see the chat) and run this again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
