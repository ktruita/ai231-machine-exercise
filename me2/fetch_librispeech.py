"""Pull the LibriSpeech utterances that contain a number word.

OpenSLR serves the tarballs at roughly 200 KB/s from here, which puts the three
training sets at about eighty hours. The same audio on HuggingFace comes down at
38 MB/s, so this reads the parquet shards instead.

Only 2.7% of utterances are usable, so a shard is scanned by its transcript
column first and the audio is read only for the row groups that survive. Shards
are held in memory rather than written down and read back: staging 56 GB on a
shared NFS volume to extract two gigabytes of speech cost more than the download.

Each drained shard leaves a marker, so an interrupted run resumes instead of
refetching, and transcripts are never appended twice.

Output mimics the LibriSpeech directory layout, so mine_librispeech.py neither
knows nor cares that the audio arrived this way.
"""
import argparse
import io
import subprocess
import time
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from mine_librispeech import COMPOUND_NEXT, TOKEN, WORD_VALUES

REPO = Path(__file__).resolve().parent
BASE = "https://huggingface.co/datasets/openslr/librispeech_asr/resolve/main"

# Shard counts come from the dataset's file listing; the three training sets
# together are the 960 hours the LibriSpeech paper describes
SUBSETS = {
    "all/train.clean.100": 14,
    "all/train.clean.360": 48,
    "all/train.other.500": 64,
}


def target_word(text: str) -> str | None:
    """
    Return the one number word in an utterance, or None if it cannot be used.

    Args:
        text: Utterance transcript

    Returns:
        The target word, or None when there is no target, more than one, or the
        target is part of a compound like "twenty five"
    """
    words = TOKEN.findall(text.upper())
    hits = [i for i, word in enumerate(words) if word in WORD_VALUES]

    if len(hits) != 1:
        return None

    index = hits[0]
    if index + 1 < len(words) and words[index + 1] in COMPOUND_NEXT:
        return None

    return words[index]


def drain_shard(shard: io.BytesIO, stage: Path) -> Counter:
    """
    Write out every usable utterance in one parquet shard.

    Args:
        shard: Downloaded parquet bytes
        stage: Root of the LibriSpeech-shaped output tree

    Returns:
        Counts of the words written
    """
    written = Counter()
    handle = pq.ParquetFile(shard)

    for group in range(handle.metadata.num_row_groups):
        text = handle.read_row_group(group, columns=["text"]).column("text").to_pylist()
        keep = [i for i, item in enumerate(text) if target_word(item)]
        if not keep:
            continue

        # Only now is the audio worth decoding, and only for these rows
        table = handle.read_row_group(group, columns=["audio", "text", "speaker_id", "chapter_id", "id"])
        audio = table.column("audio").to_pylist()
        speakers = table.column("speaker_id").to_pylist()
        chapters = table.column("chapter_id").to_pylist()
        ids = table.column("id").to_pylist()

        for i in keep:
            folder = stage / str(speakers[i]) / str(chapters[i])
            folder.mkdir(parents=True, exist_ok=True)

            (folder / f"{ids[i]}.flac").write_bytes(audio[i]["bytes"])
            with open(folder / f"{speakers[i]}-{chapters[i]}.trans.txt", "a") as trans:
                trans.write(f"{ids[i]} {text[i]}\n")

            written[target_word(text[i])] += 1

    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="data/negatives/hf_stage")
    args = ap.parse_args()

    stage = REPO / args.stage
    markers = stage / ".drained"
    markers.mkdir(parents=True, exist_ok=True)

    total = Counter()
    done = 0
    for subset, count in SUBSETS.items():
        for index in range(count):
            done += 1
            marker = markers / f"{subset.replace('/', '_')}_{index:04d}"
            if marker.exists():
                total[marker.name] = 0
                continue

            # HuggingFace drops an HTTP/2 stream every so often, which killed a
            # run 65 shards in. Retry rather than lose the rest of the fetch
            url = f"{BASE}/{subset}/{index:04d}.parquet"
            shard = None
            for attempt in range(4):
                result = subprocess.run(["curl", "-sSL", "--http1.1", url], capture_output=True)
                if result.returncode == 0 and result.stdout[:4] == b"PAR1":
                    shard = result.stdout
                    break
                print(f"    retry {attempt + 1} on {subset}/{index:04d} "
                      f"(curl {result.returncode})", flush=True)
                time.sleep(5 * (attempt + 1))

            if shard is None:
                print(f"    SKIPPED {subset}/{index:04d} after 4 attempts", flush=True)
                continue

            written = drain_shard(io.BytesIO(shard), stage)
            total.update(written)
            marker.touch()

            print(f"  [{done:3d}/126] {subset} {index:04d}  +{sum(written.values()):3d}  "
                  f"running total {sum(v for k, v in total.items() if k in WORD_VALUES):,}", flush=True)

    print("\nper-word totals:")
    for word in WORD_VALUES:
        print(f"  {word.lower():10s} {total.get(word, 0):5d}")
    print(f"\n  {sum(total.values()):,} utterances staged under {stage}")


if __name__ == "__main__":
    main()
