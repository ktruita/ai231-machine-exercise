"""A training set drawn from the class dataset alone, for a model that needs no other download.

Everything here comes from the class's Hugging Face repo
(airimonda/ai231-me2-voice-commands):

    c_train      add_class_dataset.py's c_train (the train split and the
                 supplemental synthetic clips in train voices) plus the
                 synthetic_negatives train clips - noise, babble, reversed
                 speech, cut-off commands and near-silence - as out of scope
    noise/       the negatives' noise_only train clips (DEMAND, MS-SNSD), as the
                 pool training adds background noise from, in place of MUSAN
    c_test_synthneg   the synthetic_negatives test clips, to measure how often a
                 model acts on audio that is not a command

A negative built from a clip whose speaker was held out for c_val is left out,
so c_val stays unheard. The held-out manifests are add_class_dataset.py's own,
rebased: no audio is copied, a manifest row points into ../dataset/.

Usage:
    python build_hf_only.py
"""
import argparse
import json
import os
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from add_class_dataset import iter_rows, write_clip

REPO = Path(__file__).resolve().parent
HELD_OUT = ("c_val", "c_test", "c_test_human", "c_test_negatives", "c_holdout")
COLUMNS = ["audio", "file", "transcript", "source", "duration_s", "neg_kind", "source_files"]


def val_sources(hf: Path, data: Path) -> set[str]:
    """The train split's files, as the negatives name them, whose speakers are in c_val."""
    hf_files = []
    for part in sorted((hf / "data").glob("train-*.parquet")):
        hf_files += pq.read_table(part, columns=["file"]).column("file").to_pylist()
    val = [json.loads(line) for line in open(data / "manifest_c_val.jsonl")]

    # add_class_dataset.py names each train clip by its row in the split
    return {f"train/{hf_files[int(Path(r['file']).stem)]}" for r in val if not r.get("supplemental")}


def negatives(split: str, held: set[str], hf: Path, out: Path) -> tuple[list[dict], Counter]:
    """Write one split of synthetic_negatives; return its manifest rows."""
    rows, counts = [], Counter()
    for i, row in enumerate(iter_rows(str(hf / "synthetic_negatives" / f"{split}-*.parquet"), COLUMNS)):
        sources = [s for s in (row["source_files"] or "").split(";") if s]
        if split == "train" and any(s in held for s in sources):
            counts["dropped, built from a c_val speaker"] += 1
            continue
        rel = f"negatives_{split}/none/{i:06d}.wav"
        duration = write_clip(row["audio"], out / rel)
        if duration is None:
            counts["unreadable"] += 1
            continue
        if split == "train" and row["neg_kind"] == "noise_only":
            write_clip(row["audio"], out / "noise" / "noise" / f"{i:06d}.wav")
            counts["noise pool"] += 1
        counts[row["neg_kind"]] += 1
        rows.append({"file": rel, "split": f"c_{split}", "intent": "none", "slots": {},
                     "text": row["transcript"], "model": row["source"], "speaker_id": "",
                     "speaker_key": f"{row['source']}|clip:{rel}", "accent_group": "Synthetic",
                     "variation": "", "is_synthetic": 1, "duration_s": round(duration, 3),
                     "neg_kind": row["neg_kind"]})

    return rows, counts


def rebased(row: dict, data: Path, out: Path) -> dict:
    """An add_class_dataset.py row whose file is relative to the output folder."""

    return dict(row, file=os.path.relpath(data / row["file"], out))


def write(out: Path, name: str, rows: list[dict]) -> None:
    with open(out / f"manifest_{name}.jsonl", "w") as handle:
        for record in rows:
            handle.write(json.dumps(record) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", default="class_data/v2/hf", help="the dataset repo, from fetch_class_dataset.py")
    ap.add_argument("--data", default="class_data/v2/dataset", help="add_class_dataset.py's output")
    ap.add_argument("--out", default="class_data/v2/hf_only")
    args = ap.parse_args()

    hf, data, out = REPO / args.hf, REPO / args.data, REPO / args.out
    out.mkdir(parents=True, exist_ok=True)

    held = val_sources(hf, data)
    train_negatives, train_counts = negatives("train", held, hf, out)
    test_negatives, test_counts = negatives("test", set(), hf, out)

    class_train = [rebased(json.loads(line), data, out) for line in open(data / "manifest_c_train.jsonl")]
    write(out, "c_train", class_train + train_negatives)
    for split in HELD_OUT:
        write(out, split, [rebased(json.loads(line), data, out) for line in open(data / f"manifest_{split}.jsonl")])
    write(out, "c_test_synthneg", test_negatives)

    train = class_train + train_negatives
    missing = sum(not (out / r["file"]).exists() for r in train)
    assert not missing, f"{missing} training files do not resolve"
    print(f"c_train: {len(train):,} clips - the class dataset {len(class_train):,}, synthetic negatives "
          f"{len(train_negatives):,}; out of scope {sum(r['intent'] == 'none' for r in train):,}")
    print("negatives, train:", dict(train_counts))
    print("negatives, test (c_test_synthneg):", dict(test_counts))


if __name__ == "__main__":
    main()
