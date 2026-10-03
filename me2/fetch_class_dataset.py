"""Download the class dataset from Hugging Face, at the revisions fallback_hf was trained on.

airimonda/ai231-me2-voice-commands is a dataset repo of parquet files. This
fetches the three parts the pipeline reads into class_data/v2/hf/:

    data/                  the train, test and holdout splits     revision da92a79
    supplemental_synth/    more synthetic clips in the split's
                           voices, added to training               revision da92a79
    synthetic_negatives/   generated out-of-scope audio and the
                           noise clips training mixes in           revision 5abbe53

Pinned revisions rather than main, so a rebuild reads the same clips the models
were trained and tested on; synthetic_negatives arrived in a later commit. Each
file is written to a .part file and renamed once complete, so an interrupted
run leaves nothing that looks finished, and a file already present is skipped.
About 1.4 GB; no login is needed.

Usage:
    python fetch_class_dataset.py
"""
import argparse
import shutil
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent
URL = "https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands/resolve/{revision}/{path}"
DATA_REVISION = "da92a79ffde3031d5bb2a25138d9dd7d9f7ed006"
NEGATIVES_REVISION = "5abbe539a46b9b26ff24e73d2860d1698d44e81f"
FILES = {
    "data/train-00000-of-00002.parquet": DATA_REVISION,
    "data/train-00001-of-00002.parquet": DATA_REVISION,
    "data/test-00000-of-00001.parquet": DATA_REVISION,
    "data/holdout-00000-of-00001.parquet": DATA_REVISION,
    "supplemental_synth/train-00000-of-00001.parquet": DATA_REVISION,
    "synthetic_negatives/train-00000-of-00001.parquet": NEGATIVES_REVISION,
    "synthetic_negatives/test-00000-of-00001.parquet": NEGATIVES_REVISION,
}


def fetch(path: str, revision: str, out: Path) -> None:
    """
    Download one file of the dataset repo.

    Args:
        path: The file's path in the repo
        revision: Commit to read it from
        out: Folder the repo's layout is recreated under
    """
    target = out / path
    if target.exists():
        print(f"  have     {path}")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    with urllib.request.urlopen(URL.format(revision=revision, path=path)) as response, open(partial, "wb") as handle:
        shutil.copyfileobj(response, handle, length=1 << 20)
    partial.rename(target)
    print(f"  fetched  {path}  {target.stat().st_size / 1e6:.0f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="class_data/v2/hf")
    args = ap.parse_args()

    out = REPO / args.out
    for path, revision in FILES.items():
        fetch(path, revision, out)
    (out / "COMMIT").write_text(f"{DATA_REVISION}\n{NEGATIVES_REVISION} synthetic_negatives\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
