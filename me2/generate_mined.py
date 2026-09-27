"""Render mined phrasings into the training set, replacing template audio.

`mine_templates.py` turns real transcripts into templates. This renders them
with the same Piper voices and speaker pools as `generate_dataset.py`, and
swaps them in for an equal share of the hand-written-template rows.

Replacing rather than adding is what makes the result readable. Errors on real
speech skew heavily toward `none`, so simply adding thousands of command
utterances would also shift the class prior away from `none` and could look
like a win for reasons unrelated to phrasing. The swap is done per label -
per action for media.control, per kind for query.info - so every class count
and every slot-value balance is unchanged, and phrasing is the only variable.

Only the train split is touched. Val and test keep their template audio, so
comparisons against earlier runs stay valid.

Usage:
    python generate_mined.py --fraction 0.5
"""
import argparse
import json
import random
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

from generate_dataset import SPEC, article_fix, render, surface, values_of

REPO = Path(__file__).resolve().parent
KEYED = {"media.control": "action", "query.info": "kind"}


def fill(template: str, intent: str, rng: random.Random) -> tuple[str, dict]:
    """
    Choose slot values for a mined template and speak them.

    Args:
        template: Mined phrasing with {room}, {state}, {number} or {unit}
        intent: The template's intent, which fixes the valid ranges
        rng: Random source

    Returns:
        (spoken text, slot labels)
    """
    labels = {}
    if "{state}" in template:
        labels["state"] = rng.choice(values_of("state"))
    if "{room}" in template:
        labels["room"] = rng.choice(values_of("room"))
    if "{unit}" in template:
        unit = rng.choice(values_of("unit"))
        low, high = SPEC["intents"][intent].get("number_range_by_unit", {}).get(
            unit, SPEC["slots"]["number"]["range"])
        labels["unit"], labels["number"] = unit, rng.randint(low, high)

    # "set a five minute timer": a duration used as a modifier takes the
    # singular, as the spec's {unit:singular} does for its own templates
    attributive = any(f"{{unit}} {w}" in template for w in ("timer", "countdown", "alarm"))

    text = template
    for name, value in labels.items():
        mod = "singular" if name == "unit" and attributive else None
        text = text.replace(f"{{{name}}}", str(surface(name, value, labels, mod)))

    return article_fix(text), labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="data/mined_templates.json")
    ap.add_argument("--data", default="data/dataset")
    ap.add_argument("--fraction", type=float, default=0.5,
                    help="share of each label's template rows to replace")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    data = REPO / args.data
    bank = json.load(open(REPO / args.bank))
    rng = random.Random(SPEC["splits"]["tts_speakers"]["seed"] + 1)

    # Train-split voices, weighted exactly as generate_dataset.py weights them
    splits = json.load(open(REPO / "speaker_splits.json"))["split"]
    over = SPEC["splits"]["tts_speakers"].get("oversample") or {}
    voices = [(k.split("|")[0], None if k.split("|")[1] == "-" else int(k.split("|")[1]), k)
              for k, sp in splits.items() if sp == "train"]
    weights = [over.get(m, 1.0) for m, _, _ in voices]

    rows = [json.loads(line) for line in open(data / "manifest_train.jsonl")]
    groups = defaultdict(list)
    for n, row in enumerate(rows):
        if row["intent"] in bank and row.get("model", "").startswith("en_"):
            key = row["slots"].get(KEYED[row["intent"]]) if row["intent"] in KEYED else None
            groups[(row["intent"], key)].append(n)

    drop, jobs = set(), []
    for (intent, key), members in sorted(groups.items(), key=lambda x: (x[0][0], str(x[0][1]))):
        spec = bank[intent]
        if intent in KEYED:
            pool = spec["templates_by"][KEYED[intent]].get(key, [])
        else:
            pool = spec["templates"]
        if not pool:
            print(f"  {intent:14s} {str(key):12s} no mined phrasings - left as is")
            continue

        n = round(len(members) * args.fraction)
        drop.update(rng.sample(members, n))
        for i in range(n):
            text, labels = fill(rng.choice(pool), intent, rng)
            if key is not None:
                labels = {KEYED[intent]: key}
            model, sid, voice_key = rng.choices(voices, weights=weights)[0]
            jobs.append({"uid": f"mined_{intent.replace('.', '-')}_{key or 'all'}_{i:06d}",
                         "split": "train", "intent": intent, "text": text, "slots": labels,
                         "model": model, "speaker_id": sid, "speaker_key": voice_key,
                         "seed": rng.randrange(2 ** 31)})
        print(f"  {intent:14s} {str(key):12s} replace {n:5,d} of {len(members):5,d} from {len(pool):6,d} phrasings")

    print(f"\n{len(jobs):,} utterances to render, {len(drop):,} template rows to retire")
    for job in rng.sample(jobs, 6):
        print(f"    {job['intent']:14s} {json.dumps(job['slots']):36s} \"{job['text'][:60]}\"")
    if args.dry_run:
        return

    # Contiguous chunks by voice, as generate_dataset.py does, so most workers
    # load one Piper model instead of all of them
    jobs.sort(key=lambda j: (j["model"], j["speaker_id"] if j["speaker_id"] is not None else -1))
    w = args.workers
    shards = [jobs[i * len(jobs) // w:(i + 1) * len(jobs) // w] for i in range(w)]
    with Pool(w) as p:
        rendered = [r for shard in p.map(render, [(i, s, data) for i, s in enumerate(shards)]) for r in shard]

    kept = [row for n, row in enumerate(rows) if n not in drop]
    with open(data / "manifest_train.jsonl", "w") as handle:
        for row in kept + rendered:
            handle.write(json.dumps(row) + "\n")

    print(f"\nmanifest_train.jsonl: {len(rows):,} -> {len(kept) + len(rendered):,} rows "
          f"({len(drop):,} retired, {len(rendered):,} rendered)")


if __name__ == "__main__":
    main()
