"""Mine real command phrasings from the training transcripts.

On held-out real speech the two largest error cells are query.info -> none
(263) and media.control -> none (233), and the failing transcripts share one
feature: open-vocabulary content the hand-written templates never contain.
People ask "is it warm in Florida in September" and say "play lady by kenny
roger"; the templates say "what's the weather" and "play some music". Faced
with "Florida" or "Nicki Minaj", the model has only ever heard such words in
`none`, and rejects.

STOP and SLURP transcripts carry those phrasings with gold labels. This turns
them into templates for the synthetic generator, so every Piper voice says
real command phrasings, entities included:

    STOP GET_WEATHER               -> query.info  kind=weather   verbatim
    STOP PLAY/PAUSE/STOP/SKIP      -> media.control action       verbatim
    STOP CREATE_TIMER              -> timer.set   duration abstracted to {number} {unit}
    SLURP light / media / query    -> the matching intent, room and state abstracted

Two guards keep the evaluation honest. Only train-split transcripts are
mined, and any phrasing that exactly matches a test transcript is dropped, so
a gain on the test set is generalisation rather than memorised text.
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from add_stop import SLOT_RE, parse_duration, parse_intent

REPO = Path(__file__).resolve().parent
SPEC = yaml.safe_load(open(REPO / "commands.yaml"))

MIN_WORDS, MAX_WORDS = 2, 16

STOP_MEDIA = {"PLAY_MUSIC": "play", "PAUSE_MUSIC": "pause",
              "STOP_MUSIC": "stop", "SKIP_TRACK_MUSIC": "next"}
NESTED = re.compile(r"\[SL:[A-Z_]+[^\]]*\[IN:")


def clean(text: str) -> str:
    """Lowercase, keep letters and apostrophes, collapse whitespace."""
    text = re.sub(r"[^a-z' ]", " ", text.lower().replace("’", "'"))

    return re.sub(r"\s+", " ", text).strip()


def usable(text: str) -> bool:
    """Whether a phrasing is worth rendering: speakable length, no digits."""
    return MIN_WORDS <= len(text.split()) <= MAX_WORDS and not re.search(r"\d", text)


def unbracket(parse: str, replace: dict[str, str]) -> str | None:
    """
    Rebuild the spoken text from a STOP seqlogical parse.

    Args:
        parse: The normalized_seqlogical column, words with bracketed slots
        replace: Slot name to replacement text; unlisted slots keep their words

    Returns:
        The rebuilt utterance, or None when the parse nests an intent inside a
        slot, which the flat rewrite cannot represent
    """
    if NESTED.search(parse):
        return None

    def swap(match: re.Match) -> str:
        return replace.get(match.group(1), match.group(2))

    text = SLOT_RE.sub(swap, parse)
    text = re.sub(r"\[IN:[A-Z_]+", " ", text).replace("]", " ")

    return re.sub(r"\s+", " ", text).strip()


def mine_stop(tsv: Path) -> dict[str, list]:
    """
    Pull weather, media and timer phrasings from STOP's training manifest.

    Returns:
        Intent mapped to a list of (text, key) pairs, key being the
        templates_by value for query.info and media.control, None for timers
    """
    found = defaultdict(list)
    for line in tsv.read_text().splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 9:
            continue
        domain, utterance, seqlogical = parts[1], parts[6], parts[7]
        intent = parse_intent(parts[8])

        if intent == "GET_WEATHER":
            found["query.info"].append((clean(utterance), "weather"))

        elif intent in STOP_MEDIA:
            found["media.control"].append((clean(utterance), STOP_MEDIA[intent]))

        elif intent == "CREATE_TIMER":
            spans = dict(SLOT_RE.findall(seqlogical))
            if "DATE_TIME" not in spans or parse_duration(spans["DATE_TIME"]) is None:
                continue
            text = unbracket(seqlogical, {"DATE_TIME": "NUMBER_UNIT"})
            if text:
                found["timer.set"].append((clean(text).replace("number unit", "{number} {unit}"), None))

    return found


def mine_slurp(manifest: Path) -> dict[str, list]:
    """
    Pull SLURP phrasings from the training manifest, abstracting room and state.

    A slot value is abstracted only when it occurs exactly once: "turn on the
    light on the table" has two candidate "on"s, and guessing wrong would
    teach the template the wrong word.

    Returns:
        Intent mapped to a list of (text, key) pairs
    """
    found = defaultdict(list)
    for line in open(manifest):
        record = json.loads(line)
        if record.get("model") != "slurp" or record["intent"] == "none":
            continue

        text, slots, intent = clean(record["text"]), record["slots"], record["intent"]

        if intent in ("media.control", "query.info"):
            key = slots.get("action" if intent == "media.control" else "kind")
            if key:
                found[intent].append((text, key))
            continue

        if intent not in ("light.set", "light.dim") or "percent" in slots:
            continue

        template, ok = f" {text} ", True
        for name in ("room", "state"):
            value = slots.get(name)
            if value is None:
                continue
            hits = template.count(f" {value} ")
            if hits != 1:
                ok = False
                break
            template = template.replace(f" {value} ", f" {{{name}}} ")

        # A state-less light.set cannot be rendered with a value, so skip it
        if ok and (intent != "light.set" or "{state}" in template):
            found[intent].append((template.strip(), None))

    return found


def spec_phrasings() -> set[str]:
    """Every hand-written template, so mining only adds what is new."""
    seen = set()
    for intent in SPEC["intents"].values():
        seen.update(clean(t) for t in intent.get("templates", []))
        for phrasings in (intent.get("templates_by") or {}).values():
            for group in phrasings.values():
                seen.update(clean(t) for t in group)

    return seen


def test_texts(data: Path) -> set[str]:
    """Normalised transcripts of every test row, the leakage exclusion set."""
    texts = set()
    for line in open(data / "manifest_test.jsonl"):
        texts.add(clean(json.loads(line).get("text", "")))

    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stop", default="data/stop/stop/manifests/train.tsv")
    ap.add_argument("--data", default="data/dataset")
    ap.add_argument("--out", default="data/mined_templates.json")
    args = ap.parse_args()

    data = REPO / args.data
    raw = defaultdict(list)
    for source in (mine_stop(REPO / args.stop), mine_slurp(data / "manifest_train.jsonl")):
        for intent, items in source.items():
            raw[intent].extend(items)

    spec, test = spec_phrasings(), test_texts(data)
    report, bank = {}, {}
    for intent, items in raw.items():
        seen, kept = set(), []
        dropped = Counter()
        for text, key in items:
            probe = text.replace("{number}", "").replace("{unit}", "").replace("{room}", "").replace("{state}", "")
            if not usable(probe):
                dropped["length or digits"] += 1
            elif text in spec:
                dropped["already a template"] += 1
            elif text in test:
                dropped["matches a test transcript"] += 1
            elif (text, key) in seen:
                dropped["duplicate"] += 1
            else:
                seen.add((text, key))
                kept.append((text, key))

        if intent in ("media.control", "query.info"):
            grouped = defaultdict(list)
            for text, key in kept:
                grouped[key].append(text)
            name = "action" if intent == "media.control" else "kind"
            bank[intent] = {"templates_by": {name: {k: sorted(v) for k, v in sorted(grouped.items())}}}
            sizes = {k: len(v) for k, v in grouped.items()}
        else:
            bank[intent] = {"templates": sorted(t for t, _ in kept)}
            sizes = len(kept)

        report[intent] = {"mined": len(items), "kept": len(kept), "by_key": sizes, "dropped": dict(dropped)}

    out = REPO / args.out
    json.dump(bank, open(out, "w"), indent=1)

    for intent, r in report.items():
        print(f"{intent:14s} mined {r['mined']:6,d}  kept {r['kept']:6,d}   {r['by_key']}")
        print(f"{'':14s} dropped {r['dropped']}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
