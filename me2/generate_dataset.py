"""Render the ME2 voice-command dataset from commands.yaml with Piper.

Labels are free: the slot values are chosen before the sentence is spoken, so
nothing needs transcribing. Splits are speaker-disjoint and read from
speaker_splits.json.
"""
import argparse, itertools, json, os, random, re, wave
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import resample_poly

REPO = Path(__file__).resolve().parent
SPEC = yaml.safe_load(open(REPO / "commands.yaml"))
SR = SPEC["audio"]["sample_rate"]
PH = re.compile(r"\{(\w+)(?::(\w+))?\}")

ONES = ["zero","one","two","three","four","five","six","seven","eight","nine","ten",
        "eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen",
        "eighteen","nineteen"]
TENS = {20:"twenty",30:"thirty",40:"forty",50:"fifty",60:"sixty",70:"seventy",
        80:"eighty",90:"ninety"}


def say_number(n):
    if n < 20: return ONES[n]
    if n == 100: return "one hundred"
    t, o = divmod(n, 10)
    return TENS[t * 10] + ("" if o == 0 else " " + ONES[o])


def surface(name, value, ctx, mod=None):
    s = SPEC["slots"][name]
    if s.get("type") == "number":
        txt = say_number(value)
        return txt + (" " + s["spoken_suffix"] if s.get("spoken_suffix") else "")
    if "singular" in s and (mod == "singular" or ctx.get("number") == 1):
        return s["singular"][value]
    return value


def article_fix(t):
    return re.sub(r"\ba (eight|eleven|eighteen|eighty)\b", r"an \1", t)


def add_carrier(text: str, imperative: bool, rng: random.Random) -> str:
    """
    Wrap a bare template in address and politeness, the way people actually speak.

    Args:
        text: The rendered template
        imperative: False for question templates, which request forms cannot
            prefix - "can you what time is it" is not English
        rng: Random source, seeded per utterance so the result is reproducible

    Returns:
        The template with carriers applied
    """
    carriers = SPEC["generation"].get("carriers")
    if not carriers:
        return text

    used_request = False

    if rng.random() < carriers["prefix_probability"]:
        pool = carriers["address"] + (carriers["request"] if imperative else [])
        prefix = rng.choice(pool)
        # Address plus request together is common: "hey marvin can you ..."
        if imperative and prefix in carriers["address"] and rng.random() < 0.4:
            prefix = f"{prefix} {rng.choice(carriers['request'])}"
        used_request = any(r in prefix for r in carriers["request"])
        text = f"{prefix} {text}"

    if rng.random() < carriers["suffix_probability"]:
        # "would you turn off the lights will you" doubles the request form
        options = [x for x in carriers["suffixes"] if not (used_request and x == "will you")]
        text = f"{text} {rng.choice(options)}"

    return text


def values_of(n):
    s = SPEC["slots"][n]
    return (list(range(s["range"][0], s["range"][1] + 1, s.get("step", 1)))
            if s.get("type") == "number" else s["values"])


def expand(iname):
    i, out = SPEC["intents"][iname], []
    if i.get("templates_by"):
        k = next(iter(i["templates_by"]))
        return [(p, {k: v}) for v, ps in i["templates_by"][k].items() for p in ps]
    declared = i.get("slots", [])
    by_unit, over = i.get("number_range_by_unit") or {}, i.get("slot_ranges") or {}
    for t in i.get("templates", []):
        names = [n for n, _ in PH.findall(t)]
        if not names:
            out.append((t, {s: None for s in declared})); continue
        def vals(n):
            v = values_of(n)
            if n in over:
                lo, hi = over[n]; v = [x for x in v if lo <= x <= hi]
            return v
        for combo in itertools.product(*[vals(n) for n in names]):
            ctx = dict(zip(names, combo)); u = ctx.get("unit")
            if u in by_unit and not (by_unit[u][0] <= ctx.get("number", 0) <= by_unit[u][1]):
                continue
            txt = article_fix(PH.sub(
                lambda m: str(surface(m.group(1), ctx[m.group(1)], ctx, m.group(2))), t))
            out.append((txt, {s: ctx.get(s) for s in declared}))
    return out


_CACHE = {}


def _single_thread_onnx():
    """Piper builds a bare SessionOptions(), and onnxruntime sizes its intra-op
    pool from total system cores - it ignores both OMP_NUM_THREADS and the CPU
    affinity mask. That gave 17 threads per worker fighting over one pinned core
    (load average 84, ~2.5 utt/s overall). Patch the factory so every session
    Piper creates is single-threaded.
    """
    import onnxruntime as ort
    if getattr(ort, "_vcm_patched", False):
        return
    orig = ort.SessionOptions

    def factory():
        so = orig()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        return so

    ort.SessionOptions = factory
    ort._vcm_patched = True


def get_voice(model):
    _single_thread_onnx()
    from piper import PiperVoice
    if model not in _CACHE:
        # Models are ~75 MB and a load costs ~10 s, so keep them all resident
        # rather than thrashing when the work is not grouped by model.
        _CACHE[model] = PiperVoice.load(REPO / f"data/piper_voices/{model}.onnx")
    return _CACHE[model]


def render(job):
    from piper.config import SynthesisConfig
    core, items, outdir = job
    os.sched_setaffinity(0, {core})         # hard cap: one core per worker
    rows = []
    for it in items:
        rng = random.Random(it["seed"])
        v = get_voice(it["model"])
        cfg = SynthesisConfig(
            speaker_id=it["speaker_id"], normalize_audio=False,
            length_scale=round(rng.uniform(0.85, 1.15), 3),
            noise_scale=round(rng.uniform(0.60, 0.75), 3),
            noise_w_scale=round(rng.uniform(0.70, 0.90), 3))
        ch = list(v.synthesize(it["text"], syn_config=cfg))
        a = np.concatenate([np.frombuffer(c.audio_int16_bytes, dtype=np.int16) for c in ch])
        a = resample_poly(a.astype(np.float64), SR, ch[0].sample_rate)
        pk = np.abs(a).max(); ceil = 32767 * 10 ** (-1 / 20)
        if pk > ceil: a *= ceil / pk
        a = np.clip(a, -32768, 32767).astype(np.int16)
        lead = rng.randint(*SPEC["audio"]["lead_silence_ms"])
        trail = rng.randint(*SPEC["audio"]["trail_silence_ms"])
        a = np.concatenate([np.zeros(lead * SR // 1000, np.int16), a,
                            np.zeros(trail * SR // 1000, np.int16)])
        d = len(a) / SR
        if d > SPEC["audio"]["max_duration_s"]:
            a = a[:int(SPEC["audio"]["max_duration_s"] * SR)]; d = len(a) / SR
        rel = f"{it['split']}/{it['intent']}/{it['uid']}.wav"
        fp = outdir / rel; fp.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(fp), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(a.tobytes())
        rows.append({"file": rel, "split": it["split"], "intent": it["intent"],
                     "text": it["text"], "slots": it["slots"], "model": it["model"],
                     "speaker_id": it["speaker_id"], "speaker_key": it["speaker_key"],
                     "duration_s": round(d, 3), "length_scale": cfg.length_scale})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-intent", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="data/dataset")
    ap.add_argument("--only-intent", default=None,
                    help="regenerate a single intent, leaving the rest of the manifest alone")
    ap.add_argument("--append", action="store_true",
                    help="append to the manifests instead of overwriting, so real-speech rows survive")
    args = ap.parse_args()

    outdir = REPO / args.out; outdir.mkdir(parents=True, exist_ok=True)
    splits = json.load(open(REPO / "speaker_splits.json"))["split"]
    cfg = SPEC["splits"]["tts_speakers"]
    over = cfg.get("oversample") or {}

    pools = defaultdict(list)
    for key, sp in splits.items():
        model, sid = key.split("|")
        pools[sp].append((model, None if sid == "-" else int(sid), key))

    fracs = {"train": cfg["train_fraction"], "val": cfg["val_fraction"], "test": cfg["test_fraction"]}
    rng = random.Random(cfg["seed"])
    jobs = []
    for intent in SPEC["intents"]:
        if intent == "none": continue
        if args.only_intent and intent != args.only_intent: continue
        sents = expand(intent)
        for sp, frac in fracs.items():
            n = round(args.per_intent * frac)
            pool = pools[sp]
            # oversampling applies to training only - eval sets stay natural
            w = [over.get(m, 1.0) if sp == "train" else 1.0 for m, _, _ in pool]
            chosen = rng.choices(pool, weights=w, k=n)
            picks = (rng.sample(sents, n) if n <= len(sents)
                     else [sents[rng.randrange(len(sents))] for _ in range(n)])
            imperative = SPEC["intents"][intent].get("imperative", True)
            for i, ((txt, lbl), (model, sid, key)) in enumerate(zip(picks, chosen)):
                txt = add_carrier(txt, imperative, rng)
                jobs.append({"uid": f"{intent.replace('.','-')}_{sp}_{i:06d}", "split": sp,
                             "intent": intent, "text": txt,
                             "slots": {k: v for k, v in lbl.items() if v is not None},
                             "model": model, "speaker_id": sid, "speaker_key": key,
                             "seed": rng.randrange(2**31)})

    rng.shuffle(jobs)
    # Sort by model, then hand each worker a CONTIGUOUS chunk. Round-robin would
    # make every worker load all 8 models (~10s each); contiguous chunks mean
    # most workers load one. 84% of speakers are libritts_r, so that model gets
    # most of the workers, which is the right balance anyway.
    jobs.sort(key=lambda j: (j["model"], j["speaker_id"] if j["speaker_id"] is not None else -1))
    n, w = len(jobs), args.workers
    shards = [jobs[i * n // w:(i + 1) * n // w] for i in range(w)]
    print(f"{len(jobs):,} utterances | {args.workers} workers (cores 0-{args.workers-1}) | -> {outdir}")

    with Pool(args.workers) as p:
        results = p.map(render, [(i, sh, outdir) for i, sh in enumerate(shards)])

    rows = [r for shard in results for r in shard]
    by_split = defaultdict(list)
    for r in rows: by_split[r["split"]].append(r)
    mode = "a" if (args.only_intent or args.append) else "w"
    for sp, rs in by_split.items():
        with open(outdir / f"manifest_{sp}.jsonl", mode) as f:
            for r in sorted(rs, key=lambda x: x["file"]):
                f.write(json.dumps(r) + "\n")
        print(f"  {sp:6s} {len(rs):7,d} utts  {sum(x['duration_s'] for x in rs)/3600:6.2f} h")
    print(f"  TOTAL  {len(rows):7,d} utts  {sum(x['duration_s'] for x in rows)/3600:6.2f} h")


if __name__ == "__main__":
    main()
