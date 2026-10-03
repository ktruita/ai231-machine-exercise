"""Choose the demo's rejection bias for a pair of runs, on the validation speakers.

The demo adds a constant to the `none` intent before decoding: positive says
"didn't catch that" more often, negative less. Behind a wake word the command
model hears only what follows "Marvin", so it can afford to act readily - as
long as the commands it stops rejecting come out right, and requests the device
cannot serve are still left alone. So the bias is the one with the most
(correct - wrong) actions on c_val's commands at the demo's 0.6 threshold that
still leaves at least 90% of c_val's out-of-scope clips alone, swept from +3 to
-3 in steps of 0.1; a tie goes to the larger, more cautious bias. For
fallback_hf this gives +1.1, the value in vcm_demo/run_demo.sh.

Each model runs once over c_val, and every bias is arithmetic on the cached
log-probabilities, decoded as the demo decodes: probabilities averaged, the bias
added to `none`, then the threshold.

Usage:
    python select_bias.py modelstore/vcm_hf:8000 modelstore/vcm_hf_s2:7500
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from dataloaders.vcm_dataloader import VoiceCommandDataset
from evaluate import load_run, spec_slots

DEMO_THRESHOLD = 0.6
MIN_LEFT_ALONE = 0.90
# Highest first, so a tie goes to the more cautious bias
BIASES = np.round(np.linspace(3.0, -3.0, 61), 2)


def cache(modules: list, dataset, device: str, batch: int = 128) -> tuple[list[dict], list[dict]]:
    """
    Run every model over a dataset once.

    Args:
        modules: Trained modules with the same label space
        dataset: Un-augmented dataset, in manifest order
        device: Device to run on
        batch: Items per forward pass (default: 128)

    Returns:
        Per model, log-probabilities keyed by head, (N, C) each; per row, the
        label it is scored against
    """
    logp, truths = [{} for _ in modules], []
    with torch.inference_mode():
        for start in range(0, len(dataset), batch):
            idx = range(start, min(start + batch, len(dataset)))
            waves = torch.stack([dataset[i]["waveform"] for i in idx]).to(device)
            for store, module in zip(logp, modules):
                for head, z in module(waves).items():
                    store.setdefault(head, []).append(torch.log_softmax(z.float(), -1).cpu().numpy())

            # A slot the row cannot give a value for is not graded, as in training
            for i in idx:
                record = dataset.records[i]
                truth = {"intent": record["intent"]}
                truth.update({n: record["slots"].get(n) for n in spec_slots(modules[0], record["intent"])
                              if n not in record.get("unsupervise", ())})
                truths.append(truth)

    return [{head: np.concatenate(parts) for head, parts in store.items()} for store in logp], truths


def decode(module, logp: list[dict], bias: float) -> tuple[list[dict], np.ndarray]:
    """
    Average the models' probabilities, add the bias to `none`, and decode.

    Returns:
        The commands, and the confidence of each chosen intent
    """
    heads = {}
    for head in logp[0]:
        stack = np.stack([member[head] for member in logp])                 # (M, N, C)
        heads[head] = np.log(np.exp(stack).mean(axis=0) + 1e-12)            # (N, C)

    intent = heads["intent"].copy()
    intent[:, module.backbone.intent_names.index("none")] += bias
    intent -= np.log(np.exp(intent).sum(axis=1, keepdims=True))
    heads["intent"] = intent
    commands = module.backbone.decode({head: torch.from_numpy(v) for head, v in heads.items()})

    return commands, np.exp(intent.max(axis=1))


def rates(commands: list[dict], confidence: np.ndarray, truths: list[dict]) -> tuple[float, float, float]:
    """
    Score one bias at the demo threshold.

    Returns:
        Shares of the commands acted on correctly and wrongly, and the share of
        out-of-scope rows left alone
    """
    correct = wrong = total = left = out_of_scope = 0
    for command, p, truth in zip(commands, confidence, truths):
        acts = command["intent"] != "none" and p >= DEMO_THRESHOLD
        if truth["intent"] == "none":
            out_of_scope += 1
            left += not acts
            continue
        total += 1
        exact = command["intent"] == truth["intent"] and all(
            command.get(k) == v for k, v in truth.items() if k != "intent")
        correct += acts and exact
        wrong += acts and not exact

    return correct / max(total, 1), wrong / max(total, 1), left / max(out_of_scope, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="the runs averaged in the demo, each as run or run:step")
    ap.add_argument("--split", default="c_val")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    modules, configs = [], []
    for run in args.runs:
        path, _, step = run.partition(":")
        module, cfg, ckpt_path = load_run(path, int(step) if step else None, args.device)
        modules.append(module)
        configs.append(cfg)
        print(f"  {path}  {ckpt_path.name}")

    # The first run's data and spec; the others share its label space
    run_dir = Path(args.runs[0].partition(":")[0])
    dataset = VoiceCommandDataset(data_dir=configs[0].dataloader.dataset.data_dir,
                                  spec_path=str(run_dir / "commands.yaml"), split=args.split,
                                  random_offset=False, augment=None)
    logp, truths = cache(modules, dataset, args.device)
    commands_n = sum(t["intent"] != "none" for t in truths)
    print(f"{args.split}: {commands_n:,} commands, {len(truths) - commands_n:,} out of scope; "
          f"threshold {DEMO_THRESHOLD}\n")

    print(f"  {'bias':>6s}  {'correct':>8s} {'wrong':>7s} {'c - w':>7s}  {'out of scope left alone':>24s}")
    chosen, best = None, -1.0
    for bias in BIASES:
        correct, wrong, left = rates(*decode(modules[0], logp, bias), truths)
        if left >= MIN_LEFT_ALONE and correct - wrong > best:
            chosen, best = bias, correct - wrong
        if abs(bias * 2 - round(bias * 2)) < 1e-9:
            print(f"  {bias:+6.1f}  {correct:8.3f} {wrong:7.3f} {correct - wrong:7.3f}  {left:24.3f}")

    if chosen is None:
        print(f"\nno bias leaves {MIN_LEFT_ALONE:.0%} of the out-of-scope clips alone; use 0")
        return
    correct, wrong, left = rates(*decode(modules[0], logp, chosen), truths)
    print(f"\nbias {chosen:+.1f}: correct {correct:.3f}, wrong {wrong:.3f}, out of scope left alone {left:.3f}"
          f"\nset BIAS={chosen:.1f} in vcm_demo/run_demo.sh, or pass --none-bias {chosen:.1f}")


if __name__ == "__main__":
    main()
