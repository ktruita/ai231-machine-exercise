"""Export a trained run to a deployable bundle.

The bundle is everything the device needs and nothing it does not: an ONNX
graph, the mel filterbank as a plain array, and the label space as JSON. With
those three, inference needs only numpy and onnxruntime - no torch, no Hydra,
no commands.yaml.

The graph starts at the mel spectrogram rather than the waveform because
torch.stft does not export ("STFT does not currently support complex types").
The front-end therefore lives in vcm/features_numpy.py, which is verified to
give identical predictions to the torch version.

Bundles are written to vcm_demo/deploy/<run>/, where the demo reads them. A run
named as run:step exports that checkpoint, as select_checkpoint.py chose it.

Usage:
    python export_onnx.py vcm_hf:8000 vcm_hf_s2:7500    # fallback_hf's two models
    python export_onnx.py wakeword_marvin
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from utils import find_latest_checkpoint
from vcm.features import build_mel_filterbank
from vcm.spec import load_command_spec, build_label_space, active_slots, digit_slots

REPO = Path(__file__).resolve().parent


class MelToLogits(torch.nn.Module):
    """
    Wraps a backbone so the exported graph is mel in, logits out.

    The heads return a dictionary, which ONNX cannot express, so outputs are
    flattened to a tuple in sorted key order and the order is recorded in the
    bundle metadata.

    Flow: (B, num_mels, T) -> tuple of logits
    """

    def __init__(self, backbone: torch.nn.Module, head_names: list[str]) -> None:
        """
        Initialize the wrapper.

        Args:
            backbone: The trained backbone
            head_names: Output names, in the order the graph will emit them
        """
        super().__init__()

        self.backbone = backbone
        self.head_names = head_names

    def forward(self, mel: torch.Tensor):
        output = self.backbone(mel)

        if not isinstance(output, dict):
            return output

        return tuple(output[name] for name in self.head_names)


def load_module(run_dir: Path, spec_path: str, step: int | None = None):
    """
    Load a trained module in eval mode, the same way predict.py does.

    Args:
        run_dir: Run directory under modelstore/
        spec_path: Spec to build the label space from. The module reads this
            itself when constructing its heads, so overriding only the metadata
            would leave the backbone shaped by whatever commands.yaml currently
            says - which is how a pre-decomposition checkpoint failed to load.
        step: Checkpoint step to load, the newest if None (default: None)
    """
    cfg = OmegaConf.load(run_dir / "config.yaml")
    cfg.module.spec_path = str(spec_path)
    OmegaConf.resolve(cfg.module)

    module = instantiate(cfg.module.module, cfg.module)
    ckpt = (find_latest_checkpoint(run_dir / "checkpoints") if step is None
            else run_dir / "checkpoints" / f"checkpoint_step_{step}.ckpt")
    module.load_state_dict(torch.load(ckpt, weights_only=False, map_location="cpu")["state_dict"])

    return module.eval(), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run directories under modelstore/, each as run or run:step")
    ap.add_argument("--out", default="vcm_demo/deploy")
    ap.add_argument("--spec", default="commands.yaml",
                    help="spec that defined this run's label space; runs trained "
                         "before a spec change need the spec they were trained against")
    ap.add_argument("--int8", action="store_true",
                    help="also write model_int8.onnx, weights quantised to int8 for the Pi")
    args = ap.parse_args()

    spec = load_command_spec(REPO / args.spec)
    intent_names, slot_classes = build_label_space(spec)
    audio = spec["audio"]

    for spec_run in args.runs:
        run, _, step = spec_run.partition(":")
        run_dir = REPO / "modelstore" / run
        module, cfg = load_module(run_dir, REPO / args.spec, int(step) if step else None)
        backbone = module.backbone

        is_wakeword = not hasattr(backbone, "intent_names")
        head_names = [] if is_wakeword else ["intent"] + list(backbone.slot_names)

        # One second for the wake word, the full command window otherwise
        frames = 101 if is_wakeword else int(audio["max_duration_s"] * audio["sample_rate"] / 160) + 1
        num_mels = cfg.module.features.num_mels

        out_dir = REPO / args.out / run
        out_dir.mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            MelToLogits(backbone, head_names).eval(),
            torch.randn(1, num_mels, frames),
            str(out_dir / "model.onnx"),
            input_names=["mel"],
            output_names=head_names or ["logits"],
            # Frames vary with utterance length, so the time axis stays dynamic
            dynamic_axes={"mel": {0: "batch", 2: "frames"},
                          **{name: {0: "batch"} for name in (head_names or ["logits"])}},
            dynamo=False,
        )

        filterbank = build_mel_filterbank(
            sample_rate=audio["sample_rate"],
            n_fft=cfg.module.features.n_fft,
            num_mels=num_mels,
            f_min=cfg.module.features.f_min,
        ).numpy()
        np.save(out_dir / "filterbank.npy", filterbank)

        meta = {
            "run": spec_run,
            "kind": "wakeword" if is_wakeword else "command",
            "sample_rate": audio["sample_rate"],
            "n_fft": cfg.module.features.n_fft,
            "hop_length": cfg.module.features.hop_length,
            "num_mels": num_mels,
            "window_s": 1.0 if is_wakeword else audio["max_duration_s"],
            "head_names": head_names or ["logits"],
        }
        if not is_wakeword:
            # Everything the decoder needs, so the device never reads the spec
            meta.update({
                "intent_names": intent_names,
                "slot_classes": {k: list(v) for k, v in slot_classes.items()},
                "intent_slots": {name: active_slots(spec, name) for name in intent_names},
                "digit_slots": {k: list(v) for k, v in digit_slots(spec).items()},
            })

        json.dump(meta, open(out_dir / "meta.json", "w"), indent=1)

        if args.int8:
            # Dynamic quantisation covers the convolutions too, which hold 97% of
            # the weights here; activations are quantised per batch at run time,
            # so no calibration set is needed
            from onnxruntime.quantization import QuantType, quantize_dynamic
            quantize_dynamic(str(out_dir / "model.onnx"), str(out_dir / "model_int8.onnx"),
                             weight_type=QuantType.QInt8)
            print(f"  {run:18s} int8 {(out_dir / 'model_int8.onnx').stat().st_size / 1024:7.0f} KB")

        size = (out_dir / "model.onnx").stat().st_size / 1024
        params = sum(p.numel() for p in backbone.parameters())
        print(f"  {run:18s} {params:8,d} params  {size:7.0f} KB  "
              f"{len(head_names) or 1} outputs  window {meta['window_s']}s")


if __name__ == "__main__":
    main()
