# AI 231 — Machine Exercises

Coursework for AI 231. Each machine exercise is self-contained in its own folder.

| ME | Topic | Result |
|----|-------|--------|
| [ME1 - Einops/Einsum](me1/) | 3-layer CNN on MNIST, all layers built from `einops`/`einsum` | 99.09% test accuracy |
| [ME2 - Voice Command Model](me2/) | On-device voice commands for a Raspberry Pi: a 685k-parameter CNN trained from scratch, behind a "Marvin" wake word | 70% of real commands right at the demo threshold, 0.05 unwanted actions per hour |

ME1 is a notebook committed with its outputs, so its results are visible without
re-running it. ME2 is a training pipeline, with the Raspberry Pi demo in `me2/vcm_demo/`.
