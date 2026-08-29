# ME1 - Einops/Einsum

Build a 3-layer CNN for MNIST classification with every layer and operation implemented
using `einops`/`einsum`, train for 5 epochs, report the test split accuracy, and display
16 sampled images in a 4x4 grid with the ground truth and the prediction.

Notebook: [`mnist_einops_cnn.ipynb`](mnist_einops_cnn.ipynb) (committed with outputs)

## Result

**Test split accuracy: 99.09%** — 9909 of 10000 test images, after 5 epochs.

| Epoch | Train loss | Test accuracy |
|-------|-----------|---------------|
| 1 | 0.1611 | 98.65% |
| 2 | 0.0379 | 98.53% |
| 3 | 0.0236 | 98.81% |
| 4 | 0.0176 | 99.23% |
| 5 | 0.0133 | 99.09% |

Test accuracy peaks at 99.23% on epoch 4 and settles at 99.09% while the train loss keeps
falling, so by the end of the 5-epoch budget the model has started to overfit slightly.
The reported figure is the accuracy after the required 5 epochs.

Of the 16 sampled test images, 16 are classified correctly.

## Preprocessing

The normalization mean and standard deviation are **measured from the training split**
rather than hardcoded:

```python
raw_train = datasets.MNIST(root='data', train=True, download=True).data.float() / 255
data_mean = raw_train.mean().item()   # 0.1307
data_std = raw_train.std().item()     # 0.3081
```

`.data` is the raw uint8 tensor before any transform, so dividing by 255 reproduces
exactly what `ToTensor` produces. Only the training split is used, otherwise information
from the test split would leak into the preprocessing. The mean is low because 80.9% of
MNIST pixels are exactly zero background. The same two values are reused to undo the
normalization when the sampled images are displayed.

## Architecture

```
Input                1 x 28 x 28
Conv1  1  -> 32   3x3 pad 1  ReLU  MaxPool 2x2  ->  32 x 14 x 14
Conv2  32 -> 64   3x3 pad 1  ReLU  MaxPool 2x2  ->  64 x 7 x 7
Conv3  64 -> 128  3x3 pad 1  ReLU               ->  128 x 7 x 7
Flatten                                         ->  6272
Linear 6272 -> 10                               ->  logits
```

155,402 trainable parameters. Three convolution layers; the classifier head is not
counted as a layer.

## What is implemented from scratch

| Operation | Implementation |
|---|---|
| `conv2d` | `Tensor.unfold` sliding-window view → `rearrange` → `einsum('b p l, o p -> b o l')` |
| `maxpool2d` | `reduce(x, 'b c (h p1) (w p2) -> b c h w', 'max')` |
| `flatten` | `rearrange(x, 'b c h w -> b (c h w)')` |
| global avg pool | `reduce(x, 'b c h w -> b c', 'mean')` |
| `linear` | `einsum(x, W, 'b i, o i -> b o') + bias` |
| `relu` | `x.clamp(min=0)` |
| cross entropy | log-sum-exp form, `einsum` against a one-hot matrix |

A convolution is a dot product between the kernel and every sliding window of the input,
so it becomes a single contraction once the windows are laid out as a matrix. The
`einsum` contracts over an axis of length `c * kh * kw`, which is exactly the per-window
dot product, for all output positions and all kernels at once.

No `nn.Conv2d`, `nn.Linear`, `nn.MaxPool2d`, `F.conv2d` or `F.unfold` is used in the
model. `Tensor.unfold` is used rather than `F.unfold` because it is a generic strided
window view rather than a convolution-specific helper, so no convolution machinery is
borrowed. Taken from PyTorch: autograd for the backward pass, the Adam optimizer,
tensor padding, and the data loaders.

Every operation is verified against its PyTorch reference in the notebook before
training, in double precision. Worst-case difference: `1.07e-14` (convolution); the rest
are exactly `0`. This matters because a subtly wrong convolution still trains and still
reaches a plausible accuracy, so the assertion rather than the accuracy is what
establishes correctness.

Global average pooling is implemented and verified but not used by the final model. An
earlier version used it in place of the flatten and reached 98.09%; collapsing each 7x7
feature map to a single number discards where in the image a feature fired, which costs
about a point on digits.

## Running it

```bash
pip install -r requirements.txt
jupyter lab mnist_einops_cnn.ipynb
```

Run all cells. MNIST downloads to `me1/data/` on first run (~64 MB, gitignored).
Takes about 45 seconds end to end on an A100; it runs on CPU too, just slower.

Measured on: Python 3.11, torch 2.10.0+cu128, einops 0.8.2, NVIDIA A100-SXM4-40GB.
Seeds are fixed, but GPU reductions are not bit-deterministic, so a rerun may move the
accuracy by around 0.1%.
