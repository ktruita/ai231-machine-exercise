# ME2 - Voice Command Model

Voice commands on a Raspberry Pi. Say "Marvin", then a command - "set a timer
for thirty seconds", "change color to blue" - and the device answers with the
command and its value, or "didn't catch that".

The model is **`fallback_hf`**: two copies of one small CNN, trained from
scratch with different random seeds on the class dataset alone,
[airimonda/ai231-me2-voice-commands](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands),
with their probabilities averaged. It knows the dataset's 19 commands and
leaves anything else alone.

| | `fallback_hf` |
|---|---|
| Class test, 4,367 commands by speakers never heard in training | 92.8% right with no threshold, 91.4% at the demo's 0.6 |
| ... the 3,590 in synthetic voices | 98.1% / 97.6% |
| ... the 777 recorded by real people | 68.3% / 62.7% |
| Wrong actions on those commands at 0.6 | 1.8%; another 6.8% get "didn't catch that" |
| Out-of-scope test clips acted on at 0.6 | 7 of 76 |
| Generated non-commands acted on (noise, babble, reversed or cut-off speech) | 13 of 250 |
| Size | 684,556 parameters and 2.7 MB per model (fp32 ONNX); the wake word 165 KB |
| Raspberry Pi 5, one thread | about 20 ms per command (p95) |

"Right" means the command and its value are both correct. Real voices are the
hard part: most of the dataset is synthetic speech, so the model is at its best
on synthetic voices, and weaker on real people, in heavy noise (73.0% at 0 dB,
no threshold) and in rooms it has not heard (66.5%).

1. [Setup](#1-setup)
2. [Run the trained model](#2-run-the-trained-model)
3. [Get the data](#3-get-the-data)
4. [Train](#4-train)
5. [Choose checkpoints, evaluate, export](#5-choose-checkpoints-evaluate-export)
6. [The wake word](#6-the-wake-word)
7. [How it works](#7-how-it-works)

Commands run from this folder, `me2/`, unless they say otherwise.

## 1. Setup

Training needs Python 3.11 and an NVIDIA GPU; `requirements.txt` pins the
versions it was run with. Running the trained model needs only numpy,
onnxruntime and sounddevice: no torch, no GPU.

    git clone https://github.com/ktruita/ai231-machine-exercise.git
    cd ai231-machine-exercise/me2
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

For the demo alone, on a laptop or a Raspberry Pi:

    pip install numpy onnxruntime sounddevice

On Linux, `sounddevice` also needs PortAudio: `sudo apt install libportaudio2`.

## 2. Run the trained model

The trained models are in `vcm_demo/deploy/`: `vcm_hf` and `vcm_hf_s2`, the
two command models, and `wakeword_marvin`. Nothing has to be trained or
downloaded to use them.

### On recordings

    cd vcm_demo
    ./run_demo.sh fallback_hf --wav clip1.wav clip2.wav clip3.wav

prints a line per file - here a timer command, someone talking about
something else, and a request the model is unsure of:

    preset   fallback_hf: deploy/vcm_hf deploy/vcm_hf_s2  (none bias 1.1)
    model    deploy/vcm_hf + deploy/vcm_hf_s2  (6.0s window, 7 heads)
      clip1                                  -> TIMER          duration=30 seconds                p=0.999  [15.7 ms]
      clip2                                  -> none                                              p=0.947  [12.8 ms]
      clip3                                  -> VOLUME_DOWN                                       p=0.328  [13.0 ms]  (below threshold)

`p` is the confidence. `none` means "not a command", and below 0.6 the live
demo answers "didn't catch that". The files must be 16 kHz mono 16-bit WAV;
only the first 6 s are used. Convert anything else with
`ffmpeg -i input.m4a -ar 16000 -ac 1 -sample_fmt s16 clip.wav`.

### Live, with a microphone

    cd vcm_demo
    python demo.py --list-devices            # find the microphone's number
    ./run_demo.sh fallback_hf --device 2     # say "Marvin", then the command

Then open http://localhost:8000, or the Pi's address from another machine,
for the browser console: the recognised command lit on the board of 19
commands, a widget acting on it (a countdown, a light, a thermostat...), and
the confidence and latency. The terminal prints the same decisions.

- Just speak and stop: speech is found by its level. After "Marvin" the demo
  waits up to 5 s for the command.
- `--no-wakeword` decodes everything said, without "Marvin".
- If the `rms` readout sits above 0.015 while nobody talks, raise the bar:
  `--speech-level 0.03`.
- On Windows, run `run_demo.sh` from Git Bash, or call the demo directly:
  `python demo.py --ui --threshold 0.6 --model deploy/vcm_hf deploy/vcm_hf_s2 --none-bias 1.1 --device 2`

### From Python

```python
import wave

import numpy as np
from vcm.runtime import CommandRecogniser          # run from vcm_demo/

recogniser = CommandRecogniser(["deploy/vcm_hf", "deploy/vcm_hf_s2"], none_bias=1.1)
with wave.open("clip.wav") as handle:              # 16 kHz mono 16-bit
    audio = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)

print(recogniser(audio.astype(np.float32) / 32768))
# {'command': {'intent': 'TIMER', 'duration': '30 seconds'}, 'confidence': 0.9986593723297119}
```

Inside, each model gives seven arrays of scores: 20 for the command (the 19
commands and `none`) and 4 for each of the six slots (its three values and
N/A). The command decides which slot is read.

### On a Raspberry Pi 5

- Use 64-bit Raspberry Pi OS (onnxruntime has no 32-bit ARM wheels) and a USB
  microphone: the Pi 5 has no audio input.
- `sudo apt install libportaudio2`, then in a virtual environment
  `pip install numpy onnxruntime sounddevice`.
- Copy `vcm_demo/` to the Pi and run it as above.
- `python bench_device.py` times the pair on the device.
- To check accuracy on the device itself, run `python build_validation_pack.py`
  in `vcm_demo/` on the training machine, once the data is built (section 3).
  It writes `vcm_demo_validation_hf.tar.gz`, 59 MB of held-out clips with the
  server's own decode of each. Extract it on the Pi in the folder that holds
  `vcm_demo/`, then run `python validate_pi.py` in `vcm_demo/`. It reports
  accuracy, non-commands acted on, wake word hits, latency, and whether every
  decode matches the server's. The pack holds the dataset's recordings, so
  keep it to yourself.

### For the class's live benchmark

    cd vcm_demo
    ./run_demo.sh fallback_hf --device 2 --bench-log <your-id>

writes `~/vcm_benchmark/<your-id>_<date-time>.log`, one JSON line per
decision, which the class benchmark ([airimonda/vcm-benchmark](https://github.com/airimonda/vcm-benchmark))
reads.

## 3. Get the data

    python fetch_class_dataset.py     # the dataset's parquet files, 1.4 GB, into class_data/v2/hf/
    python add_class_dataset.py       # WAVs and manifests, into class_data/v2/dataset/
    python build_hf_only.py           # the training set, into class_data/v2/hf_only/

No login is needed, and the download is pinned to the revisions the model was
trained on, so the result is byte for byte the data it was trained on.
Together they take 2.6 GB of disk and about 20 minutes after the download.

| split | clips | |
|---|---|---|
| `c_train` | 14,033 | the train split and its supplemental synthetic clips, and 882 generated negatives |
| `c_val` | 1,097 | 12% of the train split's speakers, held out to choose checkpoints and the bias |
| `c_test` | 4,443 | the test split: 4,367 commands and 76 out-of-scope clips |
| `c_holdout` | 202 | the holdout split, for live tests |

No speaker is in two splits. `class_data/` is never committed: the dataset
includes Fluent Speech Commands, which is for non-commercial use only, and the
group's own recordings.

## 4. Train

    python main.py experiment=hf
    python main.py experiment=hf seed=232 name=vcm_hf_s2 exp_dir=modelstore/vcm_hf_s2

- Each run is 8,000 steps of 128 clips, about 35 minutes on one A100, and saves
  a checkpoint every 500 steps to `modelstore/<name>/checkpoints/`.
- The network starts from random weights. The recipe is
  `configs/experiment/hf.yaml`.
- Run the same command again to resume an interrupted run from its newest
  checkpoint.
- `cluster.cpus=4` sets the data-loader workers (8 by default);
  `CUDA_VISIBLE_DEVICES=1` picks the GPU.
- Losses and validation accuracy are logged to `mlruns/`; `mlflow ui`, run
  from this folder, shows them.

## 5. Choose checkpoints, evaluate, export

    python select_checkpoint.py modelstore/vcm_hf modelstore/vcm_hf_s2
    python select_bias.py modelstore/vcm_hf:8000 modelstore/vcm_hf_s2:7500
    python evaluate_benchmark.py "modelstore/vcm_hf:8000+modelstore/vcm_hf_s2:7500@1.1"
    python export_onnx.py vcm_hf:8000 vcm_hf_s2:7500

1. `select_checkpoint.py` scores every checkpoint on `c_val` and writes each
   run's best step to `logs/selected_checkpoints.json`. For the models here
   that was 8,000 and 7,500; a run is then named `run:step`.
2. `select_bias.py` picks the `none` bias: the most correct minus wrong
   actions on `c_val` at the 0.6 threshold that still leaves 90% of its
   out-of-scope clips alone. For the models here, +1.1.
3. `evaluate_benchmark.py` scores the pair on `c_test` as the demo runs it.
   `--robustness` adds noise and reverberation (it needs MUSAN and RIRS_NOISES,
   section 6), and `--group-by accent_group model variation` breaks the score
   down.
4. `export_onnx.py` writes the ONNX bundles to `vcm_demo/deploy/`, replacing
   the shipped ones. If your bias differs, set `BIAS` in
   `vcm_demo/run_demo.sh`.

For a single model there is also `python evaluate.py modelstore/vcm_hf --step 8000`
(real and synthetic speech scored apart), and `python predict.py clip.wav --run
modelstore/vcm_hf --step 8000` decodes your own WAVs with the checkpoint, in
torch.

## 6. The wake word

The shipped `wakeword_marvin` was trained from scratch on Google Speech
Commands: "marvin" against the other 34 words. Retraining it needs downloads
from outside the class dataset:

| folder | what | from |
|---|---|---|
| `data/real_speech/speech_commands/` | Speech Commands v0.02 | [speech_commands_v0.02.tar.gz](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz) |
| `data/augment/musan/` | MUSAN noise and music | [openslr.org/17](https://www.openslr.org/17/) |
| `data/augment/RIRS_NOISES/` | room impulse responses | [openslr.org/28](https://www.openslr.org/28/) |
| `data/negatives/LibriSpeech/dev-clean/` | read speech, for false wakes | [openslr.org/12](https://www.openslr.org/12/) |

    python main.py experiment=wakeword           # 4,000 steps
    python export_onnx.py wakeword_marvin
    python evaluate_wakeword.py --n 3 --m 5      # false wakes per hour and misses, under the demo's rule

## 7. How it works

1. **Wake word.** A CNN of about 40k parameters scores the last second of
   audio every 100 ms and fires when 3 of the last 5 scores are above 0.99.
2. **Capture.** After "Marvin", the audio up to 0.8 s of silence, at most
   6 s, is the command.
3. **Features.** A 40-band log-mel spectrogram, a frame every 10 ms: 40 x 601
   for the 6-second window.
4. **Command model.** A 1-D CNN over time: a convolution stem, then three
   residual blocks of depthwise-separable convolutions whose dilation and
   stride grow, so each of the 76 output frames sees the whole window. Seven
   learned queries pool over time, one for the command and one per slot,
   into a command head (20 classes) and six slot heads (4 classes each).
5. **Decision.** The two models' probabilities are averaged, the bias is
   added to `none`, and the top command is acted on if its confidence is at
   least 0.6. The command picks which slot head is read.

Training grades a slot head only on clips of its own command, weights rare
commands up, and mixes in the dataset's own noise clips and a simulated
microphone and codec. Both models start from random weights: no pretrained
network is part of them or teaches them.

| file | |
|---|---|
| `commands.yaml` | the 19 commands, their slots and values |
| `fetch_class_dataset.py`, `add_class_dataset.py`, `build_hf_only.py` | the data |
| `main.py`, `configs/` | training, configured with Hydra |
| `vcm/` | the network, its heads and the log-mel front-end |
| `dataloaders/`, `lightning_module/` | data loading and augmentation, the training loop |
| `select_checkpoint.py`, `select_bias.py`, `evaluate*.py`, `export_onnx.py`, `predict.py` | choosing, scoring and exporting |
| `vcm_demo/` | the demo: `demo.py`, `run_demo.sh`, the browser console, the torch-free runtime, the models in `deploy/`, and tools for the Pi |
