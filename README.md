# Few-Step Image Generation: Flow Matching vs. DDPM

Single-student project for the Deep Learning course (DLAI, Sapienza).

Modern image generators (DDPM) make great images but need **many** sampling steps. This
project asks whether **Flow Matching / Rectified Flow** can match them in **very few** steps,
using MNIST as a clean testbed, and studies image quality (FID) as a function of the number
of sampling steps.

**Research question.** How does FID behave as a function of the number of sampling steps for
Flow Matching vs. DDPM? *Hypothesis:* Flow Matching stays sharp at 2–8 steps; DDPM needs 50+.
*Optional add-on:* does *reflow* straighten the sampling paths so even fewer steps suffice?

**Headline result (first Kaggle T4 run, class-conditional MNIST, InceptionV3-FID, no
guidance).** Flow Matching reaches near-best FID already at ~4–8 steps; DDPM needs many more.
Even at **2 steps** Flow Matching produces clearly readable digits:

| Flow Matching — 2 steps | Flow Matching — 8 steps |
|---|---|
| ![](figures/flow_2steps.png) | ![](figures/flow_8steps.png) |

**How generation works (the iterations).** Each row is one digit; columns go left→right from
pure noise to the final image as the ODE is integrated in 8 Euler steps:

![Flow Matching sampling trajectory: noise to digit in 8 steps](figures/flow_trajectory_8steps.png)

`notebooks/03_walkthrough.ipynb` reproduces these and more (16-step trajectories, guidance
sweep, DDPM comparison).

---

## Table of contents

- [What's in here](#whats-in-here)
- [Install](#install)
- [Quickstart](#quickstart)
- [Local usage (detailed)](#local-usage-detailed)
- [Tests](#tests)
- [Notebooks](#notebooks)
- [Running on Kaggle (GPU)](#running-on-kaggle-gpu)
- [Method details](#method-details)
- [Second dataset: Roman coins](#second-dataset-roman-coins)
- [Troubleshooting](#troubleshooting)

---

## What's in here

```
fmfs/                         importable package (the "library")
  models/unet.py              time- and class-conditioned UNet (+ CFG null class)
  flow/rectified_flow.py      interpolation, velocity target, Euler sampler, reflow coupling
  flow/ddpm.py                baseline: beta schedule, eps prediction, DDPM/DDIM samplers
  data/datasets.py            MNIST loader; coins loader (FlatImageDataset)
  metrics/                    InceptionV3 FID (pytorch-fid) + lightweight MNIST-FID
  utils/                      seeds, device, EMA, plotting (grids, curves, trajectories)
  inference.py                checkpoint loader
scripts/
  train.py                    train a model            (--method flow|ddpm)
  reflow.py                   reflow a trained flow model
  sample.py                   generate grids at several step counts
  eval_fid.py                 FID vs. step count for one or more checkpoints
notebooks/
  01_demo.ipynb               short end-to-end demo (Kaggle)
  02_results.ipynb            trains everything + produces all report figures (Kaggle)
  03_walkthrough.ipynb        visual tour: few-step grids + sampling trajectories (local/Kaggle)
tests/                        fast CPU tests (pytest)
figures/                      headline figures committed for the report
report/REPORT.md              2-page report (draft)
AI_USAGE.md                   mandatory AI-use statement
```

Everything is **seeded**, and checkpoints are **self-describing** (they store the model
architecture), so `sample.py` / `eval_fid.py` rebuild the exact model from a checkpoint.

---

## Install

Local, with [uv](https://docs.astral.sh/uv/) (recommended — Python 3.11 is pinned):

```bash
cd flow-matching-fewstep
uv sync                       # creates .venv with torch, torchvision, pytorch-fid, ...
```

That's it. All commands below are prefixed with `uv run` so they use that environment. On
Apple Silicon the code automatically uses the **MPS** GPU; on a CUDA box it uses the GPU;
otherwise CPU. (No manual device flags.)

Prefer plain pip? `python -m venv .venv && source .venv/bin/activate && pip install -r
requirements.txt` works too — then drop the `uv run` prefix.

---

## Quickstart

Fastest path to *seeing* results locally (a short, rough training — minutes on a GPU, longer
on CPU; quality improves with more epochs):

```bash
uv run python -m scripts.train --method flow --dataset mnist --epochs 3
uv run python -m scripts.sample --ckpt results/flow_mnist/ckpt.pt --steps 1,2,4,8
open results/flow_mnist/samples_steps8.png     # macOS; use your image viewer otherwise
```

For the polished, illustrated version open **`notebooks/03_walkthrough.ipynb`** (it loads the
checkpoint you just trained, or trains a small one itself).

---

## Local usage (detailed)

All scripts are run as modules from the repo root (`python -m scripts.<name>`). Outputs go to
`results/<method>_<dataset>/` by default. Add `--help` to any script for the full flag list.

### 1) Train — `scripts/train.py`

```bash
uv run python -m scripts.train --method flow --dataset mnist --epochs 30
uv run python -m scripts.train --method ddpm --dataset mnist --epochs 30
```

Writes `results/<method>_mnist/`: `ckpt.pt` (weights + EMA + arch), `loss_curve.png`,
`samples.png` (class-conditional preview grid).

Key flags (defaults in brackets):

| flag | meaning |
|------|---------|
| `--method {flow,ddpm}` | Flow Matching or the DDPM baseline `[flow]` |
| `--dataset {mnist,coins}` | dataset `[mnist]` |
| `--epochs N` | training epochs `[30]` |
| `--max-steps N` | cap total optimizer steps (0 = full epochs) — handy for a quick local check `[0]` |
| `--batch-size N` | `[128]` |
| `--lr F` | AdamW learning rate `[2e-4]` |
| `--ema-decay F` | EMA decay for the sampling weights `[0.999]` |
| `--cfg-dropout F` | label-dropout prob for classifier-free guidance `[0.1]` |
| `--preview-steps N` / `--preview-cfg F` | sampling steps / guidance for the preview grid `[8] [2.0]` |
| `--seed N` | `[0]` |
| `--out DIR` | output root `[results]` |

Quick smoke (a few dozen steps, just to confirm the pipeline runs):

```bash
uv run python -m scripts.train --method flow --max-steps 50 --preview-steps 8
```

### 2) Sample at several step counts — `scripts/sample.py`

```bash
uv run python -m scripts.sample --ckpt results/flow_mnist/ckpt.pt \
    --steps 1,2,4,8,16,50,100 --per-class 10 --cfg-scale 2.0
```

Writes `samples_steps<K>.png` next to the checkpoint (one grid per step count; same random
seed per grid so grids are comparable). For DDPM add `--eta 0` (deterministic DDIM, default)
or `--eta 1` (stochastic ancestral DDPM).

### 3) Reflow a trained flow model — `scripts/reflow.py`

```bash
uv run python -m scripts.reflow --ckpt results/flow_mnist/ckpt.pt \
    --pairs 50000 --gen-steps 100 --epochs 30
```

Generates `(noise, sample)` pairs from the teacher and retrains a student on them to
straighten the ODE paths. The reflow checkpoint is a normal flow checkpoint, so `sample.py` /
`eval_fid.py` treat it like any other.

### 4) FID vs. step count — `scripts/eval_fid.py`

```bash
uv run python -m scripts.eval_fid \
    --ckpts flow=results/flow_mnist/ckpt.pt ddpm=results/ddpm_mnist/ckpt.pt \
    --steps 1,2,4,8,16,50,100 --n-samples 5000 --cfg-scale 1.0 --metric both
```

`--ckpts` takes `name=path` entries (any number). Writes `results/fid/fid.json` and one
`fid_vs_steps_<metric>.png` per metric. Two metrics: `inception` (standard pytorch-fid,
InceptionV3) and `mnist` (a small classifier trained on MNIST — domain-matched). **Measure
FID without guidance (`--cfg-scale 1.0`)**: guidance inflates FID and over-saturates DDPM at
high step counts, which would confound the steps comparison. First run needs internet (to
download the InceptionV3 weights, once).

---

## Tests

```bash
uv run pytest            # 7 fast CPU tests, a few seconds
```

Covers: UNet forward shapes, Flow/DDPM loss + backward, samplers at several step counts, the
DDIM/ancestral `eta` paths, classifier-free-guidance path, EMA updates, sampler determinism
under a fixed seed, the sampling-trajectory shape, and the coins loader (via synthetic images).

---

## Notebooks

| notebook | what it does | where |
|----------|--------------|-------|
| `01_demo.ipynb` | short end-to-end demo: train a small flow model, show few-step samples | Kaggle |
| `02_results.ipynb` | trains flow + ddpm (+ reflow), runs the FID sweep, produces **all** report figures | Kaggle |
| `03_walkthrough.ipynb` | visual tour: few-step grids, **sampling trajectories** (noise → digit), guidance effect | local **or** Kaggle |

Run a notebook locally:

```bash
uv run jupyter lab        # then open notebooks/03_walkthrough.ipynb
```

`03_walkthrough.ipynb` runs from the repo root directly. `01`/`02` are written for Kaggle
(their first cell clones the repo); locally you can run their commands from this README instead.

---

## Running on Kaggle (GPU)

The full training + FID sweep is meant for a free Kaggle GPU.

1. New Notebook → **Settings → Accelerator: GPU T4** (see the pitfall below) and **Internet: On**.
2. Upload `notebooks/02_results.ipynb` (or copy its cells). Its first cell clones this repo and
   installs `pytorch-fid`.
3. Run top to bottom. Figures land under `results/` and render inline.

Headless via the Kaggle CLI (what this project used):

```bash
kaggle kernels push -p <folder> --accelerator NvidiaTeslaT4
```

> **Pitfall (important):** Kaggle's default "GPU" is sometimes a **P100**, whose CUDA
> capability (sm_60) is **not supported** by the preinstalled PyTorch build — training fails
> with `CUDA error: no kernel image is available for execution on the device`. Always pick a
> **T4** (UI: "GPU T4 x2"; CLI: `--accelerator NvidiaTeslaT4`).

---

## Method details

- **Flow Matching / Rectified Flow.** `x_t = (1-t)·noise + t·data`, constant target velocity
  `data − noise`, MSE loss. Sampling integrates the ODE with an Euler sampler at a chosen
  step count.
- **DDPM baseline.** Linear β schedule, ε-prediction. A single DDIM update covers deterministic
  DDIM (`eta=0`) and stochastic ancestral DDPM (`eta=1`) on a respaced timestep schedule.
- **Shared UNet** (~6.5M params): sinusoidal time embedding + class embedding with a **null
  class** for classifier-free guidance; 3 resolution levels; self-attention at 16×16; EMA
  weights for sampling.
- **Reflow.** Retrain the flow model on its own `(noise, sample)` pairs to straighten paths.
- **Metrics.** InceptionV3 FID (`pytorch-fid`) + a lightweight MNIST-FID (small CNN classifier).

---

## Second dataset: Roman coins

The pipeline supports a second, unconditional grayscale dataset behind `--dataset coins`:

```bash
uv run python -m scripts.train --method flow --dataset coins --data-root /path/to/coin_images
```

`--data-root` is any folder of images (searched recursively; all treated as one class). On
Kaggle, attach a coins dataset and point `--data-root` at `/kaggle/input/<dataset>`. This is
scaffolded and unit-tested; a full coins run is future work.

---

## Troubleshooting

| symptom | cause / fix |
|---------|-------------|
| `CUDA error: no kernel image ...` on Kaggle | You got a **P100**. Switch the accelerator to **T4** (see above). |
| `ModuleNotFoundError: fmfs` when running a script | Run from the **repo root** as a module: `python -m scripts.train`. In notebooks the clone dir is added to `sys.path`. |
| FID step fails to download InceptionV3 | Enable **Internet** in Kaggle settings; the weights download once. |
| MNIST download fails | Same — needs internet the first time; cached afterwards under `data/`. |
| Training feels slow locally | MPS/CPU is much slower than a T4. Use `--max-steps` for a quick check, or run the full training on Kaggle. |

See [`AI_USAGE.md`](AI_USAGE.md) for the mandatory AI-use statement and [`report/REPORT.md`](report/REPORT.md) for the write-up.
