# Few-Step Image Generation: Flow Matching vs. DDPM

Single-student project for the Deep Learning course (DLAI, Sapienza).

**Research question.** How does image quality (FID) behave as a function of the number of
sampling steps for Flow Matching / Rectified Flow compared to DDPM? Hypothesis: Flow Matching
stays sharp at very few steps (2–8), while DDPM needs many steps (50+). Optional add-on: does
*reflow* measurably straighten the sampling paths and allow even fewer steps?

## Methods (in brief)

- **Flow Matching / Rectified Flow.** Linear interpolation `x_t = (1-t)·noise + t·data` with a
  constant target velocity `data - noise`. A UNet regresses the velocity (MSE); sampling
  integrates the ODE with an Euler sampler at a chosen step count.
- **DDPM baseline.** Linear beta schedule, ε-prediction. A unified DDIM update covers both the
  deterministic DDIM sampler (`eta=0`) and stochastic DDPM ancestral sampling (`eta=1`) on a
  respaced timestep schedule.
- Both are **class-conditional** with **classifier-free guidance** and share the same UNet,
  EMA weights, and evaluation code.
- **Reflow** retrains the flow model on its own `(noise, sample)` pairs to straighten the paths.

## Setup

Local (Apple Silicon / CPU / CUDA) with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run python -m scripts.train --method flow --dataset mnist   # or --method ddpm
```

On Kaggle (T4 GPU), torch is preinstalled — clone the repo and:

```bash
pip install -r requirements.txt   # or just: pip install pytorch-fid
```

## Reproduce the results

The two notebooks run top to bottom on a Kaggle T4:

- **`notebooks/01_demo.ipynb`** — end-to-end demo: trains a small flow model and shows
  class-conditional samples at 1/2/4/8 steps.
- **`notebooks/02_results.ipynb`** — trains both methods (+ optional reflow), runs the FID
  sweep, and produces every report figure (loss curves, FID-vs-steps, sample grids).

Or from the command line:

```bash
uv run python -m scripts.train --method flow --dataset mnist --epochs 30
uv run python -m scripts.train --method ddpm --dataset mnist --epochs 30
uv run python -m scripts.reflow --ckpt results/flow_mnist/ckpt.pt          # optional
uv run python -m scripts.sample --ckpt results/flow_mnist/ckpt.pt --steps 1,2,4,8,16,50,100
uv run python -m scripts.eval_fid \
    --ckpts flow=results/flow_mnist/ckpt.pt ddpm=results/ddpm_mnist/ckpt.pt \
    --steps 1,2,4,8,16,50,100 --n-samples 5000 --metric both
```

Everything is seeded; checkpoints are self-describing (architecture stored inside), so
`sample.py`/`eval_fid.py` rebuild the exact model from a checkpoint.

## Structure

```
fmfs/
  models/unet.py          time- and class-conditioned UNet (+ CFG null class)
  flow/rectified_flow.py  interpolation, velocity target, Euler sampler, reflow coupling
  flow/ddpm.py            baseline: beta schedule, eps prediction, DDPM/DDIM samplers
  data/datasets.py        MNIST (+ optional coins), 32x32, [-1, 1]
  metrics/                InceptionV3 FID (pytorch-fid) + lightweight MNIST-FID
  utils/                  seeds, device, EMA, plotting
  inference.py            checkpoint loader
scripts/
  train.py                --method {flow,ddpm}
  reflow.py               reflow a trained flow model
  sample.py               generate at multiple step counts
  eval_fid.py             FID vs. step count for one or more checkpoints
notebooks/
  01_demo.ipynb           end-to-end demo, runs on Kaggle
  02_results.ipynb        produces all report plots
```

## Report alignment (2 pages)

question/motivation · setup (Flow Matching + DDPM briefly, architecture, data) · results
(FID-vs-steps plot, sample grids) · discussion/limitations. All plots are reproducible directly
from `notebooks/02_results.ipynb`.

## Status

- [x] 1 — repo init, utils/seeds, MNIST dataloader (32x32, [-1, 1])
- [x] 2 — time- and class-conditioned UNet
- [x] 3 — Rectified Flow: interpolation, velocity target, Euler sampler
- [x] 4 — training script + loss curve + first samples
- [x] 5 — DDPM baseline (DDPM + DDIM samplers)
- [x] 6 — sampling at multiple step counts, sample grids
- [x] 7 — FID vs. step count (pytorch-fid + MNIST-FID)
- [x] 8 — reflow experiment (class-conditional + CFG already built in)
- [ ] 9 — optional second dataset (Roman coins) — dataloader is structured for it
- [x] 10 — demo + results notebooks, README

See [`AI_USAGE.md`](AI_USAGE.md) for the AI-use statement (mandatory deliverable).
