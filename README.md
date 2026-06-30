# Few-Step Image Generation: Flow Matching vs. DDPM

Single-student project for the Deep Learning course (DLAI, Sapienza).

**Research question.** How does image quality (FID) behave as a function of the number of
sampling steps for Flow Matching / Rectified Flow compared to DDPM? Hypothesis: Flow Matching
stays sharp at very few steps (2–8), while DDPM needs many steps (50+). Optional add-on: does
*reflow* measurably straighten the sampling paths and allow even fewer steps?

## Setup

Local (Apple Silicon / CPU / CUDA) with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run python -m scripts.train --method flow --dataset mnist   # or --method ddpm
```

On Kaggle (T4 GPU), torch is preinstalled:

```bash
pip install -r requirements.txt
```

## Structure

```
src/
  models/unet.py          time- and class-conditioned UNet
  flow/rectified_flow.py  interpolation, target velocity, ODE sampler
  flow/ddpm.py            baseline: noising schedule, DDPM/DDIM samplers
  data/datasets.py        MNIST (+ optional coins), 32x32, [-1, 1]
  utils/                  seeds, device, ema, plotting
scripts/
  train.py                --method {flow,ddpm} --dataset {mnist,coins}
  sample.py               generate at multiple step counts
  eval_fid.py             FID vs. step count
notebooks/
  01_demo.ipynb           end-to-end demo, runs on Kaggle
  02_results.ipynb        loads checkpoints, produces all report plots
```

## Status

Work in progress, built milestone by milestone:

- [x] 1 — repo init, utils/seeds, MNIST dataloader (32x32, [-1, 1])
- [x] 2 — time- and class-conditioned UNet
- [x] 3 — Rectified Flow: interpolation, velocity target, Euler sampler
- [x] 4 — training script + loss curve + first samples
- [x] 5 — DDPM baseline (DDPM + DDIM samplers)
- [ ] 6 — sampling at multiple step counts, sample grids
- [ ] 7 — FID vs. step count (pytorch-fid + MNIST-FID)
- [ ] 8 — class-conditional + classifier-free guidance, reflow
- [ ] 9 — optional second dataset (Roman coins)
- [ ] 10 — demo + results notebooks, final README

See [`AI_USAGE.md`](AI_USAGE.md) for the AI-use statement (mandatory deliverable).
