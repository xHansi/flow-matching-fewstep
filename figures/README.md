# Figures

Committed results so the repo tells its story without re-running anything. Regenerate with
`scripts/eval_fid.py` / `scripts/sample.py` or the notebooks.

```
mnist/          results on MNIST (digits)
  fid_vs_steps_inception.png    FID vs. steps, InceptionV3 features
  fid_vs_steps_classifier.png   FID vs. steps, domain-matched classifier features
  fid.json                      the raw numbers behind both plots
  flow_2steps.png / flow_8steps.png     Flow Matching sample grids
  flow_trajectory_{8,16}steps.png       noise → digit, step by step
  flow_loss.png                 training loss
  ddpm_8steps.png               DDPM sample grid
  ddpm_100steps_cfg{1,2}_*.png  the guidance-vs-steps finding (§3.3 of the report)
fashion/        the same, on Fashion-MNIST (clothing) — the domain-shift run
comparison/
  fid_vs_steps_inception.png    MNIST vs. Fashion side by side (InceptionV3)
  fid_vs_steps_classifier.png   MNIST vs. Fashion side by side (classifier-FID)
```

Regenerate the comparison plots from the committed JSONs with
`python -m scripts.compare_datasets` (no training needed).
