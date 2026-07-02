import argparse
import json
from pathlib import Path

import torch

from src.data import DATASET_META, make_loaders
from src.inference import load_checkpoint, sample_kwargs
from src.metrics import MNISTFID, InceptionFID, frechet, generate
from src.metrics.fid import features_over
from src.utils import get_device, save_fid_curve, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FID vs. sampling steps for one or more checkpoints.")
    p.add_argument("--ckpts", nargs="+", required=True, help="entries of the form name=path")
    p.add_argument("--dataset", default="mnist")
    p.add_argument("--steps", type=str, default="1,2,4,8,16,50,100")
    p.add_argument("--n-samples", type=int, default=5000)
    p.add_argument("--cfg-scale", type=float, default=2.0)
    p.add_argument("--eta", type=float, default=0.0, help="DDPM only: 0=DDIM, 1=ancestral")
    p.add_argument("--metric", choices=["inception", "mnist", "both"], default="both")
    p.add_argument("--clf-epochs", type=int, default=3, help="MNIST-FID classifier epochs")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/fid")
    return p.parse_args()


def real_images(dataset: str, n: int) -> torch.Tensor:
    _, test = make_loaders(dataset, batch_size=256, num_workers=2)
    imgs = []
    for x, _ in test:
        imgs.append(x)
        if sum(t.size(0) for t in imgs) >= n:
            break
    return torch.cat(imgs)[:n]


def build_metrics(names: list[str], device: torch.device, clf_epochs: int) -> dict:
    m = {}
    if "inception" in names:
        m["inception"] = InceptionFID(device)
    if "mnist" in names:
        m["mnist"] = MNISTFID(device, epochs=clf_epochs)
    return m


def main() -> None:
    args = parse_args()
    device = get_device()
    steps_list = [int(s) for s in args.steps.split(",")]
    metric_names = ["inception", "mnist"] if args.metric == "both" else [args.metric]

    metrics = build_metrics(metric_names, device, args.clf_epochs)
    reals = real_images(args.dataset, args.n_samples)
    real_feats = {k: features_over(m.features, reals, args.batch) for k, m in metrics.items()}

    results: dict = {k: {} for k in metrics}
    for entry in args.ckpts:
        name, path = entry.split("=", 1)
        model, method, ckpt = load_checkpoint(path, device)
        meta = DATASET_META[ckpt["dataset"]]
        extra = sample_kwargs(ckpt["method"], args.eta)
        for steps in steps_list:
            set_seed(args.seed)
            gen = generate(
                method, model, meta, args.n_samples, steps, args.cfg_scale, args.batch, **extra
            )
            for k, m in metrics.items():
                fid = frechet(real_feats[k], features_over(m.features, gen, args.batch))
                results[k].setdefault(name, {})[steps] = fid
                print(f"[{k}] {name} steps={steps:>3}: FID={fid:.2f}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fid.json").write_text(json.dumps(results, indent=2))
    for k, per_method in results.items():
        curves = {name: (steps_list, [d[s] for s in steps_list]) for name, d in per_method.items()}
        save_fid_curve(curves, str(out_dir / f"fid_vs_steps_{k}.png"), f"FID vs. steps ({k})")
    print(f"saved metrics and plots to {out_dir}")


if __name__ == "__main__":
    main()
