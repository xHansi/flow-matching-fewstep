import argparse
from pathlib import Path

import torch

from fmfs.data import DATASET_META
from fmfs.inference import load_checkpoint, sample_kwargs
from fmfs.utils import get_device, save_samples, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate sample grids at several step counts.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--steps", type=str, default="1,2,4,8,16,50,100")
    p.add_argument("--per-class", type=int, default=10)
    p.add_argument("--cfg-scale", type=float, default=2.0)
    p.add_argument("--eta", type=float, default=0.0, help="DDPM only: 0=DDIM, 1=ancestral")
    p.add_argument("--use-ema", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None, help="defaults to the checkpoint folder")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    model, method, ckpt = load_checkpoint(args.ckpt, device, args.use_ema)
    meta = DATASET_META[ckpt["dataset"]]
    extra = sample_kwargs(ckpt["method"], args.eta)

    out_dir = Path(args.out) if args.out else Path(args.ckpt).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    y = torch.arange(meta["num_classes"], device=device).repeat(args.per_class)
    for steps in (int(s) for s in args.steps.split(",")):
        set_seed(args.seed)
        x = method.sample(
            model,
            y,
            steps=steps,
            image_size=meta["image_size"],
            channels=meta["channels"],
            cfg_scale=args.cfg_scale,
            **extra,
        )
        path = out_dir / f"samples_steps{steps}.png"
        save_samples(x, str(path), nrow=meta["num_classes"])
        print(f"steps={steps:>3} -> {path}")


if __name__ == "__main__":
    main()
