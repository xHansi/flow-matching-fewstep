import argparse
from pathlib import Path

import torch
from torch import optim
from tqdm import tqdm

from fmfs.data import DATASET_META, make_loaders
from fmfs.flow import make_method
from fmfs.models import UNet
from fmfs.utils import EMA, get_device, save_loss_curve, save_samples, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Flow Matching or DDPM on MNIST.")
    p.add_argument("--method", choices=["flow", "ddpm"], default="flow")
    p.add_argument("--dataset", choices=["mnist", "fashion"], default="mnist")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--max-steps", type=int, default=0, help="cap total steps (0 = full epochs)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--cfg-dropout", type=float, default=0.1, help="label dropout for CFG")
    p.add_argument("--base", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--preview-steps", type=int, default=8)
    p.add_argument("--preview-cfg", type=float, default=2.0)
    p.add_argument("--out", type=str, default="results")
    return p.parse_args()


def drop_labels(y: torch.Tensor, p: float, null_class: int) -> torch.Tensor:
    y = y.clone()
    y[torch.rand(y.size(0), device=y.device) < p] = null_class
    return y


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    meta = DATASET_META[args.dataset]

    arch = dict(
        in_ch=meta["channels"],
        base=args.base,
        num_classes=meta["num_classes"],
        image_size=meta["image_size"],
    )
    model = UNet(**arch).to(device)
    method = make_method(args.method)
    ema = EMA(model, args.ema_decay)
    opt = optim.AdamW(model.parameters(), lr=args.lr)

    train_loader, _ = make_loaders(
        args.dataset,
        image_size=meta["image_size"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    out_dir = Path(args.out) / f"{args.method}_{args.dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)

    losses: list[float] = []
    step = 0
    done = False
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            y = drop_labels(y, args.cfg_dropout, model.null_class)
            loss = method.loss(model, x, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)

            losses.append(loss.item())
            step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            if args.max_steps and step >= args.max_steps:
                done = True
                break
        if done:
            break

    save_loss_curve(losses, str(out_dir / "loss_curve.png"))

    ema_model = UNet(**arch).to(device)
    ema.copy_to(ema_model)
    ema_model.eval()
    y = torch.arange(meta["num_classes"], device=device).repeat_interleave(10)
    samples = method.sample(
        ema_model,
        y,
        steps=args.preview_steps,
        image_size=meta["image_size"],
        channels=meta["channels"],
        cfg_scale=args.preview_cfg,
    )
    save_samples(samples, str(out_dir / "samples.png"), nrow=meta["num_classes"])

    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "arch": arch,
            "method": args.method,
            "dataset": args.dataset,
            "losses": losses,
        },
        out_dir / "ckpt.pt",
    )
    print(f"saved checkpoint, loss curve and samples to {out_dir}")


if __name__ == "__main__":
    main()
