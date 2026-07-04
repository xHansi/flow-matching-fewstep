import argparse
from pathlib import Path

import torch
from torch import optim
from tqdm import tqdm

from fmfs.data import DATASET_META
from fmfs.flow import RectifiedFlow
from fmfs.inference import load_checkpoint
from fmfs.models import UNet
from fmfs.utils import EMA, get_device, save_loss_curve, save_samples, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reflow: retrain a flow model on its own noise->sample pairs."
    )
    p.add_argument("--ckpt", required=True, help="teacher flow checkpoint")
    p.add_argument("--pairs", type=int, default=50000, help="number of coupling pairs")
    p.add_argument("--gen-steps", type=int, default=100, help="teacher ODE steps for coupling")
    p.add_argument("--gen-cfg", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--base", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--preview-steps", type=int, default=4)
    p.add_argument("--preview-cfg", type=float, default=2.0)
    p.add_argument("--out", default="results")
    return p.parse_args()


@torch.no_grad()
def build_coupling(rf, teacher, meta, n, steps, cfg, batch):
    device = next(teacher.parameters()).device
    x0s, x1s, ys = [], [], []
    for i in tqdm(range(0, n, batch), desc="coupling"):
        b = min(batch, n - i)
        y = torch.randint(0, meta["num_classes"], (b,), device=device)
        x0, x1 = rf.generate_coupling(teacher, y, steps, meta["image_size"], meta["channels"], cfg)
        x0s.append(x0.cpu())
        x1s.append(x1.cpu())
        ys.append(y.cpu())
    return torch.cat(x0s), torch.cat(x1s), torch.cat(ys)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    teacher, method, ckpt = load_checkpoint(args.ckpt, device, use_ema=True)
    assert ckpt["method"] == "flow", "reflow requires a flow checkpoint"
    meta = DATASET_META[ckpt["dataset"]]
    rf = RectifiedFlow()

    x0, x1, y = build_coupling(
        rf, teacher, meta, args.pairs, args.gen_steps, args.gen_cfg, args.batch_size
    )

    arch = ckpt["arch"]
    student = UNet(**arch).to(device)
    ema = EMA(student, args.ema_decay)
    opt = optim.AdamW(student.parameters(), lr=args.lr)

    out_dir = Path(args.out) / f"reflow_{ckpt['dataset']}"
    out_dir.mkdir(parents=True, exist_ok=True)

    losses: list[float] = []
    step, done = 0, False
    n = x0.size(0)
    for epoch in range(args.epochs):
        perm = torch.randperm(n)
        pbar = tqdm(range(0, n - args.batch_size + 1, args.batch_size), desc=f"epoch {epoch}")
        for i in pbar:
            idx = perm[i : i + args.batch_size]
            loss = rf.loss_coupled(
                student, x0[idx].to(device), x1[idx].to(device), y[idx].to(device)
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            ema.update(student)
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
    yy = torch.arange(meta["num_classes"], device=device).repeat_interleave(10)
    samples = rf.sample(
        ema_model,
        yy,
        steps=args.preview_steps,
        image_size=meta["image_size"],
        channels=meta["channels"],
        cfg_scale=args.preview_cfg,
    )
    save_samples(samples, str(out_dir / "samples.png"), nrow=meta["num_classes"])

    torch.save(
        {
            "model": student.state_dict(),
            "ema": ema.state_dict(),
            "arch": arch,
            "method": "flow",
            "dataset": ckpt["dataset"],
            "losses": losses,
            "reflow": True,
        },
        out_dir / "ckpt.pt",
    )
    print(f"saved reflow checkpoint, loss curve and samples to {out_dir}")


if __name__ == "__main__":
    main()
