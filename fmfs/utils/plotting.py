import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from torchvision.utils import save_image  # noqa: E402


def save_samples(x: torch.Tensor, path: str, nrow: int = 10) -> None:
    save_image(x, path, nrow=nrow, normalize=True, value_range=(-1, 1))


def save_trajectory(traj: torch.Tensor, path: str) -> None:
    """traj: (steps+1, B, C, H, W). Lays out B rows x (steps+1) columns (noise -> sample)."""
    s, b = traj.shape[:2]
    imgs = traj.permute(1, 0, 2, 3, 4).reshape(b * s, *traj.shape[2:])
    save_image(imgs, path, nrow=s, normalize=True, value_range=(-1, 1))


def save_fid_curve(curves: dict[str, tuple[list[int], list[float]]], path: str, title: str) -> None:
    plt.figure(figsize=(6, 4))
    for name, (steps, fids) in curves.items():
        plt.plot(steps, fids, marker="o", label=name)
    plt.xscale("log", base=2)
    plt.xlabel("sampling steps")
    plt.ylabel("FID")
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def save_loss_curve(losses: list[float], path: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(losses, lw=0.8)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
