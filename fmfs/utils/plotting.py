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


def save_comparison(
    panels: list[tuple[str, dict]], steps: list[int], path: str, title: str
) -> None:
    """Side-by-side FID-vs-steps panels. Each panel is (label, {method: {step: fid}})."""
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4.2), sharey=True)
    axes = axes if len(panels) > 1 else [axes]
    styles = {"flow": "o-", "ddpm": "s--", "reflow": "^:"}
    for ax, (label, data) in zip(axes, panels, strict=True):
        for name, fmt in styles.items():
            if name in data:
                ax.plot(steps, [data[name][str(s)] for s in steps], fmt, label=name)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("sampling steps")
        ax.set_title(label)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("FID")
    axes[0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_loss_curve(losses: list[float], path: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(losses, lw=0.8)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
