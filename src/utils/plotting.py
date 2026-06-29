import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from torchvision.utils import save_image  # noqa: E402


def save_samples(x: torch.Tensor, path: str, nrow: int = 10) -> None:
    save_image(x, path, nrow=nrow, normalize=True, value_range=(-1, 1))


def save_loss_curve(losses: list[float], path: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(losses, lw=0.8)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
