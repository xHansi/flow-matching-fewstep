import torch
import torch.nn.functional as F
from torch import nn


def _expand(t: torch.Tensor) -> torch.Tensor:
    return t[:, None, None, None]


class DDPM:
    """eps-prediction DDPM. sample() uses the DDIM update; eta=0 is deterministic
    DDIM, eta=1 recovers stochastic DDPM ancestral sampling on the respaced schedule."""

    def __init__(self, num_steps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = num_steps
        betas = torch.linspace(beta_start, beta_end, num_steps)
        self.alphas_cumprod = torch.cumprod(1 - betas, dim=0)

    def loss(self, model: nn.Module, x0: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        t = torch.randint(0, self.T, (x0.size(0),), device=x0.device)
        noise = torch.randn_like(x0)
        acp = _expand(self.alphas_cumprod.to(x0.device)[t])
        xt = acp.sqrt() * x0 + (1 - acp).sqrt() * noise
        eps = model(xt, t.float(), y)
        return F.mse_loss(eps, noise)

    def eps(
        self, model: nn.Module, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, cfg_scale: float
    ) -> torch.Tensor:
        if cfg_scale == 1.0:
            return model(x, t, y)
        e_cond = model(x, t, y)
        e_uncond = model(x, t, torch.full_like(y, model.null_class))
        return e_uncond + cfg_scale * (e_cond - e_uncond)

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        y: torch.Tensor,
        steps: int,
        image_size: int = 32,
        channels: int = 1,
        cfg_scale: float = 1.0,
        eta: float = 0.0,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        device = next(model.parameters()).device
        acp = self.alphas_cumprod.to(device)
        seq = torch.linspace(0, self.T - 1, steps, device=device).round().long()

        x = torch.randn(y.size(0), channels, image_size, image_size, device=device)
        traj = [x.clone()]
        for i in reversed(range(steps)):
            t = seq[i]
            a_t = acp[t]
            a_prev = acp[seq[i - 1]] if i > 0 else torch.ones((), device=device)
            eps = self.eps(model, x, t.float().expand(y.size(0)), y, cfg_scale)
            x0 = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-1, 1)
            sigma = eta * ((1 - a_prev) / (1 - a_t)).sqrt() * (1 - a_t / a_prev).sqrt()
            noise = torch.randn_like(x) if eta > 0 and i > 0 else 0.0
            x = a_prev.sqrt() * x0 + (1 - a_prev - sigma**2).sqrt() * eps + sigma * noise
            traj.append(x.clone())
        return (x, torch.stack(traj)) if return_trajectory else x
