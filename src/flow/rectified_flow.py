import torch
import torch.nn.functional as F
from torch import nn

# x_t = (1-t) * noise + t * data, target velocity = data - noise.
# t is scaled to ~[0, 1000] before the UNet so the sinusoidal embedding has resolution.
TIME_SCALE = 1000.0


def _expand(t: torch.Tensor) -> torch.Tensor:
    return t[:, None, None, None]


class RectifiedFlow:
    def loss(self, model: nn.Module, x1: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.size(0), device=x1.device)
        xt = _expand(1 - t) * x0 + _expand(t) * x1
        v = model(xt, t * TIME_SCALE, y)
        return F.mse_loss(v, x1 - x0)

    def velocity(
        self, model: nn.Module, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, cfg_scale: float
    ) -> torch.Tensor:
        tin = t * TIME_SCALE
        if cfg_scale == 1.0:
            return model(x, tin, y)
        v_cond = model(x, tin, y)
        v_uncond = model(x, tin, torch.full_like(y, model.null_class))
        return v_uncond + cfg_scale * (v_cond - v_uncond)

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        y: torch.Tensor,
        steps: int,
        image_size: int = 32,
        channels: int = 1,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        device = next(model.parameters()).device
        x = torch.randn(y.size(0), channels, image_size, image_size, device=device)
        ts = torch.linspace(0, 1, steps + 1, device=device)
        for i in range(steps):
            t = ts[i].expand(y.size(0))
            x = x + self.velocity(model, x, t, y, cfg_scale) * (ts[i + 1] - ts[i])
        return x
