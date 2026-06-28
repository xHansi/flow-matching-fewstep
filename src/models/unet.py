import math

import torch
import torch.nn.functional as F
from torch import nn


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([args.cos(), args.sin()], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, dropout: float):
        super().__init__()
        self.in_layers = nn.Sequential(
            nn.GroupNorm(8, in_ch), nn.SiLU(), nn.Conv2d(in_ch, out_ch, 3, padding=1)
        )
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.out_layers = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        h = h + self.emb_proj(emb)[:, :, None, None]
        h = self.out_layers(h)
        return h + self.skip(x)


class AttnBlock(nn.Module):
    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w)
        q, k, v = (t.transpose(-1, -2) for t in qkv.unbind(1))
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-1, -2).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(F.interpolate(x, scale_factor=2, mode="nearest"))


class EmbedSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        for layer in self:
            x = layer(x, emb) if isinstance(layer, ResBlock) else layer(x)
        return x


class UNet(nn.Module):
    """Time- and class-conditioned UNet. The last label index is the null class for CFG."""

    def __init__(
        self,
        in_ch: int = 1,
        base: int = 64,
        ch_mults: tuple[int, ...] = (1, 2, 2),
        num_res_blocks: int = 2,
        num_classes: int = 10,
        attn_resolutions: tuple[int, ...] = (16,),
        dropout: float = 0.1,
        image_size: int = 32,
    ):
        super().__init__()
        emb_dim = base * 4
        self.freq_dim = base
        self.null_class = num_classes

        self.time_mlp = nn.Sequential(
            nn.Linear(base, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )
        self.label_emb = nn.Embedding(num_classes + 1, emb_dim)

        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)

        self.downs = nn.ModuleList()
        chs = [base]
        ch, res = base, image_size
        for i, mult in enumerate(ch_mults):
            out = base * mult
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, out, emb_dim, dropout)]
                ch = out
                if res in attn_resolutions:
                    layers.append(AttnBlock(ch))
                self.downs.append(EmbedSequential(*layers))
                chs.append(ch)
            if i != len(ch_mults) - 1:
                self.downs.append(EmbedSequential(Downsample(ch)))
                chs.append(ch)
                res //= 2

        self.middle = EmbedSequential(
            ResBlock(ch, ch, emb_dim, dropout), AttnBlock(ch), ResBlock(ch, ch, emb_dim, dropout)
        )

        self.ups = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mults))):
            out = base * mult
            for j in range(num_res_blocks + 1):
                layers = [ResBlock(ch + chs.pop(), out, emb_dim, dropout)]
                ch = out
                if res in attn_resolutions:
                    layers.append(AttnBlock(ch))
                if i != 0 and j == num_res_blocks:
                    layers.append(Upsample(ch))
                    res *= 2
                self.ups.append(EmbedSequential(*layers))

        self.out = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, in_ch, 3, padding=1))

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        emb = self.time_mlp(timestep_embedding(t, self.freq_dim)) + self.label_emb(y)
        h = self.in_conv(x)
        skips = [h]
        for block in self.downs:
            h = block(h, emb)
            skips.append(h)
        h = self.middle(h, emb)
        for block in self.ups:
            h = block(torch.cat([h, skips.pop()], dim=1), emb)
        return self.out(h)
