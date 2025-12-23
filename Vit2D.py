import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # Prefer using the shared config if available.
    from transformer import ViTConfig  # type: ignore
except Exception:
    @dataclass
    class ViTConfig:
        image_size: int = 224
        patch_size: int = 16
        in_channels: int = 3
        num_classes: int = 1000

        embed_dim: int = 768
        depth: int = 12
        num_heads: int = 12
        mlp_ratio: float = 4.0

        dropout: float = 0.0
        attn_dropout: float = 0.0

        # Optional: when > 0, use window attention instead of global attention.
        window_size: int = 0


def _pad_to_window_size(x: torch.Tensor, window_size: int):
    """Pad (B, H, W, C) on H/W so they are divisible by window_size."""
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h == 0 and pad_w == 0:
        return x, (H, W), (0, 0)
    # F.pad expects (..., W, C)?? For NHWC, pad is on last dims; easiest: pad in (W, H) with explicit.
    x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))  # pad: (C_left,C_right,W_left,W_right,H_left,H_right)
    return x, (H, W), (pad_h, pad_w)


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """(B, H, W, C) -> (B*num_windows, window_size*window_size, C)."""
    B, H, W, C = x.shape
    if H % window_size != 0 or W % window_size != 0:
        raise ValueError("H and W must be divisible by window_size (pad first)")
    x = x.view(
        B,
        H // window_size,
        window_size,
        W // window_size,
        window_size,
        C,
    )
    # (B, num_h, ws, num_w, ws, C) -> (B, num_h, num_w, ws, ws, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = x.view(-1, window_size * window_size, C)
    return windows


def window_reverse(
    windows: torch.Tensor,
    window_size: int,
    H: int,
    W: int,
) -> torch.Tensor:
    """(B*num_windows, window_size*window_size, C) -> (B, H, W, C)."""
    if H % window_size != 0 or W % window_size != 0:
        raise ValueError("H and W must be divisible by window_size")
    num_windows = (H // window_size) * (W // window_size)
    B = windows.shape[0] // num_windows
    x = windows.view(
        B,
        H // window_size,
        W // window_size,
        window_size,
        window_size,
        -1,
    )
    # (B, num_h, num_w, ws, ws, C) -> (B, num_h, ws, num_w, ws, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x

class PatchEmbedding(nn.Module):
    """
    把圖片 (B, C, H, W) 切成 patch, 並線性投影成 embeddings
    - flatten=True  : 回傳 (B, N, D)  (標準 ViT)
    - flatten=False : 回傳 (B, D, H/P, W/P) (保留 2D 網格)
    """
    def __init__(self, image_size: int, patch_size: int, in_channels: int, embed_dim: int, flatten: bool = True):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.flatten = flatten

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, D, H/P, W/P)
        if not self.flatten:
            return x
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class LayerNorm2d(nn.Module):
    """在 (B, D, H, W) 上做 LayerNorm（對每個位置正規化 channel 維度 D）"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, D, H, W) -> (B, H, W, D) -> LN(D) -> (B, D, H, W)
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        return x


class SpatialMLP(nn.Module):
    """2D 版 MLP：用 1x1 Conv 當作逐位置的 Linear"""
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SpatialMultiHeadSelfAttention(nn.Module):
    """
    輸入/輸出都維持 (B, D, H, W)。
    內部仍會把 H*W 展成 N 來算全域注意力。
    """
    def __init__(self, dim: int, num_heads: int, attn_dropout: float, proj_dropout: float):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, H, W)
        B, D, H, W = x.shape
        N = H * W

        qkv = self.qkv(x)  # (B, 3D, H, W)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, N)  # (B, 3, heads, Hd, N)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, heads, N, Hd)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, heads, N, Hd)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # (B, heads, N, Hd)
        out = out.transpose(2, 3).reshape(B, D, N)  # (B, D, N)
        out = out.reshape(B, D, H, W)  # (B, D, H, W)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class WindowMultiHeadSelfAttention(nn.Module):
    """Window-based MHSA for (B, D, H, W).

    Splits feature map into non-overlapping windows of size (window_size, window_size),
    then applies standard MHSA within each window.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        attn_dropout: float,
        proj_dropout: float,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if window_size <= 0:
            raise ValueError("window_size must be > 0")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, H, W)
        B, D, H, W = x.shape

        # (B, D, H, W) -> (B, H, W, D)
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        x_nhwc, (H0, W0), (pad_h, pad_w) = _pad_to_window_size(x_nhwc, self.window_size)
        Hp, Wp = x_nhwc.shape[1], x_nhwc.shape[2]

        windows = window_partition(x_nhwc, self.window_size)  # (BnW, ws*ws, D)
        BnW, N, _ = windows.shape

        qkv = self.qkv(windows)  # (BnW, N, 3D)
        qkv = qkv.reshape(BnW, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, BnW, heads, N, Hd)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (BnW, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # (BnW, heads, N, Hd)
        out = out.transpose(1, 2).reshape(BnW, N, D)  # (BnW, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)

        x_nhwc = window_reverse(out, self.window_size, Hp, Wp)  # (B, Hp, Wp, D)
        if pad_h or pad_w:
            x_nhwc = x_nhwc[:, :H0, :W0, :]

        x = x_nhwc.permute(0, 3, 1, 2).contiguous()  # (B, D, H, W)
        return x


class SpatialTransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        attn_dropout: float,
        window_size: int = 0,
    ):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        if window_size and window_size > 0:
            self.attn = WindowMultiHeadSelfAttention(
                dim,
                num_heads,
                window_size=window_size,
                attn_dropout=attn_dropout,
                proj_dropout=dropout,
            )
        else:
            self.attn = SpatialMultiHeadSelfAttention(dim, num_heads, attn_dropout=attn_dropout, proj_dropout=dropout)
        self.norm2 = LayerNorm2d(dim)
        self.mlp = SpatialMLP(dim, int(dim * mlp_ratio), dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT2D(nn.Module):
    """
    2D 介面的 ViT：
    - PatchEmbedding 回傳 (B, D, H', W')
    - 加 2D positional embedding (B, D, H', W')
    - encoder blocks 全程維持 2D
    - 最後用 global average pooling 取出 (B, D) 做分類
    """
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg

        self.patch_embed = PatchEmbedding(cfg.image_size, cfg.patch_size, cfg.in_channels, cfg.embed_dim, flatten=False)
        gs = self.patch_embed.grid_size  # H' = W' = image_size/patch_size

        self.pos_embed_2d = nn.Parameter(torch.zeros(1, cfg.embed_dim, gs, gs))
        self.pos_drop = nn.Dropout(cfg.dropout)

        window_size = int(getattr(cfg, "window_size", 0) or 0)
        self.blocks = nn.ModuleList([
            SpatialTransformerEncoderBlock(
                dim=cfg.embed_dim,
                num_heads=cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                dropout=cfg.dropout,
                attn_dropout=cfg.attn_dropout,
                window_size=window_size,
            )
            for _ in range(cfg.depth)
        ])
        self.norm = LayerNorm2d(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, cfg.num_classes)

        nn.init.trunc_normal_(self.pos_embed_2d, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)  # (B, D, H', W')
        x = x + self.pos_embed_2d
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = x.mean(dim=(2, 3))  # GAP: (B, D)
        logits = self.head(x)   # (B, num_classes)
        return logits