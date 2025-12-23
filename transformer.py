import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class PatchEmbedding(nn.Module):
    """
    把圖片 (B, C, H, W) 切成 patch, 並線性投影成 token embeddings:
    (B, N, D) where N = (H/P)*(W/P)
    Conv2d(kernel=P, stride=P) 直接完成切 patch + 投影
    """
    def __init__(self, image_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, D, H/P, W/P)
        x = x.flatten(2)  # (B, D, H/P * W/P)
        # N, D 互換
        # 其中 D 是token的向量長度
        x = x.transpose(1, 2)  # (B, N, D)
        return x


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_dropout: float, proj_dropout: float):
        super().__init__()
        # 
        if dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # 公式中的開根號 d_k
        # 為了數值穩定性, 通常會把點積除以 sqrt(d_k) -> Normalization
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        # B 個 Batch, 每個 Batch 有 N 個 tokens, 每個 token 的 embedding shape 為 D
        B, N, D = x.shape
        qkv = self.qkv(x)  # (B, N, 3D)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # Q, K, V 分別的 dimensions: (B, H, N, Hd)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # (B, H, N, Hd)
        out = out.transpose(1, 2).reshape(B, N, D)  # (B, N, D)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float, attn_dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, attn_dropout=attn_dropout, proj_dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))  # residual
        x = x + self.mlp(self.norm2(x))   # residual
        return x


class ViT(nn.Module):
    """
    簡化版 ViT:
    - PatchEmbedding -> 加上 [CLS] token + positional embedding
    - 多層 Transformer encoder blocks
    - 取 CLS token 做分類
    """
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg

        self.patch_embed = PatchEmbedding(cfg.image_size, cfg.patch_size, cfg.in_channels, cfg.embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, cfg.embed_dim))
        self.pos_drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                dim=cfg.embed_dim,
                num_heads=cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                dropout=cfg.dropout,
                attn_dropout=cfg.attn_dropout,
            )
            for _ in range(cfg.depth)
        ])
        self.norm = nn.LayerNorm(cfg.embed_dim)

        self.head = nn.Linear(cfg.embed_dim, cfg.num_classes)

        self._init_weights()

    def _init_weights(self):
        # 常見初始化：pos/cls 用較小隨機值；Linear 用 xavier；LayerNorm 常規
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) 需與 cfg.image_size 對齊（先用 transforms resize/crop）
        x = self.patch_embed(x)  # (B, N, D)
        B, N, D = x.shape

        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([cls, x], dim=1)          # (B, 1+N, D)

        x = x + self.pos_embed[:, : (1 + N), :]
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_out = x[:, 0]  # (B, D)
        # logits = self.head(cls_out)  # (B, num_classes)
        return cls_out


if __name__ == "__main__":
    # 簡單 sanity check：先確認 shape 跑得通
    cfg = ViTConfig(image_size=224, patch_size=16, num_classes=10, embed_dim=396, depth=4, num_heads=6)
    model = ViT(cfg)
    dummy = torch.randn(2, 3, 224, 224)
    y = model(dummy)
    print("logits shape:", y.shape)  # (2, 10)