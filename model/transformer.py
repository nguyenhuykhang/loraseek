import numbers

import torch
import torch.nn as nn
from einops import rearrange

import localvit


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    """LayerNorm over the channel dimension of a [B, C, H, W] tensor."""
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class LocalViTFeedForward(nn.Module):
    def __init__(self, dim, expand_ratio=4.):
        super().__init__()
        self.net = localvit.LocalityFeedForward(
            in_dim=dim,
            out_dim=dim,
            stride=1,
            expand_ratio=expand_ratio,
            act='hs+se',
            reduction=4,
            wo_dp_conv=False,
            dp_first=False,
        )

    def forward(self, x):
        return self.net(x)


class MDTAAttention(nn.Module):
    """Multi-Dconv Head Transposed Attention (Restormer): attention across channels."""
    def __init__(self, dim, num_heads, bias=False):
        super(MDTAAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1,
                                    groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        return self.project_out(out)


class TransformerBlock2D(nn.Module):
    """x = x + attn(norm1(x)); x = x + ffn(norm2(x))"""
    def __init__(self, dim, num_heads, expansion=4, bias=False):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = MDTAAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = LocalViTFeedForward(dim, expand_ratio=expansion)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerNoCLS2DHigherLevel(nn.Module):
    """Stack of TransformerBlock2D operating on [B, C, H, W]; shape is preserved."""
    def __init__(self, dim, depth, num_heads=4, expansion=4, bias=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock2D(dim, num_heads, expansion, bias) for _ in range(depth)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


if __name__ == "__main__":
    model = TransformerNoCLS2DHigherLevel(dim=128, depth=2, num_heads=8, expansion=1, bias=False)
    inp = torch.randn(4, 128, 31, 33)
    out = model(inp)
    print("Input shape: ", tuple(inp.shape))
    print("Output shape:", tuple(out.shape))
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
