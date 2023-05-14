import math
import torch
import torch as th
from torch import nn, einsum
from inspect import isfunction
from einops import rearrange
from functools import partial

# helpers functions
num_class = 3

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class TimeStepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()    
        self.dim = dim

    def forward(self, x):
        half = self.dim // 2
        freqs = th.exp(
                -math.log(1000) * th.arange(start=0, end=half, dtype=th.float32) / half
                ).to(device=x.device)
        args = x[:, None].float() * freqs[None]
        embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def Upsample(dim):
    return nn.ConvTranspose1d(dim, dim, 4, 2, 1)


def Downsample(dim):
    return nn.Conv1d(dim, dim, 4, 2, 1)


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)
    

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 25, padding='same')
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.LeakyReLU(0.3)

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LeakyReLU(0.3),  # nn.SiLU() changed
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.mlp_label = nn.Sequential(
            nn.LeakyReLU(0.3),  # nn.SiLU() changed
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, label_emb=None):
        scale_shift = None
        scale_shift_label = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1')
            scale_shift = time_emb.chunk(2, dim=1)

        if exists(self.mlp_label) and exists(label_emb):
            label_emb = self.mlp_label(label_emb)
            label_emb = rearrange(label_emb, 'b c -> b c 1')
            scale_shift_label = label_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h, scale_shift=scale_shift_label)

        return h + self.res_conv(x)



class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x -> b h c x', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c x -> b (h c) x', h=self.heads, x=w)
        return self.to_out(out)



class Attention(nn.Module):
    def __init__(self, dim, heads=2, dim_head=8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)

        q, k, v = map(lambda t: rearrange(t, 'b (h c) x -> b h c x', h=self.heads), qkv)
        q = q * self.scale
        # print('q:', q.shape, 'k:', k.shape, 'v:', v.shape)

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        # print(out.shape)
        out = rearrange(out, 'b h x c -> b (h c) x')
        # print(out.shape)
        return self.to_out(out)


def LinearEncoding(dim_in, dim_hidden):
    return nn.Sequential(
        nn.Linear(dim_in, dim_hidden),
        nn.LeakyReLU(0.3),
        nn.LayerNorm(dim_hidden),
        nn.Linear(dim_hidden, dim_hidden),
        nn.LeakyReLU(0.3), 
        nn.LayerNorm(dim_hidden),
        nn.Linear(dim_hidden, dim_hidden)
    )


class ECGUnetCondition(nn.Module):
    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=8,
            resnet_block_groups=8,
            learned_variance=False,
            condition='muse'
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels

        init_dim = default(init_dim, dim // 3 * 2)
        self.init_conv = nn.Conv1d(channels, init_dim, 25, padding='same')

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            TimeStepEmbedding(dim),
            nn.Linear(dim, time_dim),
            nn.LeakyReLU(0.3),
            nn.Linear(time_dim, time_dim)
        )

        
        if condition=='muse':
            self.label_embedding = LinearEncoding(5, time_dim)
        else:
            self.label_embedding = nn.Sequential(
                nn.Embedding(num_class, dim),
                nn.Linear(dim, time_dim),
                nn.LeakyReLU(0.3), 
                nn.Linear(time_dim, time_dim)
            )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_conv = nn.Sequential(
            block_klass(dim, dim),
            nn.Conv1d(dim, self.out_dim, 1)
        )

    def forward(self, x, time, target):
        x = self.init_conv(x)
        t = self.time_mlp(time)
        label = self.label_embedding(target)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t, label)
            x = block2(x, t, label)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t, label)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, label)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t, label)
            x = block2(x, t, label)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)



