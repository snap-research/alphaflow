import torch
import torch.nn as nn

import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

from src.structs import EasyDict, TensorGroup
from typing import Optional
from src.utils import misc

#----------------------------------------------------------------------------

class SiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float=4.0,
        qk_norm: bool=True,
        modulate: bool=True,
        norm_type: bool='layer_norm',
        enable_layernorm_kernel: bool=False,
        mlp_module: nn.Module=Mlp,
        attn_module: nn.Module=Attention,
        **block_kwargs
    ):
        super().__init__()
        norm_kwargs = {'layer_norm': dict(eps=1e-6, use_kernel=enable_layernorm_kernel)}[norm_type]
        self.norm1 = construct_norm_layer(norm_type, hidden_size, affine=not modulate, **norm_kwargs)
        self.attn = attn_module(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=qk_norm
        )
        self.norm2 = construct_norm_layer(norm_type, hidden_size, affine=not modulate, **norm_kwargs)
        self.mlp = mlp_module(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)

        attn_input = modulate(self.norm1, x, shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(attn_input)

        mlp_input = modulate(self.norm2, x, shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(mlp_input)

        return x

#----------------------------------------------------------------------------

class FinalLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        modulate: bool=True,
        norm_type: bool='layer_norm',
        enable_layernorm_kernel=False,
        **block_kwargs
    ):
        super().__init__()
        norm_kwargs = {'layer_norm': dict(eps=1e-6, use_kernel=enable_layernorm_kernel)}[norm_type]
        self.norm_final = construct_norm_layer(norm_type, hidden_size, affine=not modulate, eps=1e-6, use_kernel=enable_layernorm_kernel)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final, x, shift, scale)
        x = self.linear(x)

        return x

#----------------------------------------------------------------------------

class SiT(nn.Module):
    should_fsdp_wrap = True
    def __init__(
        self,
        cfg: EasyDict,                              # Main parameters config.
        input_resolution: tuple[int, int, int],     # Image resolution at input/output.
        in_channels: int,
        out_channels: int,
        label_shape: list[int],                   # Number of class labels, 0 = unconditional.
        **block_kwargs
    ):
        super().__init__()
        self.cfg = cfg
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = self.cfg.tokenizer.resolution[2]
        self.num_heads = self.cfg.num_heads
        self.num_classes = label_shape[0]
        self.num_frames = input_resolution[0]
        self.linear_init_scale = cfg.get("linear_init_scale", 1.0)

        self.x_embedder = PatchEmbed(tuple(input_resolution[1:]), self.patch_size, in_channels, self.cfg.dim, bias=True)
        if input_resolution[0] > 1:
            pos_embed_temporal = get_1d_sincos_pos_embed(self.cfg.dim, input_resolution[0])
            pos_embed_temporal = misc.rearrange(torch.from_numpy(pos_embed_temporal).float(), 't d -> 1 t 1 d') # [1, num_patches_temporal, 1, d]
            self.register_buffer('pos_embed_temporal', pos_embed_temporal) # [1, num_patches_temporal, 1, d]
        else:
            self.pos_embed_temporal = None
        self.noise_labels_embedder = TimestepEmbedder(self.cfg.dim) # timestep embedding type
        self.noise_labels_next_embedder = TimestepEmbedder(self.cfg.dim) # timestep embedding type
        self.y_embedder = LabelEmbedder(self.num_classes, self.cfg.dim, self.cfg.label_dropout)

        self._build_posenc()

        self.blocks = nn.ModuleList([
            SiTBlock(self.cfg.dim, self.num_heads, mlp_ratio=self.cfg.mlp_ratio, qk_norm=False) for _ in range(self.cfg.num_blocks)
        ])
        self.final_layer = FinalLayer(self.cfg.dim, self.patch_size, self.out_channels)
        self.initialize_weights()

    def _build_posenc(self):
        # Initialize pos_embed
        self.pos_embed = nn.Parameter(torch.zeros(1, self.x_embedder.num_patches, self.cfg.dim), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5)
            )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def initialize_weights(self):
        self.apply(lambda m: _basic_init(m, self.linear_init_scale))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        init_embedder_mlp(self.y_embedder.embedding_table)

        initialize_time_mlp_weights(self.noise_labels_embedder)
        initialize_time_mlp_weights(self.noise_labels_next_embedder)

        for block in self.blocks:
            zero_init(block.adaLN_modulation[-1])

        zero_init(self.final_layer.adaLN_modulation[-1])
        zero_init(self.final_layer.linear)

    def unpatchify(self, x, patch_size=None):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0] if patch_size is None else patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward(self,
            x: torch.Tensor,
            noise_labels: torch.Tensor=None,
            cond: Optional[TensorGroup]=None,
            noise_labels_next: torch.Tensor=None,
            return_logvar=False,
            sc_latents: Optional[torch.Tensor]=None,
            augment_labels: Optional[torch.Tensor]=None,
            force_label_dropout: bool=False,
        ):
        _, _, _ = return_logvar, augment_labels, sc_latents # Unused in this implementation
        x = misc.rearrange(x, 'b t c h w -> (b t) c h w') # Merge temporal dimension into height
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        if self.pos_embed_temporal is not None:
            x = misc.rearrange(x, '(b t) p d -> b t p d', t=self.num_frames) # (N, num_frames, num_patches_spatial, D)
            x = x + self.pos_embed_temporal # [b, t, p, d]
            x = misc.rearrange(x, 'b t p d -> b (t p) d') # (N, T, D), where T = num_frames * num_patches_spatial
        N, T, D = x.shape

        # Timestep and class embedding - modified for MeanFlow with r and t
        noise_labels_next = torch.zeros_like(noise_labels) if noise_labels_next is None else noise_labels_next
        assert (noise_labels >= noise_labels_next).any()
        noise_labels_embed = self.noise_labels_embedder(noise_labels.flatten(), x.dtype)   # (N, D)
        noise_labels_next_embed = self.noise_labels_next_embedder(noise_labels_next.flatten(), x.dtype)   # (N, D)
        if cond is None or force_label_dropout:
            label = torch.ones(N, dtype=torch.long, device=x.device) * self.y_embedder.embedding_table.num_embeddings - 1
        else:
            # If cond.label is all zeros for a sample, assign self.embedding_table.num_embeddings - 1 as label for that sample
            uncond_idx = cond.label.sum(dim=1) == 0
            label = torch.argmax(cond.label, dim=1).long()
            label[uncond_idx] = self.y_embedder.embedding_table.num_embeddings - 1
        y_embed = self.y_embedder(label)  # (N, D)
        c = noise_labels_embed + noise_labels_next_embed + y_embed  # (N, D)

        for block in self.blocks:
            x = block(x, c)  # (N, T, D)
        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        data_tokens_before_detok = x
        x = misc.rearrange(x, 'b (t p) d -> (b t) p d', t=self.num_frames)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        x = misc.rearrange(x, '(b t) c h w -> b t c h w', t=self.num_frames)
        return x, EasyDict(sc_latents=data_tokens_before_detok, align_features=None)

# Initializer functions.
#----------------------------------------------------------------------------

def _basic_init(module, linear_init_scale: float):
    if isinstance(module, torch.nn.Linear):
        if module.weight.requires_grad_:
            torch.nn.init.xavier_uniform_(module.weight)
            if linear_init_scale != 1.0:
                module.weight *= linear_init_scale
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)

def initialize_time_mlp_weights(time_mlp: nn.Module):
    if time_mlp is None: return
    if hasattr(time_mlp, "mlp"):
        time_mlp.mlp[0].apply(init_embedder_mlp)
        time_mlp.mlp[2].apply(init_embedder_mlp)

def init_embedder_mlp(module: nn.Module):
    torch.nn.init.normal_(module.weight, std=0.02)

def zero_init(module: nn.Module):
    torch.nn.init.constant_(module.weight, 0)
    torch.nn.init.constant_(module.bias, 0)

#----------------------------------------------------------------------------
# Sine/Cosine Positional Embedding Functions from DiT/MAE
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(
    embed_dim,
    grid_size,
    cls_token=False,
    extra_tokens=0,
    scale=1.0,
    base_size=None,
):
    """grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    grid_h = np.arange(grid_size[0], dtype=np.float32) / scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / scale
    if base_size is not None:
        grid_h *= base_size / grid_size[0]
        grid_w *= base_size / grid_size[1]
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed],
            axis=0,
        )
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2,
        grid[0],
    )  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2,
        grid[1],
    )  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb

def get_1d_sincos_pos_embed(embed_dim, length, scale=1.0):
    pos = np.arange(0, length)[..., None] / scale
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

#----------------------------------------------------------------------------

def modulate(norm_func, x, shift: torch.Tensor | None, scale: torch.Tensor | None):
    # Assumes that x is (B, N, D), shift/scale are (B, D) or (B, 1 | N, D)
    # Runs the normalization layer in full precision. This should happen already for torch.nn.LayerNorm
    dtype = x.dtype
    with torch.autocast("cuda", enabled=False):
        x = norm_func(x.to(torch.float32))

    if scale is not None:
        assert shift is not None, "Shift must be provided if scale is provided"
        scale, shift = [s.unsqueeze(1) if s.ndim == 2 else s for s in (scale, shift)]
        x = x * (scale + 1) + shift
    return x.to(dtype)

#----------------------------------------------------------------------------

def construct_norm_layer(norm_type: Optional[str], num_channels: int, **kwargs) -> nn.Module:
    if norm_type is None or norm_type == 'none':
        return nn.Identity()
    elif norm_type == 'layer_norm':
        return get_layernorm(num_channels, **kwargs)
    else:
        raise ValueError(f"Invalid normalization type: {norm_type}")

def get_layernorm(hidden_size: torch.Tensor, eps: float, affine: bool, use_kernel: bool):
    if use_kernel:
        try:
            import apex
            from apex.normalization import FusedLayerNorm
            return FusedLayerNorm(hidden_size, eps=eps, elementwise_affine=affine)
        except ImportError:
            raise RuntimeError("FusedLayerNorm not available. Please install Apex with fused operations support.")
    else:
        return torch.nn.LayerNorm(hidden_size, eps, elementwise_affine=affine)

approx_gelu = lambda: torch.nn.GELU(approximate="tanh")

#----------------------------------------------------------------------------

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        freqs = freqs.to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2 != 0:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, dtype):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if t_freq.dtype != dtype:
            t_freq = t_freq.to(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb

#----------------------------------------------------------------------------

class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = int(dropout_prob > 0)
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels, force_drop_ids=None):
        if force_drop_ids is not None:
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

#----------------------------------------------------------------------------
