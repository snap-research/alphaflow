import math
from typing import Optional, Any, Callable

import numpy as np
import torch
import einops

from src.structs import EasyDict, TensorGroup
from src.training.network_utils import load_snapshot
from src.utils import misc
from src.utils.os_utils import construct_class_by_name
from src.structs import DTYPE_STRING_TO_DTYPE, aggregate_variance, TensorLike

#----------------------------------------------------------------------------
# Fourier embedding classes.

class FourierEmbedding(torch.nn.Module):
    # Timestep embedding used in the NCSN++ architecture.
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', misc.randn42(num_channels // 2) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype)) # [b, d // 2]
        x = torch.cat([x.cos(), x.sin()], dim=1) # [b, d]
        return x


class MPFourier(torch.nn.Module):
    # An improved fourier embedding from EDMv2.
    def __init__(self, num_channels: int, bandwidth=1):
        super().__init__()
        self.register_buffer('freqs', 2 * np.pi * misc.randn42(num_channels) * bandwidth)
        self.register_buffer('phases', 2 * np.pi * misc.rand42(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.to(torch.float32) # [b]
        y = y.ger(self.freqs.to(torch.float32)) # [b, d]
        y = y + self.phases.to(torch.float32) # [b, d]
        y = y.cos() * np.sqrt(2) # [b, d]
        return y.to(x.dtype) # [b, d]


class LogVarNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fourier = MPFourier(128)
        self.linear = torch.nn.Linear(128, 1, bias=False)

    def _forward(self, noise_labels: torch.Tensor) -> torch.Tensor:
        shape = noise_labels.shape
        misc.rearrange(noise_labels, 'b t 1 h w -> (b t h w) 1') # [b * t * h * w, 1]
        logvar = self.linear(self.fourier(noise_labels)) # [b * t * h * w, 1]
        logvar = misc.rearrange(logvar, '(b t h w) 1 -> b t 1 h w', b=shape[0], t=shape[1], h=shape[3], w=shape[4]) # [b, t, 1, h, w]

        return logvar

    def forward(self, noise_labels: TensorLike) -> TensorLike:
        if isinstance(noise_labels, TensorGroup):
            return TensorGroup({k: self._forward(noise_labels.k) for k in noise_labels.keys()})
        else:
            return self._forward(noise_labels)

#----------------------------------------------------------------------------

class BaseNetwork(torch.nn.Module):
    should_fsdp_wrap = True

    def __init__(self,                         # pylint: disable=dangerous-default-value
        resolution: tuple[int, int, int],      # Video resolution.
        img_channels,                          # Number of color channels.
        label_shape          = 0,              # Number of class labels, 0 = unconditional.
        cfg: EasyDict        = {},             # Model config.
        augment_dim: int     = 0,              # Augmentation label dimensionality.
    ):
        super().__init__()
        assert cfg.act_dtype in DTYPE_STRING_TO_DTYPE, f"Invalid dtype {cfg.act_dtype}"
        self.cfg = cfg
        self.input_resolution = resolution
        self.in_channels = img_channels
        self.out_channels = img_channels
        self.input_shape = (self.input_resolution[0], self.in_channels, self.input_resolution[1], self.input_resolution[2])
        self.label_shape = label_shape

        self._pre_model_init_hook()
        self.logvar_net = LogVarNet() if self.cfg.predict_logvar else None
        self.model = construct_class_by_name(
            class_name=cfg.backbone_target_class, cfg=cfg, input_resolution=self.input_resolution, in_channels=self.in_channels,
            out_channels=self.out_channels, label_shape=label_shape, augment_dim=augment_dim)
        self._post_model_init_hook()
        self._maybe_freeze_modules()

    def _maybe_freeze_modules(self):
        for module_name in (self.cfg.freeze_modules_list or []):
            misc.replace_parameters_with_buffers_(misc.find_submodule(self, module_name))

    def _pre_model_init_hook(self):
        pass

    def _post_model_init_hook(self):
        pass

    def _decide_dtype(self, force_fp32: bool=False) -> Optional[torch.dtype]:
        if force_fp32 or self.cfg.act_dtype == 'float32':
            return torch.float32
        elif self.cfg.autocast:
            return None
        else:
            return DTYPE_STRING_TO_DTYPE[self.cfg.act_dtype]

    def construct_dummy_cond(self, batch_size: int, device: torch.device='cpu') -> TensorGroup:
        return TensorGroup(
            label=torch.zeros(batch_size, *self.label_shape, device=device), # [b, *label_shape]
            framerate=torch.ones(batch_size, device=device) * 6.0, # [b]
            dataset_id=torch.zeros(batch_size, device=device, dtype=torch.int64), # [b]
            resolution=torch.as_tensor([0.5625, 1.0], dtype=torch.float32, device=device).unsqueeze(0).repeat(batch_size, 1), # [b, 2]
            is_video=torch.ones(batch_size, device=device, dtype=torch.int64), # [b]
            is_vertical=torch.zeros(batch_size, device=device, dtype=torch.int64), # [b]
        )

    def construct_cond_from_labels(self, labels: torch.Tensor) -> TensorGroup:
        dummy_cond = self.construct_dummy_cond(len(labels), labels.device)
        dummy_cond.label = labels
        return dummy_cond

    def construct_dummy_input(self, batch_size: int, device: torch.device='cpu', input_shape: tuple[int, int, int, int]=None) -> tuple[torch.Tensor, torch.Tensor, TensorGroup]:
        input_shape = input_shape if input_shape is not None else self.input_shape # [4]
        x_noised = torch.randn(batch_size, *input_shape, device=device) # [b, t, c, h, w]
        sigma = torch.ones(batch_size, device=device).float() * (self.cfg.sampling.get('sigma_min', 0.0) + 1e-16) # [b]
        cond = self.construct_dummy_cond(batch_size, device)

        return x_noised, sigma, cond

    def _forward(self, *args, **kwargs):
        raise NotImplementedError('The _forward method must be implemented in a derived class.')

    def forward(self, *args, **kwargs):
        with misc.maybe_autocast(self.cfg):
            return self._forward(*args, **kwargs)

    @staticmethod
    def get_autowrap_policy(min_num_params: int) -> Callable:
        def auto_wrap_policy(module: torch.nn.Module, recurse: bool, **kwargs) -> bool:
            # For some reason, pytorch decided to combine two different functions (for recurse=True and recurse=False) into one.
            if recurse:
                # If recurse=True, we should return a bool specifying if the traversal should continue.
                return True

            # If recurse=False, we should return a bool specifying if the module should be wrapped.
            if hasattr(module, 'should_fsdp_wrap'):
                return module.should_fsdp_wrap

            # These instances receive tensors wrapped in custom data types, thus wrapping cannot be applied directly to them
            # if isinstance(module, (DhariwalUNet, FIT)) and not recurse:
            if isinstance(module, (Diffusion,)):
                return False

            # ModuleDict and ModuleList's inner modules are called directly not throught forward. If these modules were wrapped, internal module
            # calls would raise exceptions since parameters would not be gathered from the other nodes as part of the forward hooks
            is_class_called_through_forward = not isinstance(module, (torch.nn.ModuleDict, torch.nn.ModuleList))
            num_nonwrapped_params = kwargs["unwrapped_params"] if "unwrapped_params" in kwargs else kwargs["nonwrapped_numel"]
            has_many_params = num_nonwrapped_params >= min_num_params
            return has_many_params and is_class_called_through_forward # Do not stop wrapping because you encountered a module list or dict

        return auto_wrap_policy

#----------------------------------------------------------------------------

class Diffusion(BaseNetwork):
    def _get_scaling_factors(self, sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.cfg.precond == 'edm':
            c_skip = self.cfg.sigma_data ** 2 / (sigma ** 2 + self.cfg.sigma_data ** 2) # [b, ...]
            c_out = sigma * self.cfg.sigma_data / (sigma ** 2 + self.cfg.sigma_data ** 2).sqrt() # [b, ...]
            c_in = 1 / (self.cfg.sigma_data ** 2 + sigma ** 2).sqrt() # [b, ...]
            noise_labels = sigma.log() / 4 # [b, ...]
        elif self.cfg.precond == 'xpred':
            c_skip = None
            c_out = misc.ones_like(sigma) # [b, ...]
            c_in = 1 / (self.cfg.sigma_data ** 2 + sigma ** 2).sqrt() # [b, ...]
            noise_labels = sigma.log() / 4 # [b, ...]
        elif self.cfg.precond == 'recflow':
            c_skip = None
            c_out = misc.empty_like(sigma).fill_(math.sqrt(self.cfg.sampling.sigma_noise ** 2 + self.cfg.sigma_data ** 2)) # [b, ...]
            c_in = 1.0 / ((self.cfg.sigma_data ** 2) * (sigma.pow(2) - 2 * sigma + 1) + sigma.pow(2) * (self.cfg.sampling.sigma_noise ** 2)).sqrt() # [b, ...]
            noise_labels = sigma # [b, ...]
        elif self.cfg.precond == 'alphaflow':
            c_skip = None
            c_out = torch.ones_like(sigma) # [b, ...]
            c_in = torch.ones_like(sigma) # [b, ...]
            noise_labels = sigma # [b, ...]
        else:
            raise NotImplementedError(f"Unknown preconditioning {self.cfg.precond}")

        return c_skip, c_out, c_in, noise_labels

    def _construct_input(self, x_noised: torch.Tensor | None, sigma: torch.Tensor | None, sigma_next: torch.Tensor | None, force_fp32: bool=False):
        force_dtype = self._decide_dtype(force_fp32) # torch.dtype or None
        if sigma is not None:
            c_skip, c_out, c_in, noise_labels = self._get_scaling_factors(sigma) # [b, ...], [b, ...], [b, ...], [b, ...]
            noise_labels = noise_labels # [b, ...]. `noise_labels` is the only scaling factor which does not need to be broadcasted.
        else:
            c_skip = c_in = noise_labels = None
            c_out = 1.0 # [1]

        if sigma_next is not None and sigma is not None:
            _, _, _, noise_labels_next = self._get_scaling_factors(sigma_next)
            noise_labels_next = noise_labels - noise_labels_next # [b, ...].
        else:
            noise_labels_next = None # [b, ...].

        if x_noised is not None:
            assert len(x_noised) == len(sigma), f'x_noised.shape={x_noised.shape}, sigma.shape={sigma.shape}. Batch sizes should match.'
            x_noised = x_noised.to(torch.float32) # [b, ...]
            x_input = (x_noised * c_in).to(force_dtype) # [b, ...]
        else:
            x_input = None

        return x_input, noise_labels, noise_labels_next, c_skip, c_out, force_dtype

    def _forward(self, x_noised, sigma=None, cond=None, sigma_next=None, force_fp32=False, return_extra_output: bool=False, **model_kwargs) -> tuple[torch.Tensor, Any]:
        # Compute input and output.
        sigma = torch.ones(len(x_noised), device=x_noised.device).fill_(self.cfg.sampling.get('sigma_min', 0.0) + 1e-16) if sigma is None and x_noised is not None else sigma # [b] or None
        sigma = sigma.to(torch.float32) if sigma is not None else None # [b, ...] or None
        sigma = sigma.reshape(len(sigma), 1, 1, 1, 1) if sigma is not None and sigma.ndim == 1 else sigma # [b, ...]
        sigma_next = sigma_next.to(torch.float32).reshape(len(sigma_next), 1, 1, 1, 1) if sigma_next is not None else None # [b, 1, 1, 1, 1] or None
        x_input, noise_labels, noise_labels_next, c_skip, c_out, force_dtype = self._construct_input(x_noised, sigma, sigma_next, force_fp32) # [b, l, c, h, w], [b, ...], [b, ...], [b, ...], [b, ...], torch.dtype or None
        model_kwargs.update({"noise_labels_next": noise_labels_next} if noise_labels_next is not None else {})
        output, extra_output = self.model(x_input, noise_labels=noise_labels, cond=cond, **model_kwargs)

        # Post-process the output.
        assert force_dtype is None or output is None or output.dtype == force_dtype, f"Expected the force_dtype={force_dtype} as input, got output dtype={output.dtype}"
        output = output.to(torch.float32) if output is not None else None # [b, f, c, h, w]
        if self.cfg.reparam_recflow_from_xpred and output is not None:
            output = (x_noised - (1.0 - sigma) * output) / sigma - output # [b, f, c, h, w]
            output = output / c_out # [b, f, c, h, w]. Divide by c_out to compensate for future scaling.
        x_skip = (c_skip * x_noised) if misc.all_not_none([x_noised, c_skip]) else None # [b, l, c, h, w] or None
        diffusion_output = (c_out * output + (0.0 if x_skip is None else x_skip)) if output is not None else None # [b, l, c, h, w]

        # Post-process the extra "metadata-kind-of" output.
        if extra_output is not None:
            assert isinstance(extra_output, dict), f"Expected extra_output to be a dict, got {type(extra_output)}"
            extra_output['logvar'] = self.logvar_net(noise_labels) if self.logvar_net is not None else None # [b, 1, 1, 1, 1] or None
            # DDP/FSDP fix (even FSDP fails with "RuntimeError: Cannot writeback when the parameter shape changes" when there are unused parameters).
            diffusion_output = (diffusion_output + 0.0 * extra_output['logvar'].max()) if diffusion_output is not None and extra_output['logvar'] is not None else diffusion_output
        video_shape = None if diffusion_output is None and x_noised is None else (diffusion_output.shape if diffusion_output is not None else x_noised.shape)
        if self.cfg.is_autoencoder and all([_ is not None for _ in (video_shape, extra_output, extra_output.get('bits'))]):
            extra_output['bpp'] = extra_output['bits'] / (video_shape[1] * video_shape[3] * video_shape[4]) # [b]
        return (diffusion_output, extra_output) if return_extra_output else diffusion_output

#----------------------------------------------------------------------------

class DiffusionAutoEncoder(Diffusion):
    @property
    def compression_rate(self) -> tuple[int]:
        return [rgb_res // latent_res for rgb_res, latent_res in zip(self.input_resolution, self.model.unchunked_latent_resolution)]

    def get_encoder_parameters(self):
        return self.model.get_encoder_parameters()

    def construct_dummy_input(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor, TensorGroup]:
        x_noised, sigma, cond = super().construct_dummy_input(*args, **kwargs)
        cond.x_cond = x_noised # [b, t, c, h, w]
        return x_noised, sigma, cond

    def _forward(self, x_noised: torch.Tensor=None, sigma=None, cond: TensorGroup=None, encode: bool=True, decode: bool=True, latents=None, return_extra_output=False,
                 enable_latents_grad: bool=False, **kwargs) -> tuple[torch.Tensor, Any]: # pylint: disable=arguments-differ)
        if encode:
            assert latents is None, f"Expected `latents` to be None when encoding, got {type(latents)}"
            _, ctx = super()._forward(x_noised, sigma=sigma, cond=cond, encode=True, decode=False, return_extra_output=True, **kwargs)
            latents = ctx['latents'] # [b, tl, cl, hl, wl]
        else:
            assert latents is not None, "Expected `latents` to be not None when decoding"
            ctx = None

        if self.cfg.latent_noise_std > 0.0 and self.training:
            latents = latents + self.cfg.latent_noise_std * torch.randn_like(latents) # [b, lt, c, lh, lw]

        if enable_latents_grad:
            latents = latents.requires_grad_(True) # [b, lt, c, lh, lw]

        if decode:
            assert latents is not None, "Expected `latents` to be not None when decoding"
            pixels = super()._forward(x_noised, sigma=sigma, cond=cond, encode=False, decode=True, return_extra_output=False, latents=latents, **kwargs)
        else:
            pixels = None

        return (pixels, ctx) if return_extra_output else pixels

#----------------------------------------------------------------------------

class LatentDiffusion(Diffusion):
    def __init__(self,                         # pylint: disable=dangerous-default-value, super-init-not-called
        resolution: tuple[int, int, int],      # Video resolution.
        img_channels,                          # Number of color channels.
        label_shape          = 0,              # Number of class labels, 0 = unconditional.
        cfg: EasyDict        = {},             # Model config.
        augment_dim: int     = 0,              # Augmentation label dimensionality.
    ):
        torch.nn.Module.__init__(self) # pylint: disable=non-parent-init-called
        assert cfg.act_dtype in DTYPE_STRING_TO_DTYPE, f"Invalid dtype {cfg.act_dtype}"
        assert augment_dim == 0, f"For LDMs, expected augment_dim to be 0, got {augment_dim}"
        _ = img_channels, resolution # Unused, kept for compatibility with the parent class.
        self.cfg = cfg
        self._init_autoencoder()
        self.input_resolution = self.autoencoder.model.unchunked_latent_resolution
        self.in_channels = self.autoencoder.model.latent_channels
        self.out_channels = self.autoencoder.model.latent_channels
        self.input_shape = (self.input_resolution[0], self.in_channels, self.input_resolution[1], self.input_resolution[2])
        self.label_shape = label_shape
        self.logvar_net = LogVarNet() if self.cfg.predict_logvar else None
        assert self.cfg.latent_channel_break is None or len(self.cfg.latent_channel_break) == 3 and self.in_channels % np.prod(self.cfg.latent_channel_break) == 0, \
            f"Invalid latent_channel_break {self.cfg.latent_channel_break} for in_channels {self.in_channels}"
        backbone_input_resolution = self.input_resolution if self.cfg.latent_channel_break is None else tuple([r * b for r, b in zip(self.input_resolution, self.cfg.latent_channel_break)])
        backbone_out_channels = self.in_channels if self.cfg.latent_channel_break is None else (self.in_channels // np.prod(self.cfg.latent_channel_break)).item()
        self.model = construct_class_by_name(
            class_name=cfg.backbone_target_class, cfg=cfg, input_resolution=backbone_input_resolution,
            in_channels=backbone_out_channels, out_channels=backbone_out_channels, label_shape=label_shape)

        stats_shape = (self.input_resolution[0], self.in_channels, self.input_resolution[1], self.input_resolution[2]) if self.cfg.spatio_temporal_latent_stats else (1, self.in_channels, 1, 1) # [4]
        self.register_buffer('autoencoder_mean', torch.zeros(*stats_shape).double())
        self.register_buffer('autoencoder_var', torch.ones(*stats_shape).double())
        self.register_buffer('are_ldm_stats_inited', torch.tensor(False).float())

    def _init_autoencoder(self):
        self.autoencoder = load_snapshot(self.cfg.autoencoder_ckpt, verbose=False)[0].eval().requires_grad_(False)
        misc.set_fsdp_wrap_recursively_(self.autoencoder, False) # Entirely disable FSDP wrapping for the autoencoder.
        if self.cfg.autoencoder_ckpt.convert_params_to_buffers:
            self.autoencoder = misc.replace_parameters_with_buffers_(self.autoencoder)

    def set_autoencoder_stats(self, mean: torch.Tensor, var: torch.Tensor, num_samples_used: int):
        if not self.cfg.spatio_temporal_latent_stats and mean.numel() > self.in_channels:
            t, h, w = self.input_resolution
            mean, var = [einops.rearrange(s, '(t c h w) -> (t h w) c', c=self.in_channels, t=t, h=h, w=w) for s in (mean, var)] # [t * h * w, c], [t * h * w, c]
            var = aggregate_variance(mean, var, num_samples_used) # [c]
            mean = mean.mean(dim=0) # [c]
        self.autoencoder_mean.data = mean.reshape(self.autoencoder_mean.shape).to(device=self.autoencoder_mean.device, dtype=self.autoencoder_mean.dtype)
        self.autoencoder_var.data = var.reshape(self.autoencoder_var.shape).to(device=self.autoencoder_var.device, dtype=self.autoencoder_mean.dtype)
        self.are_ldm_stats_inited.data.fill_(1.0)

    def normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return ((latents - self.autoencoder_mean.unsqueeze(0)) / (self.autoencoder_var.unsqueeze(0) + 1e-16).sqrt()).float() # [b, t, c, h, w]

    def denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return (latents * (self.autoencoder_var.unsqueeze(0) + 1e-16).sqrt() + self.autoencoder_mean.unsqueeze(0)).float() # [b, t, c, h, w]

    @torch.no_grad()
    def encode(self, x: torch.Tensor, cond: Optional[TensorGroup]=None, only_normalize: bool=False) -> torch.Tensor:
        if only_normalize:
            latents = x # [b, lt, lc, lh, lw]
        else:
            latents = self.autoencoder(x, sigma=None, cond=cond, return_extra_output=True, encode=True, decode=False, force_label_dropout=self.cfg.autoencoder_ignore_labels)[1].latents # [b, lt, lc, lh, lw]
        return self.normalize_latents(latents) # [b, lt, lc, lh, lw]

    @torch.no_grad()
    def decode(self, latents: torch.Tensor, cond: Optional[TensorGroup]=None) -> torch.Tensor:
        latents = self.denormalize_latents(latents) # [b, lt, lc, lh, lw]
        return self.autoencoder(None, sigma=None, cond=cond, latents=latents, encode=False, decode=True, force_label_dropout=self.cfg.autoencoder_ignore_labels)

    def maybe_break_latents(self, latents: torch.Tensor):
        assert latents.ndim == 5, f"Expected latents to be 5D tensor of [b, t, c, h, w] shape, got {latents.shape}."
        if self.cfg.latent_channel_break is None:
            return latents
        return misc.rearrange(latents, 'b t (c ct ch cw) h w -> b (t ct) c (h ch) (w cw)', ct=self.cfg.latent_channel_break[0], ch=self.cfg.latent_channel_break[1], cw=self.cfg.latent_channel_break[2])

    def maybe_unbreak_latents(self, latents: torch.Tensor):
        assert latents.ndim == 5, f"Expected latents to be 5D tensor of [b, t, c, h, w] shape, got {latents.shape}."
        if self.cfg.latent_channel_break is None:
            return latents
        return misc.rearrange(latents, 'b (t ct) c (h ch) (w cw) -> b t (c ct ch cw) h w', ct=self.cfg.latent_channel_break[0], ch=self.cfg.latent_channel_break[1], cw=self.cfg.latent_channel_break[2])

    def _forward(self, x_noised, *args, return_extra_output: bool=False, **kwargs) -> tuple[torch.Tensor, Any]:
        latents_noised = self.maybe_break_latents(x_noised) # [b, (t ct) c (h ch) (w cw)]
        out = super()._forward(latents_noised, *args, return_extra_output=return_extra_output, **kwargs)
        latents_denoised = self.maybe_unbreak_latents(out[0] if return_extra_output else out) # [b, t, c, h, w]
        return (latents_denoised, out[1]) if return_extra_output else latents_denoised

#----------------------------------------------------------------------------

class MultiModalDiffusion(Diffusion):
    """Multi-modal diffusion for joint audio-video, video-to-audio and audio-to-video modeling."""
    def __init__(self,                         # pylint: disable=dangerous-default-value
        resolution: tuple[int, int, int],      # Video resolution.
        img_channels,                          # Number of color channels.
        label_shape           = 0,              # Number of class labels, 0 = unconditional.
        cfg: EasyDict         = {},             # Model config.
        augment_dim: int      = 0,              # Augmentation label dimensionality.
        audio_len: int | None = None,           # Audio waveform length.
        audio_channels: int   = 1,              # Number of audio channels.
    ):
        torch.nn.Module.__init__(self) # pylint: disable=non-parent-init-called
        assert cfg.act_dtype in DTYPE_STRING_TO_DTYPE, f"Invalid dtype {cfg.act_dtype}"
        _ = img_channels # Unused, kept for compatibility with the parent class.
        self.cfg = cfg
        self.input_resolution = resolution
        self._video_resolution = resolution
        self._audio_len = audio_len
        self._audio_channels = audio_channels
        self.in_channels = img_channels
        self.out_channels = img_channels
        self.input_shape = EasyDict({
            'video': (self._video_resolution[0], self.in_channels, self._video_resolution[1], self._video_resolution[2]), # [t, c, h, w]
            'audio': (self._audio_len, self._audio_channels, 1, 1),
        })
        self.label_shape = label_shape

        self._pre_model_init_hook()
        self.logvar_net = LogVarNet() if self.cfg.predict_logvar else None
        self.model = construct_class_by_name(
            class_name=cfg.backbone_target_class,
            cfg=cfg,
            video_input_resolution=self.input_resolution,
            video_in_channels=self.in_channels,
            video_out_channels=self.out_channels,
            audio_len=self._audio_len,
            audio_channels=self._audio_channels,
            label_shape=label_shape,
            augment_dim=augment_dim,
        )
        self._post_model_init_hook()
        self._maybe_freeze_modules()

    def construct_dummy_input(self, batch_size: int, device: torch.device='cpu', input_shape: tuple[int, int, int, int]=None) -> tuple[TensorGroup, TensorGroup, TensorGroup]:
        input_shape = input_shape if input_shape is not None else self.input_shape
        video_noised = torch.zeros(batch_size, *input_shape.video, device=device) # [b, t, c, h, w]
        audio_noised = torch.randn(batch_size, *input_shape.audio, device=device) if self._audio_len is not None else None # [b, ta, ca, 1, 1]
        sigma_video = sigma_audio = torch.ones(batch_size, 1, 1, 1, 1, device=device).float() * (self.cfg.sampling.get('sigma_min', 0.0) + 1e-16) # [b]
        cond = self.construct_dummy_cond(batch_size, device)

        x_noised = TensorGroup({'video': video_noised, 'audio': audio_noised})
        sigma = TensorGroup({'video': sigma_video, 'audio': sigma_audio})

        return x_noised, sigma, cond

    def _forward(self, x_noised: TensorLike, sigma: TensorLike, cond: TensorGroup=None, force_fp32=False, return_extra_output: bool=False, **model_kwargs) -> tuple[torch.Tensor, Any]:
        sigma = sigma.float() # [b, ...]
        x_noised_input, noise_labels, c_skip, c_out, force_dtype = self._construct_input(x_noised, sigma, force_fp32) # [b, l, c, h, w], [b], torch.dtype or None
        output, extra_output = self.model(x_noised_input, noise_labels, cond=cond, **model_kwargs)

        # Post-process the output.
        assert force_dtype is None or output is None or output.dtype == force_dtype, f"Expected the force_dtype={force_dtype} as input, got output dtype={output.dtype}"
        output = TensorGroup(output).to(torch.float32) # [b, f, c, h, w]
        x_skip = (c_skip * x_noised) if misc.all_not_none([x_noised, c_skip]) else None # [b, l, c, h, w] or None
        diffusion_output = (c_out * output + (0.0 if x_skip is None else x_skip)) if output is not None else None # [b, l, c, h, w]

        # Post-process the extra "metadata-kind-of" output.
        if extra_output is not None:
            assert isinstance(extra_output, dict), f"Expected extra_output to be a dict, got {type(extra_output)}"
            extra_output['logvar'] = self.logvar_net(noise_labels) if self.logvar_net is not None else None # [b, 1, 1, 1, 1] or None
            # DDP/FSDP fix (even FSDP fails with "RuntimeError: Cannot writeback when the parameter shape changes" when there are unused parameters).
            diffusion_output = (diffusion_output + 0.0 * extra_output['logvar'].max()) if diffusion_output is not None and extra_output['logvar'] is not None else diffusion_output

        return (diffusion_output, extra_output) if return_extra_output else diffusion_output

#----------------------------------------------------------------------------
