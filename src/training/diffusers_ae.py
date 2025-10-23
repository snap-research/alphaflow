import math
from typing import  Optional

import torch
import einops
from diffusers import AutoencoderKLTemporalDecoder, AutoencoderKLCogVideoX, AutoencoderKL
from diffusers.configuration_utils import FrozenDict

from src.structs import ResumeConfig, EasyDict, TensorGroup
from src.utils.autoencoder_utils import vae_latents_forward
from src.utils import misc

#----------------------------------------------------------------------------

def init_diffusers_ae(
        ae_name: str,
        model_variant: str | None=None,
        model_path: str | None=None,
        subfolder: str | None=None,
        **reinit_kwargs,
    ) -> tuple[torch.nn.Module, int, tuple[int, int, int], int | None]:

    vocab_size = None

    if ae_name == 'svd_ae':
        model_class = AutoencoderKLTemporalDecoder
        model = model_class.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", subfolder="vae", torch_dtype=torch.float16, variant="fp16")
        param_dtype = torch.float16
        model = maybe_reinit_diffusers_ae(model, model_class, param_dtype=param_dtype, **reinit_kwargs)
        latent_channels = model.config.latent_channels
        compression_rate = [1, 8, 8]
    elif ae_name == 'sd2_ae':
        model_class = AutoencoderKL
        model_kwargs = dict() if model_variant == 'fp32' else dict(torch_dtype=torch.float16, variant="fp16")
        model = model_class.from_pretrained("stabilityai/stable-diffusion-2", subfolder="vae", **model_kwargs)
        param_dtype = None if model_variant == 'fp32' else torch.float16
        model = maybe_reinit_diffusers_ae(model, model_class, param_dtype=param_dtype, **reinit_kwargs)
        latent_channels = model.config.latent_channels
        compression_rate = [1, 8, 8]
    elif ae_name == 'sdxl_ae':
        model_class = AutoencoderKL
        model = model_class.from_pretrained("stabilityai/sdxl-vae")
        model = maybe_reinit_diffusers_ae(model, model_class, **reinit_kwargs)
        latent_channels = model.config.latent_channels
        compression_rate = [1, 8, 8]
    elif ae_name == 'sd3m_ae':
        model_class = AutoencoderKL
        model_kwargs = dict() if model_variant == 'fp32' else dict(torch_dtype=torch.float16, variant="fp16")
        model = model_class.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", subfolder="vae", **model_kwargs)
        param_dtype = None if model_variant == 'fp32' else torch.float16
        model = maybe_reinit_diffusers_ae(model, model_class, param_dtype=param_dtype, **reinit_kwargs)
        latent_channels = model.config.latent_channels
        compression_rate = [1, 8, 8]
    elif ae_name == 'sd_vae_ft_ema':
        model_class = AutoencoderKL
        model = model_class.from_pretrained("stabilityai/sd-vae-ft-ema")
        param_dtype = None
        model = maybe_reinit_diffusers_ae(model, model_class, param_dtype=param_dtype, **reinit_kwargs)
        latent_channels = model.config.latent_channels
        compression_rate = [1, 8, 8]
    elif ae_name == 'sd-vae-ft-mse':
        model_class = AutoencoderKL
        model = model_class.from_pretrained(f"stabilityai/sd-vae-ft-mse")
        param_dtype = None
        model = maybe_reinit_diffusers_ae(model, model_class, param_dtype=param_dtype, **reinit_kwargs)
        latent_channels = model.config.latent_channels
        compression_rate = [1, 8, 8]
    elif ae_name == 'cogvideo_ae':
        model_class = AutoencoderKLCogVideoX
        model = model_class.from_pretrained("THUDM/CogVideoX-5b", subfolder="vae")
        model = maybe_reinit_diffusers_ae(model, model_class, **reinit_kwargs)
        latent_channels = model.config.latent_channels
        compression_rate = [4, 8, 8]
    elif ae_name == 'ltx_ae':
        from diffusers import AutoencoderKLLTXVideo # pylint: disable=import-outside-toplevel
        model_class = AutoencoderKLLTXVideo
        model = model_class.from_pretrained("Lightricks/LTX-Video", subfolder="vae", torch_dtype=torch.float32)
        model = maybe_reinit_diffusers_ae(model, model_class, **reinit_kwargs)
        latent_channels = 128
        compression_rate = [8, 32, 32]
    elif ae_name == 'flux_ae':
        model_class = AutoencoderKL
        model = model_class.from_pretrained("black-forest-labs/FLUX.1-schnell", subfolder="vae")
        model = maybe_reinit_diffusers_ae(model, model_class, **reinit_kwargs)
        latent_channels = 16
        compression_rate = [1, 8, 8]
    elif ae_name == 'wan_ae':
        from diffusers import AutoencoderKLWan # pylint: disable=import-outside-toplevel
        model_class = AutoencoderKLWan
        model_path = 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers' if model_path is None else model_path
        subfolder = 'vae' if subfolder is None else subfolder
        model = model_class.from_pretrained(model_path, subfolder=subfolder, torch_dtype=torch.float32)
        model = maybe_reinit_diffusers_ae(model, model_class, **reinit_kwargs)
        spatial_compression_rate = 2 ** (len(model.config.dim_mult) - 1)
        compression_rate = [2 ** sum(model.config.temperal_downsample), spatial_compression_rate, spatial_compression_rate]
        latent_channels = model.config.z_dim
    elif ae_name == 'allegro_ae':
        from diffusers import AutoencoderKLAllegro
        model_class = AutoencoderKLAllegro
        model = model_class.from_pretrained("rhymes-ai/Allegro", subfolder="vae", torch_dtype=torch.float32).to("cuda")
        model = maybe_reinit_diffusers_ae(model, model_class, param_dtype=torch.float32, **reinit_kwargs)
        model.enable_tiling()
        compression_rate =param_dtype,  [4, 8, 8]
        latent_channels = model.config.latent_channels
    else:
        raise ValueError(f"Unknown AE name: {ae_name}")

    return model, latent_channels, tuple(compression_rate), vocab_size


def maybe_reinit_diffusers_ae(model_orig: torch.nn.Module, model_class, pretrained: bool, param_dtype=None, config_overrides: dict | None=None, pretrained_lenient: bool=False, pretrained_lenient_resume_kwargs: dict=None) -> torch.nn.Module:
    if pretrained:
        assert config_overrides is None, f"config_overrides is only supported for non-pretrained models, but got {config_overrides}. Use `pretrained_lenient` instead."
        model = model_orig
    else:
        config = FrozenDict({**model_orig.config, **config_overrides}) if config_overrides is not None else model_orig.config
        model = model_class.from_config(config).to(param_dtype)
        if pretrained_lenient:
            misc.load_state_dict_(model, model_orig.state_dict(), resume_cfg=ResumeConfig(**pretrained_lenient_resume_kwargs))

    return model

#----------------------------------------------------------------------------

class DiffusersAutoEncoderBackbone(torch.nn.Module):
    should_fsdp_wrap = True
    def __init__(self,                              # pylint: disable=super-init-not-called
        cfg: EasyDict,
        input_resolution: tuple[int, int, int],     # Image resolution at input/output.
        in_channels,                                # Number of color channels at input.
        out_channels,                               # Number of color channels at output.
        label_shape         = 0,                    # Number of class labels, 0 = unconditional.
        augment_dim         = 0,                    # Augmentation label dimensionality, 0 = no augmentation.
    ):
        super().__init__()
        _ = label_shape, input_resolution, augment_dim # Unused, but kept for compatibility.
        assert in_channels == 3, f'Input channels must be 3, but got {in_channels}'
        assert out_channels == 3, f'Output channels must be 3, but got {out_channels}'

        self.cfg = cfg
        self.input_resolution = input_resolution
        self.model, self._latent_channels, self.compression_rate, self.vocab_size = init_diffusers_ae(
            self.cfg.arch,
            model_variant=self.cfg.model_variant,
            model_path=self.cfg.model_path,
            subfolder=self.cfg.subfolder,
            pretrained=self.cfg.pretrained,
            config_overrides=self.cfg.config_overrides,
            pretrained_lenient=self.cfg.pretrained_lenient,
            pretrained_lenient_resume_kwargs=self.cfg.pretrained_lenient_resume_kwargs,
        )

        if self.cfg.use_grad_checkpointing:
            self.model.enable_gradient_checkpointing()

        if self.cfg.freeze_encoder:
            misc.replace_parameters_with_buffers_(self.model.encoder)

    @property
    def latent_resolution(self) -> tuple[int, int, int]:
        return [math.ceil(r / f) for r, f in zip(self.input_resolution, self.compression_rate)]

    @property
    def unchunked_latent_resolution(self) -> tuple[int, int, int]:
        return self.latent_resolution

    @property
    def latent_channels(self) -> int:
        return self._latent_channels

    def forward(self,
        x: torch.Tensor,
        noise_labels: torch.Tensor=None,
        cond: Optional[TensorGroup]=None,
        augment_labels: Optional[torch.Tensor]=None,
        latents: Optional[torch.Tensor]=None,
        force_label_dropout: bool=False,
        encode: bool=True,
        decode: bool=True,
        latents_noise_std: Optional[torch.Tensor]=None,
        sc_latents: torch.Tensor=None,
        **vae_forward_kwargs,
    ) -> tuple[torch.Tensor, EasyDict]:
        _ = noise_labels, cond, augment_labels, force_label_dropout, latents_noise_std, sc_latents # Unused, but kept for compatibility.

        vae_dist_data = EasyDict(kl=None, mean=None, logvar=None)
        if encode:
            if self.cfg.extra_upsample:
                x = einops.reduce(x, 'b t c (h fh) (w fw) -> b t c h w', 'mean', fh=2, fw=2) # [b, t, c, h // 2, w // 2]
            latents_merged_all = []
            batch_size = x.shape[0]
            chunk_size = self.cfg.enc_split_input_size if self.cfg.enc_split_input_size is not None else x.shape[1]
            num_chunks = (x.shape[1] + chunk_size - 1) // chunk_size if chunk_size is not None else 1
            for chunk_idx in range(num_chunks):
                x_cur = x[:, chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size] if num_chunks > 1 else x
                if self.cfg.is_temporal:
                    x_cur = x_cur.transpose(1, 2) # [b, c, t, h, w]
                else:
                    x_cur = einops.rearrange(x_cur, 'b t c h w -> (b t) c h w') # [b * t, c, h, w]
                enc_output = self.model.encode(x_cur)
                if self.vocab_size is None:
                    latent_dist = enc_output.latent_dist # [b, c * 2, lt, lh, lw]
                    latents_merged_cur = torch.cat([latent_dist.mean, latent_dist.logvar], dim=1) # [b, c * 2, lt, lh, lw]
                else:
                    latents_merged_cur = enc_output.latents # [b, lt, lc, lh, lw]
                if self.cfg.is_temporal:
                    latents_merged_cur = latents_merged_cur.transpose(1, 2) # [b, t, c, h, w]
                else:
                    latents_merged_cur = einops.rearrange(latents_merged_cur, '(b t) c h w -> b t c h w', b=batch_size)
                latents_merged_all.append(latents_merged_cur)
            latents_merged = torch.cat(latents_merged_all, dim=1) # [b, lt, lc, lh, lw]
            if self.vocab_size is None:
                latents, vae_dist_data = vae_latents_forward(lambda x: x, latents_merged, use_mean_latents=self.cfg.use_mean_latents_in_eval and not self.training, **vae_forward_kwargs)
            else:
                latents = latents_merged # [b, lt, lc, lh, lw]

        if decode:
            pixels_all = []
            batch_size, num_frames, _channels, _height, _width = latents.shape
            chunk_size = self.cfg.dec_split_latent_size if self.cfg.dec_split_latent_size is not None else num_frames
            num_chunks = (num_frames + chunk_size - 1) // chunk_size if chunk_size is not None else 1

            for chunk_idx in range(num_chunks):
                latents_chunk = latents[:, chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size] if num_chunks > 1 else latents # [b, lt, lc, lh, lw]
                num_latents_cur = latents_chunk.shape[1]
                latents_chunk = einops.rearrange(latents_chunk, 'b t c h w -> b c t h w' if self.cfg.is_temporal else 'b t c h w -> (b t) c h w') # [b * t, c, h, w] or [b, c, t, h, w]
                num_frames_kwargs = dict(num_frames=num_latents_cur) if self.cfg.arch == 'svd_ae' else dict() # A dirty hack to make SVD AE work...
                pixels_cur = self.model.decode(latents_chunk, **num_frames_kwargs).sample  # [b * t, c, h, w] or [b, c, t, h, w]
                pixels_cur = einops.rearrange(pixels_cur, 'b c t h w -> b t c h w' if self.cfg.is_temporal else '(b t) c h w -> b t c h w', b=batch_size)
                pixels_all.append(pixels_cur)
            pixels = torch.cat(pixels_all, dim=1)  # [b, num_frames, c, h, w]
        else:
            pixels = None
        return pixels, EasyDict(latents=latents, **vae_dist_data)

#----------------------------------------------------------------------------
