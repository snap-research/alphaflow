import os
import math
from urllib.parse import urlparse
from typing import Sequence, BinaryIO, Iterator, Any
import shutil

from beartype import beartype
from loguru import logger
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TVF

from src.data.video_decoder import VideoDecoder
import src.utils.distributed as dist_utils
from src.structs import BatchInfo

#----------------------------------------------------------------------------
# Some constants.

VIDEO_EXTENSIONS = {'.avi', '.mp4', '.webm', '.mov', '.mkv', '.m4v', '.mpg', '.mpeg', '.wmv', '.flv', '.3gp', '.3g2'}

#----------------------------------------------------------------------------
# Data processing utils.

@beartype
def decode_video(
        video_file: BinaryIO | bytes | str | None=None,
        num_frames_to_extract: int=1,
        video_decoder: VideoDecoder | None=None,
        random_offset: bool=True,
        frame_seek_timeout_sec: float=5.0,
        allow_shorter_videos: bool=False,
        framerate: float | None = None,
        thread_type: str | None = None,
        approx_frame_seek: bool = False,
        real_duration: float | None = None, # If provided, we ignore the video metadata.
        real_framerate: float | None = None, # If provided, we ignore the video metadata.
    ) -> tuple[Sequence[Image.Image], float]:
    """
    Decodes frames from a video file or bytes. Either video_file or video_decoder must be provided.
    """
    should_close_decoder = video_decoder is None
    if video_decoder is None:
        assert video_file is not None, "Video bytes must be provided if no video decoder is specified."
        video_decoder = VideoDecoder(src=video_file, default_thread_type=thread_type)
        should_close_decoder = True # We should close it since it's us who opened it.

    # Computing the full and target video durations and framerates.
    # AV video metadata is not always accurate, so we can opt for using our pre-computed one.
    base_framerate = video_decoder.framerate if real_framerate is None else real_framerate
    target_framerate = base_framerate if framerate is None else framerate
    full_video_duration = (video_decoder.video_stream.frames / base_framerate) if real_duration is None else real_duration
    clip_duration = num_frames_to_extract / target_framerate

    if clip_duration > full_video_duration:
        assert allow_shorter_videos, f"Video duration {full_video_duration} is shorter than the requested clip duration {clip_duration} while allow_shorter_videos={allow_shorter_videos}."
        clip_duration = full_video_duration
        num_frames_to_extract = max(1, round(clip_duration * target_framerate))

    start_frame_timestamp = np.random.rand() * max(full_video_duration - clip_duration, 0.0) if random_offset else 0.0
    frame_timestamps = np.linspace(start_frame_timestamp, start_frame_timestamp + clip_duration, num_frames_to_extract,)
    frame_timestamps = [t for t in frame_timestamps if t <= full_video_duration] # Filter out timestamps that are beyond the video duration.
    decoding_fn = video_decoder.decode_frames_at_times_approx if approx_frame_seek else video_decoder.decode_frames_at_times
    frames = decoding_fn(frame_timestamps, frame_seek_timeout_sec=frame_seek_timeout_sec) # (num_frames, Image)

    if should_close_decoder:
        video_decoder.close()

    return frames, clip_duration

@beartype
def sample_image_vae_latents(latents_dict: dict[str, torch.Tensor | np.ndarray]) -> torch.Tensor:
    mean, logvar = latents_dict['mean'], latents_dict['logvar'] # [lc, lh, lw], [lc, lh, lw]
    mean = torch.as_tensor(mean) if isinstance(mean, np.ndarray) else mean # [lc, lh, lw]
    logvar = torch.as_tensor(logvar) if isinstance(logvar, np.ndarray) else logvar # [lc, lh, lw]

    assert mean.ndim == 3, f"Unsupported latent shape: {mean.shape}. Expected 3D tensor."
    assert logvar.shape == mean.shape, f"Mean and logvar shapes do not match: {mean.shape} vs {logvar.shape}."

    return mean + torch.randn_like(mean) * torch.exp(0.5 * logvar) # [lc, lh, lw]

@beartype
def sample_video_vae_latents(
        latents_dict: dict[str, torch.Tensor | np.ndarray],
        orig_shape: tuple[int, int, int, int],
        num_rgb_frames_to_extract: int | None=None,
        fps_orig: float | None=None,
        fps_trg: float | None=None,
        random_offset: bool=True,
    ) -> torch.Tensor:

    mean, logvar = latents_dict['mean'], latents_dict['logvar'] # [lt | null, lc, lh, lw], [lt | null, lc, lh, lw]
    mean = torch.as_tensor(mean) if isinstance(mean, np.ndarray) else mean # [lt | null, lc, lh, lw]
    logvar = torch.as_tensor(logvar) if isinstance(logvar, np.ndarray) else logvar # [lt | null, lc, lh, lw]

    assert mean.ndim == 4, f"Unsupported latent shape: {mean.shape}. Expected 4D tensor."
    assert logvar.shape == mean.shape, f"Mean and logvar shapes do not match: {mean.shape} vs {logvar.shape}."

    temporal_compression_rate = math.ceil(orig_shape[0] / mean.shape[0])
    num_rgb_frames_to_extract = orig_shape[0] if num_rgb_frames_to_extract is None else num_rgb_frames_to_extract
    num_latent_frames_to_extract = math.ceil(num_rgb_frames_to_extract / temporal_compression_rate)
    if fps_trg is not None:
        assert fps_orig is not None, "Original FPS must be provided if target FPS is specified."
        assert (fps_orig / fps_trg).is_integer() and fps_orig >= fps_trg, f"FPS ratio {fps_orig} / {fps_trg} is not a positive integer. For latents, we cant decode at arbitrary framerates."
        frames_skip_factor = int(fps_orig / fps_trg)
    else:
        frames_skip_factor = 1

    assert (num_latent_frames_to_extract * frames_skip_factor) <= mean.shape[0], f"Requested {num_latent_frames_to_extract} latent frames with skip factor {frames_skip_factor}, but only {mean.shape[0]} are available in the latents."

    start_frame_idx = np.random.randint(low=0, high=mean.shape[0] - num_latent_frames_to_extract + 1) if random_offset else 0
    latent_frames_idx = np.arange(start_frame_idx, start_frame_idx + num_latent_frames_to_extract, frames_skip_factor)
    latent_frames_mean = mean[latent_frames_idx]  # [clip_length, lc, lh, lw]
    latent_frames_logvar = logvar[latent_frames_idx]  # [clip_length, lc, lh, lw]

    frames = latent_frames_mean + torch.randn_like(latent_frames_logvar) * np.exp(0.5 * latent_frames_logvar) # [clip_length | null, lc, lh, lw]

    return frames

@beartype
def convert_pil_image_to_byte_tensor(image: Image.Image, convert_to_rgb: bool=True, cut_alpha: bool=False) -> torch.Tensor:
    img_pt = torch.from_numpy(np.array(image)) # [h, w, c | null]
    img_pt = img_pt.unsqueeze(-1) if img_pt.ndim == 2 else img_pt # [h, w, 1]
    img_pt = img_pt.repeat(1, 1, 3) if img_pt.shape[2] == 1 and convert_to_rgb else img_pt # [h, w, 3]
    img_pt = img_pt[:, :, :3] if img_pt.shape[2] > 3 and cut_alpha else img_pt # [h, w, 3]
    img_pt = img_pt.permute(2, 0, 1) # [3, h, w]

    return img_pt # [3, h, w]

@beartype
def lean_resize_frames(
        frames: Sequence[Image.Image] | Sequence[torch.Tensor] | torch.Tensor,
        resolution: tuple[int, int],
        crop_before_resize: bool=True,
        allow_vertical: bool=False,
        random_resize: dict[str, float] | None=None,
        interpolation_mode='lanczos',
    ) -> Sequence[Image.Image] | Sequence[torch.Tensor]:
    """
    Resizes each frame in the batch to the specified resolution.
    Possibly inverts it if it's vertical and allowed to do so.
    Also, can randomly downsample the frames given the `random_resize` dict of the form {(h,w): probability}.
    Args:
        - frames: List of frames to resize, either as PIL Images or torch Tensors.
        - resolution: Target resolution as a tuple (width, height).
        - crop_before_resize: If True, crops the frames to the target aspect ratio before resizing.
        - allow_vertical: If True, allows the frames to be resized to a vertical resolution via flipping input `resolution` as (width, height).
        - random_resize: A dictionary mapping resolutions to their probabilities for random downsampling.
        - interpolation_mode: Interpolation mode to use for resizing.
    Returns:
        - List of resized frames.
    """
    assert len(resolution) == 2, f"Wrong resolution: {resolution}"
    w, h = frames[0].size if isinstance(frames[0], Image.Image) else (frames[0].shape[2], frames[0].shape[1]) # [h, w]
    is_originally_vertical = h > w

    if random_resize is not None:
        assert sum(random_resize.values()) == 1.0, f"Probabilities should sum to 1.0: {random_resize}"
        random_resize = {k: v for k, v in random_resize.items() if k[0] <= w and k[1] <= h} # Only keep resolutions that are smaller than the original one.
        if len(random_resize) > 0:
            resolutions, probs = zip(*random_resize.items()) # [num_resolutions], [num_resolutions]
            resolution = resolutions[np.random.choice(len(resolutions), p=np.array(probs) / sum(probs))] # [2]

    h_trg, w_trg = (max(resolution), min(resolution)) if is_originally_vertical and allow_vertical else resolution

    if w == w_trg and h == h_trg:
        # TVF.resize has a similar shortcut, but here we won't even iterate.
        return frames

    if crop_before_resize:
        frames = [crop_to_aspect_ratio(x, target_aspect_ratio=w_trg / h_trg) for x in frames]
    frames = [TVF.resize(x, size=(h_trg, w_trg), interpolation=TVF.InterpolationMode(interpolation_mode)) for x in frames]

    return frames

@beartype
def crop_to_aspect_ratio(image: Image.Image | torch.Tensor, target_aspect_ratio: float) -> Image.Image | torch.Tensor:
    """Crops the image to the specified aspect ratio."""
    if isinstance(image, Image.Image):
        cur_w, cur_h = image.size
    elif isinstance(image, torch.Tensor):
        assert image.shape[0] in (1, 3, 4), f"Unsupported number of channels in shape {image.shape}. Must be 1, 3, or 4."
        _c, cur_h, cur_w = image.shape
    else:
        raise TypeError(f"Unsupported type: {type(image)}. Must be PIL.Image or torch.Tensor.")

    cur_aspect_ratio = cur_w / cur_h

    if cur_aspect_ratio > target_aspect_ratio:
        # Too wide: crop width
        new_width = int(cur_h * target_aspect_ratio)
        offset_left = (cur_w - new_width) // 2
        return _apply_crop(image, (offset_left, 0, offset_left + new_width, cur_h))
    else:
        # Too tall: crop height
        new_height = int(cur_w / target_aspect_ratio)
        offset_top = (cur_h - new_height) // 2
        return _apply_crop(image, (0, offset_top, cur_w, offset_top + new_height))

@beartype
def _apply_crop(x: Image.Image | torch.Tensor, crop: tuple[int, int, int, int]) -> Image.Image | torch.Tensor:
    if isinstance(x, Image.Image):
        return x.crop(crop)
    elif isinstance(x, torch.Tensor):
        return x[:, crop[1]:crop[3], crop[0]:crop[2]]

#----------------------------------------------------------------------------
# Data loading utils.

def infer_batch_info(batch_size: int | None = None, batch_gpu: int | None = None, num_accum_rounds: int | None = None) -> BatchInfo:
    """
    Fixes batch config options (with some values missing) into the fully-filled one.
    We provide the option to specify the batch size either through batch size or batch_gpu + num_accum_rounds.
    """
    world_size = dist_utils.get_world_size()
    if batch_size is None:
        assert batch_gpu is not None, f"If batch_size={batch_size} is None, batch_gpu={batch_gpu} must be specified."
        num_accum_rounds = 1 if num_accum_rounds is None else int(num_accum_rounds)
        batch_size = int(batch_gpu * world_size * num_accum_rounds)
    else:
        # Note: batch_size/batch_gpu/num_accum_rounds can be equal to 0 --- that means that we don't train on a given dataset.
        assert batch_size % world_size == 0, f"batch_size={batch_size} must be divisible by world_size={world_size}"
        assert batch_gpu is None or (batch_size // world_size) % batch_gpu == 0, f"If batch_size is specified, batch_gpu must be divisible by (batch_size={batch_size} // world_size={world_size})"
        batch_gpu = (batch_size // world_size) if batch_gpu is None else int(batch_gpu)
        num_accum_rounds = 0 if batch_gpu == 0 else (batch_size // (batch_gpu * world_size))

    return BatchInfo(batch_size, batch_gpu, num_accum_rounds)

#----------------------------------------------------------------------------
# S3/os utils utils.

@beartype
def get_s3_file_paths(s3_path: str, is_main_process: bool=False, s3_listdir_cache_dir: str | None=None) -> list[str]:
    import boto3
    import s3fs

    assert s3_path.startswith('s3://'), f'Expected an S3 path, but got: {s3_path}'
    s3_path = s3_path[len('s3://'):]
    bucket_name, prefix = s3_path.split('/', 1) if '/' in s3_path else (s3_path, '')

    # Before checking S3, see if we have a cached inventory file.
    s3_path_key = path_key(s3_path, num_parts=-1, drop_ext=True)
    s3_path_hash_name = f'{s3_path_key}.txt'
    s3_path_inventory_file = os.path.join(s3_listdir_cache_dir, s3_path_hash_name) if s3_listdir_cache_dir is not None else None
    s3 = s3fs.S3FileSystem()
    if s3_path_inventory_file is not None and s3.exists(s3_path_inventory_file):
        logger.info(f'Loading S3 inventory from {s3_path_inventory_file}...')
        with s3.open(s3_path_inventory_file, 'r') as f:
            file_paths = [line.strip() for line in f.readlines() if line.strip()]
        return file_paths

    paginator = boto3.client('s3').get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    file_paths = []
    page_iterator = tqdm(page_iterator, desc=f'Listing S3 files in {s3_path}', unit='page') if is_main_process else page_iterator
    for page in page_iterator:
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                if not key.endswith('/'):  # Exclude directories
                    file_paths.append(f's3://{bucket_name}/{key}')

    if s3_path_inventory_file is not None and is_main_process:
        logger.info(f'Caching S3 inventory to {s3_path_inventory_file}...')
        os.makedirs(os.path.dirname(s3_path_inventory_file), exist_ok=True)
        with s3.open(s3_path_inventory_file, 'w') as f:
            for file_path in file_paths:
                f.write(f'{file_path}\n')

    return file_paths

@beartype
def copy_file(src: str, dst: str, skip_if_exists: bool=True):
    """Copies a file from `src` to `dst`. Supperts S3->local and local->local copies."""
    if os.path.exists(dst) and os.path.getsize(dst) > 0 and skip_if_exists:
        return

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if src.startswith('s3://'):
        import boto3
        from botocore.config import Config

        s3 = boto3.client('s3', config=Config(retries={'mode': 'adaptive'}))
        bucket_name, key = src[len('s3://'):].split('/', 1)
        s3.download_file(bucket_name, key, dst)
    else:
        shutil.copyfile(src, dst)

def path_key(path: str, num_parts: int = 2, drop_ext: bool=False) -> str:
    assert num_parts >= 1 or num_parts == -1
    path = path.rstrip('/')
    u = urlparse(path)
    parts = [p for p in u.netloc.split('/') + u.path.strip('/').split('/') if p]
    parts = parts[-num_parts:] if num_parts != -1 else parts
    assert len(parts) > 0, f"Path {path} has no valid parts to form a key."
    parts[-1] = os.path.splitext(parts[-1])[0] if drop_ext else parts[-1]
    prefix = f"{u.scheme}-" if u.scheme else ""
    result = prefix + "-".join(parts)

    return result

@beartype
def inf_loop_dataloader(dataloader: torch.utils.data.DataLoader) -> Iterator[dict[str, Any]]:
    while True: yield from dataloader

#----------------------------------------------------------------------------
