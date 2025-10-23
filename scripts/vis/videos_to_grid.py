import os
import math
import argparse
from typing import Optional

import torch
import torchvision
import einops
import numpy as np
import torchvision.transforms.functional as TVF

from src.utils.logging_utils import save_video
from sds.transforms.functional import lean_resize_frames
from scripts.utils import find_videos_in_dir, find_images_in_dir
from src.data.video_decoder import VideoDecoder
from src.utils import misc
from src.utils.os_utils import lean_tqdm

#----------------------------------------------------------------------------

def convert_video_dir_to_grids(
        src: os.PathLike,
        dst: Optional[os.PathLike],
        src_mirrors: list[os.PathLike]=None,
        max_num_videos_per_grid: int=None,
        random_seed: int=None,
        ids_list: list[int]=None,
        num_grids: int=1,
        images: bool=False,
        video_names: Optional[list[str]]=None,
        verbose: bool=False,
        **convert_videos_to_grid_kwargs,
    ):
    if video_names is None:
        video_paths = find_videos_in_dir(src) if not images else find_images_in_dir(src)
    else:
        video_paths = [os.path.join(src, v) for v in video_names]
    if ids_list is not None:
        video_paths = [video_paths[i] for i in ids_list]
    if max_num_videos_per_grid is not None:
        if random_seed is not None:
            np.random.RandomState(random_seed).shuffle(video_paths)
        video_paths = video_paths[:max_num_videos_per_grid * num_grids]

    if src_mirrors is not None:
        # Using the exact same filenames from `src_mirror`, spacing them in-between the original ones.
        video_paths_mirrors = [[os.path.join(src_mirror, os.path.basename(vp)) for vp in video_paths] for src_mirror in src_mirrors]
        video_paths = [vp for row in zip(video_paths, *video_paths_mirrors) for vp in row]
        max_num_videos_per_grid = (max_num_videos_per_grid * (len(video_paths_mirrors) + 1)) if max_num_videos_per_grid is not None else None

    assert num_grids == 1 or max_num_videos_per_grid is not None, f'Got num_grids {num_grids} but max_num_videos_per_grid is None.'
    assert len(video_paths) // (max_num_videos_per_grid or len(video_paths)) == num_grids, f'Number of videos {len(video_paths)} is not divisible by max_num_videos_per_grid {max_num_videos_per_grid}.'

    save_path_main = f"{src.rstrip('/')}-all" if dst is None else os.path.splitext(dst)[0]
    for grid_idx, cur_video_paths in lean_tqdm(enumerate(np.split(np.array(video_paths), num_grids)), disable=not verbose):
        idx_suffix = f'-{grid_idx:02d}' if num_grids > 1 else ''
        convert_videos_to_grid(cur_video_paths, save_path=f'{save_path_main}{idx_suffix}', images=images, **convert_videos_to_grid_kwargs)

#----------------------------------------------------------------------------

def convert_videos_to_grid(
        video_paths, save_path: os.PathLike, nrow: int=None, max_len: int=None, resize=None,
        show_mirror_diff: bool=False, num_frames: int=None, images: bool=False, padding: int=2, unroll_frames: bool=False, **kwargs):
    videos = [read_object(vp, max_len, images) for vp in video_paths] # [n, 1, c, h, w]
    if num_frames is not None:
        random = np.random.RandomState(42)
        offsets = [random.randint(0, v.shape[0] - num_frames) for v in videos]
        videos = [v[offset:offset+num_frames] for v, offset in zip(videos, offsets)]
    if max_len is not None:
        # Pad each video with zeros to the same length.
        videos = [torch.cat([v, torch.zeros(max_len - v.shape[0], *v.shape[1:], dtype=v.dtype)], dim=0) if v.shape[0] < max_len else v for v in videos]
    max_len = min([v.shape[0] for v in videos]) if max_len is None else max_len
    videos = [v[:max_len] for v in videos] # (n, [l, c, h_i, w_i])
    if resize is not None:
        assert len(resize) == 2, f'Got resize {resize} but expected 2 values (it is spatial).'
        videos = [einops.rearrange(v, 't h w c -> t c h w') for v in videos]
        videos = [torch.stack(list(lean_resize_frames(v, resolution=resize, allow_vertical=False, maybe_crop=False, interpolation_mode=TVF.InterpolationMode.BILINEAR))) for v in videos]
        videos = einops.rearrange(videos, 'n t c h w -> n t h w c')
    else:
        videos = torch.stack(videos) # [n, t, c, h, w]
    nrow = math.ceil(math.sqrt(len(videos))) if nrow is None else nrow
    if show_mirror_diff:
        # Compute the difference between the original and mirrored videos.
        assert len(videos) % 2 == 0, f'Expected even number of videos, got {len(videos)}.'
        videos_fp32 = misc.maybe_rescale_uint8_to_fp32(videos) # [b, t, c, h, w]
        diff = misc.maybe_rescale_fp32_to_uint8((videos_fp32[::2] - videos_fp32[1::2]).abs()) # [b, t, c, h, w]
        videos = torch.stack([videos[::2], videos[1::2], diff], dim=0).view(-1, *videos.shape[1:]) # [b * 1.5, t, c, h, w]
    if unroll_frames:
        videos = einops.rearrange(videos, 'b t h w c -> 1 (b t) c h w') # [1, b * t, c, h, w]
    else:
        videos = einops.rearrange(videos, 'b t h w c -> t b c h w') # [num_frames, num_samples, t, c, h, w]
    grid = torch.stack([torchvision.utils.make_grid(v, nrow=nrow, padding=padding) for v in videos]).permute(0, 2, 3, 1) # [t, gh, gw, c]
    save_video(grid.numpy(), save_path, **kwargs)

def read_object(p, video_len: int, images: bool=False):
    if images:
        img = torchvision.io.read_image(p).permute(1,2,0).unsqueeze(0) # [1, h, w, c]
        if img.shape[3] == 1:
            img = img.repeat(1, 1, 1, 3)
        return img
    try:
        return read_video(p).permute(0, 2, 3, 1) # [num_frames, h, w, c]
    except Exception as e: # pylint: disable=broad-except
        print(f'Failed to read video {p} due to {e}.')
        return torch.zeros(128 if video_len is None else video_len, 1920, 1080, 3, dtype=torch.uint8)

def read_video(f):
    video_decoder = VideoDecoder(f, default_thread_type='NONE')
    duration = video_decoder.video_stream.duration * video_decoder.video_stream.time_base
    num_frames = math.floor(duration * video_decoder.framerate)
    frames = video_decoder.decode_frames_at_indexes(np.arange(0, num_frames)) # (num_frames, Image)
    frames = np.array([np.array(x) for x in frames]).transpose(0, 3, 1, 2) # [num_frames, c, h, w]
    return torch.from_numpy(frames)

#----------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('src', type=str, help='Path to the directory with videos.')
    parser.add_argument('--src_mirrors', type=str, default=None, nargs='+', help='Path to the directory with mirrored videos.')
    parser.add_argument('--dst', type=str, default=None, help='Path to the directory to save the grids to.')
    parser.add_argument('--nrow', type=int, default=None, help='Number of rows in the grid.')
    parser.add_argument('--max', type=int, default=None, help='Maximum amount of videso to convert.')
    parser.add_argument('--random_seed', type=int, help='Random seed specification if we want to randomly sample videos. We would pick random videos when visializing k < n available ones.')
    parser.add_argument('--ids', type=str, default=None, help='list of ids to convert.')
    parser.add_argument('--gif', action='store_true', help='Should we store as gif?')
    parser.add_argument('--num_grids', type=int, default=1, help='Number of grids to save.')
    parser.add_argument('--max_len', type=int, default=None, help='Maximum number of frames of the videos.')
    parser.add_argument('--resize', type=str, default=None, help='Resize the videos to the specified resolution.')
    parser.add_argument('--crf', type=int, default=10, help='CRF for the video.')
    parser.add_argument('--fps', type=int, default=24, help='FPS for the video.')
    parser.add_argument('--num_frames', type=int, default=None, help='Cut the video to some number of frames.') # TODO: merge with resolution?
    parser.add_argument('--images', action='store_true', help='Convert images instead of videos.')
    parser.add_argument('--video_names', type=str, default=None, help='Comma-separated list of video names to convert.')
    parser.add_argument('--show_mirror_diff', action='store_true', help='Show the difference between the original and mirrored videos.')
    parser.add_argument('--padding', type=int, default=2, help='Padding for the grid.')
    parser.add_argument('--unroll_frames', action='store_true', help='Instead of constructing an mp4 video grid, we should unroll the frames into a png.')
    parser.add_argument('--verbose', action='store_true', help='Verbose mode.')
    args = parser.parse_args()

    convert_video_dir_to_grids(
        src=args.src,
        dst=args.dst,
        src_mirrors=args.src_mirrors,
        max_num_videos_per_grid=args.max,
        random_seed=args.random_seed,
        ids_list=None if args.ids is None else map(int, args.ids.strip().split(',')),
        num_grids=args.num_grids,
        video_names=None if args.video_names is None else args.video_names.strip().split(','),

        save_gif=args.gif,
        # Extra kwargs.
        max_len=args.max_len,
        resize=[int(s) for s in args.resize.split('x')] if args.resize is not None else None,
        num_frames=args.num_frames,
        images=args.images,
        show_mirror_diff=args.show_mirror_diff,
        nrow=args.nrow,
        crf=args.crf,
        fps=args.fps,
        padding=args.padding,
        unroll_frames=args.unroll_frames,
        verbose=args.verbose,
    )

#----------------------------------------------------------------------------
