import os
import re
import shutil
import random
import traceback
import contextlib
import subprocess
from typing import Optional, Callable, Union

import joblib
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TVF
from torchvision.utils import make_grid
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import pil_loader

from src.utils.os_utils import file_ext, listdir_full_paths
from src.data.utils import VIDEO_EXTENSIONS

#----------------------------------------------------------------------------
# Image/video processing utils.

def display_dir(dir_path: os.PathLike, num_imgs: int=25, selection_strategy: str="order", n_skip_imgs: int=0, ignore_regex=None, **kwargs) -> "Image":
    Image.init()
    if selection_strategy in ('order', 'random'):
        img_paths = find_images_in_dir(dir_path, ignore_regex=ignore_regex)
        img_paths = img_paths[n_skip_imgs:]

    if selection_strategy == 'order':
        img_paths = img_paths[:num_imgs]
    elif selection_strategy == 'random':
        img_paths = random.sample(img_paths, k=num_imgs)
    elif selection_strategy == 'random_imgs_from_subdirs':
        img_paths = [p for d in [d for d in listdir_full_paths(dir_path) if os.path.isdir(d)] for p in random.sample(listdir_full_paths(d), k=num_imgs)]
    else:
        raise NotImplementedError(f'Unknown selection strategy: {selection_strategy}')

    return display_imgs(img_paths, **kwargs)

def display_imgs(img_paths: list[os.PathLike], nrow: bool=None, resize: int=None, crop: tuple=None, center_crop: bool=True, padding: int=2, pad_to_square: bool=False) -> "Image":
    imgs = [Image.open(p) for p in img_paths]
    imgs = [(x.convert('RGB') if np.array(x).dtype != np.int32 else x) for x in imgs]
    if not crop is None:
        imgs = [img.crop(crop) for img in imgs]
    if pad_to_square:
        imgs = [pad_image_to_square_tensor(x) for x in imgs]
    if not resize is None:
        imgs = [TVF.resize(x, size=resize, interpolation=TVF.InterpolationMode.LANCZOS) for x in imgs]
    if center_crop:
        imgs = [TVF.center_crop(x, output_size=min(x.size)) for x in imgs] # [num_imgs, c, h, w]
    imgs = [TVF.to_tensor(x) for x in imgs] # [num_imgs, c, h, w]
    imgs = [((x.float() / 2 ** 16) if x.dtype == torch.int32 else x) for x in imgs] # [num_imgs, c, h, w]
    imgs = [x.repeat(3, 1, 1) if x.shape[0] == 1 else x for x in imgs] # [num_imgs, c, h, w]
    imgs = torch.stack(imgs) # [num_imgs, c, h, w]
    grid = make_grid(imgs, nrow=(int(np.sqrt(imgs.shape[0])) if nrow is None else nrow), padding=padding) # [c, grid_h, grid_w]
    grid = TVF.to_pil_image(grid)

    return grid

def resize_and_save_image(src_path: str, trg_path: str, size: int, ignore_grayscale: bool=False, ignore_broken: bool=False, ignore_existing: bool=False, force_three_channels: bool=False):
    Image.init()
    assert file_ext(src_path) in Image.EXTENSION, f"Unknown image extension: {src_path}"
    assert file_ext(trg_path) in Image.EXTENSION, f"Unknown image extension: {trg_path}"

    if ignore_existing and os.path.isfile(trg_path):
        return

    try:
        img = Image.open(src_path)
        if img.mode == 'L' and ignore_grayscale:
            return
        img.load() # required for png.split()
    except: # pylint: disable=bare-except
        if ignore_broken:
            return
        else:
            raise

    img = center_resize_crop(img, size)
    jpg_kwargs = {'quality': 95} if file_ext(trg_path) == '.jpg' else {}
    num_channels = len(img.split())

    if file_ext(trg_path) in ('.jpg', '.jpeg') and num_channels == 4:
        jpg = Image.new("RGB", img.size, (255, 255, 255))
        jpg.paste(img, mask=img.split()[3]) # 3 is the alpha channel
        jpg.save(trg_path, **jpg_kwargs)
    else:
        if img.mode == "CMYK" or (force_three_channels and num_channels == 4):
            img = img.convert("RGB")
        img.save(trg_path, **jpg_kwargs)

def resize_and_save_video(src_path: str, trg_path: str, size: Union[int, tuple[int, int]], ignore_grayscale: bool=False, ignore_broken: bool=False, ignore_existing: bool=False, force_three_channels: bool=False, min_acceptable_size=2048):
    _ = ignore_grayscale, force_three_channels # For backward compatibility with resize_and_save_image

    if ignore_existing and os.path.isfile(trg_path) and os.path.getsize(trg_path) > min_acceptable_size:
        return

    h, w = (size, size) if isinstance(size, int) else size
    gcd = np.gcd(h, w)
    aw, ah = w // gcd, h // gcd

    cmd = [
        '/usr/bin/ffmpeg', '-i', src_path,
        '-vcodec', 'libx265',
        '-x265-params', 'keyint=5:no-open-gop=1:tune=fastdecode',
        '-pix_fmt', 'rgb24',
        '-crf', '10',
        '-an',
        '-vf', f"crop='if(gte(a,{aw}/{ah}),ih*{aw}/{ah},iw):if(lt(a,{aw}/{ah}),iw/{aw}*{ah},ih)',scale={w}:{h}:flags=bicubic,setpts=PTS-STARTPTS",
        '-y', trg_path,
    ]

    num_retries = 3 # Sometimes, ffmpeg does not produce good results when launched from multiple threads :|
    while num_retries >= 0:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            if result.returncode != 0:
                raise ChildProcessError(f"ffmpeg returned an error while resizing {src_path}:\n{cmd}\nGave error:\n{result.stderr}") # pylint: disable=broad-except
            assert os.path.exists(trg_path), f"FFmpeg did not produce the output file: {trg_path}"
            assert os.path.getsize(trg_path) > min_acceptable_size, f"FFmpeg produced a too small file: {trg_path}: {os.path.getsize(trg_path)} bytes"
            return
        except ChildProcessError as e: # pylint: disable=broad-except
            if ignore_broken:
                return
            elif num_retries <= 0:
                print(f'Failed to run the command:\n{" ".join(cmd)}')
                print(e)
                traceback.print_exc()
                raise
            else:
                num_retries -= 1
                continue

def center_crop_video(video: torch.Tensor, crop_size: int=None) -> torch.Tensor:
    assert len(video.shape) == 5, f"Input tensor must be 5D: {video.shape}"
    _, _, _, h, w = video.shape
    crop_size = min(h, w) if crop_size is None else crop_size
    assert video.shape[3] >= crop_size and video.shape[4] >= crop_size, "Crop size should be less than or equal to video dimensions"

    start_x = (w - crop_size) // 2
    start_y = (h - crop_size) // 2

    end_x = start_x + crop_size
    end_y = start_y + crop_size

    return video[..., start_y:end_y, start_x:end_x]

def center_resize_crop(img: Image, size: int) -> Image:
    img = TVF.center_crop(img, min(img.size)) # First, make it square
    img = TVF.resize(img, size, interpolation=TVF.InterpolationMode.LANCZOS) # Now, resize it

    return img

def lanczos_resize_tensors(x: torch.Tensor, size):
    x = [TVF.to_pil_image(img) for img in x]
    x = [TVF.resize(img, size=size, interpolation=TVF.InterpolationMode.LANCZOS) for img in x]
    x = [TVF.to_tensor(img) for img in x]

    return torch.stack(x)

class ImagePathsDataset(VisionDataset):
    def __init__(self, img_paths: list[os.PathLike], transform: Optional[Callable]=None): # pylint: disable=super-init-not-called
        self.imgs_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.imgs_paths)

    def __getitem__(self, idx: int):
        image = pil_loader(self.imgs_paths[idx])

        if not self.transform is None:
            image = self.transform(image)

        return dict(image=image, path=self.imgs_paths[idx])

def pad_image_to_square_tensor(image, fill=255, padding_mode='constant'):
    """
    Written by ChatGPT (don't judge me).
    """
    width, height = image.size
    max_dim = max(width, height)
    padding_left = (max_dim - width) // 2
    padding_right = max_dim - width - padding_left
    padding_top = (max_dim - height) // 2
    padding_bottom = max_dim - height - padding_top
    padding = (padding_left, padding_top, padding_right, padding_bottom)
    padded_tensor = TVF.pad(image, padding, fill=fill, padding_mode=padding_mode)
    return padded_tensor

#----------------------------------------------------------------------------
# OS utils.

def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def copy_files(src_path: os.PathLike, files_to_copy: list[os.PathLike], dst_path: os.PathLike, flatten_parent_dir: bool=False, num_jobs: int=8, should_copy: bool=True):
    """
    src_path --- main dataset directory
    files_to_copy --- filepaths inside the `src_path` directory
    dst_path --- where to save the files
    TODO: rename the function or why we copy by default?
    """
    jobs = []
    dirs_to_create = []

    for filepath in tqdm(files_to_copy, desc='Collecting jobs'):
        src_file_path = os.path.join(src_path, filepath)
        if flatten_parent_dir:
            parent_dir = os.path.dirname(filepath)
            dst_file_path = os.path.join(dst_path, os.path.dirname(parent_dir), f'{os.path.basename(parent_dir)}_{os.path.basename(filepath)}')
        else:
            dst_file_path = os.path.join(dst_path, filepath)
        dirs_to_create.append(os.path.dirname(dst_file_path))
        jobs.append(joblib.delayed(shutil.copy if should_copy else shutil.move)(src=src_file_path, dst=dst_file_path))

    for d in tqdm(list(set(dirs_to_create)), desc='Creating necessary directories'):
        if d != '':
            os.makedirs(d, exist_ok=True)

    with tqdm_joblib(tqdm(desc="Executing jobs", total=len(jobs))):
        joblib.Parallel(n_jobs=num_jobs)(jobs)

def remove_empty_files(dir_path: os.PathLike, verbose: bool=False):
    for f in tqdm(get_all_files(dir_path)):
        if os.stat(f).st_size == 0:
            if verbose:
                print(f'Removing empty file: {f}')
            os.remove(f)

def maybe_makedirs(d: os.PathLike):
    # TODO: what the hell is this function name?
    if d != '':
        os.makedirs(d, exist_ok=True)

def topmost_dir(path: os.PathLike) -> str:
    assert not path == ''
    if path == os.path.sep: return os.path.sep
    if path.startswith(os.path.sep): path = path[1:]
    return path.split(os.path.sep)[0]

def get_all_files(dir_path: os.PathLike, full_path: bool=True, ext_white_list: list[str]=None) -> list[os.PathLike]:
    all_files = [os.path.join(root, fname) for root, _dirs, files in os.walk(dir_path) for fname in files]
    if not ext_white_list is None:
        ext_white_list = set(list(ext_white_list))
        all_files = [f for f in all_files if file_ext(f) in ext_white_list]
    if full_path:
        all_files = [os.path.join(dir_path, f) for f in all_files]
    return all_files

def compress_to_zip(dir_to_compress: os.PathLike, delete: bool=False):
    shutil.make_archive(dir_to_compress, 'zip', root_dir=os.path.dirname(dir_to_compress), base_dir=os.path.basename(dir_to_compress))

    if delete:
        shutil.rmtree(dir_to_compress)

def find_images_in_dir(dir_path: os.PathLike, **kwargs):
    Image.init()
    return find_files_in_dir(dir_path, Image.EXTENSION, **kwargs)

def find_videos_in_dir(dir_path: os.PathLike, **kwargs):
    return find_files_in_dir(dir_path, VIDEO_EXTENSIONS, **kwargs)

def find_files_in_dir(dir_path: os.PathLike, extensions, ignore_regex=None, full_path: bool=True):
    files_all = [os.path.relpath(os.path.join(root, fname), start=dir_path) for root, _dirs, files in os.walk(dir_path) for fname in files]
    files_filtered = sorted([f for f in files_all if file_ext(f) in extensions])
    if full_path:
        files_filtered = [os.path.join(dir_path, f) for f in files_filtered]
    if not ignore_regex is None:
        files_filtered = [f for f in files_filtered if not re.fullmatch(ignore_regex, f)]
    return files_filtered

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

#----------------------------------------------------------------------------
# Snapshot management utils.

def get_snapshot_iters(snapshot_names: list[str], snapshot_ext='.pt'):
    ckpt_regex = re.compile(rf".*snapshot-(\d{{8}}){snapshot_ext}$")
    matches = [ckpt_regex.match(c) for c in snapshot_names]
    snapshot_iters = [int(m.group(1)) if m else float('inf') for m in matches]
    return snapshot_iters

def list_snapshots(experiment_dir: os.PathLike, snapshost_ext: str='.pt'):
    return [d for d in listdir_full_paths(experiment_dir) if d.startswith('snapshot-') and d.endswith(snapshost_ext)]

#----------------------------------------------------------------------------
# AWS utils.

def get_s3_directory_files(s3_client, s3_path: str) -> list[str]:
    print(f'Getting contents of {s3_path}...', end=' ')
    bucket_name, dir_path = s3_path.split('/', 1)
    assert dir_path.endswith('/'), f"Directory path must end with a slash (got {dir_path}) since it serves as a prefix and we want to prevent inadvertent matches."

    paginator = s3_client.get_paginator('list_objects_v2')
    contents = []

    for page in paginator.paginate(Bucket=bucket_name, Prefix=dir_path):
        if not 'Contents' in page:
            continue
        contents.extend([obj['Key'] for obj in page['Contents']])

    contents = sorted(contents)
    print(f'Done! (found {len(contents)} files)')

    return contents

def slow_check_if_s3_path_exists(s3_path) -> bool:
    import boto3 # pylint: disable=import-outside-toplevel

    if not s3_path.startswith("s3://"):
        return os.path.exists(s3_path)

    bucket_name, _, key = s3_path[5:].partition('/')
    s3 = boto3.client('s3')

    try:
        s3.head_object(Bucket=bucket_name, Key=key)
        return True
    except s3.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            return False
        raise

#----------------------------------------------------------------------------
