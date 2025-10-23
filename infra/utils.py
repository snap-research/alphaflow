import os
import re
import math
import shutil
import hashlib
import subprocess
from shutil import copyfile
from typing import Optional, Any

from hydra.utils import instantiate
import click
import git
from omegaconf import DictConfig
from src.utils.os_utils import listdir_full_paths

#----------------------------------------------------------------------------
# OS utils.

def copy_objects(objects_to_copy: list[os.PathLike], dst_dir: os.PathLike, create_dst_dir: bool=False):
    """
    Copies objects (files, directories or symlines) to the destination directory.
    """
    if create_dst_dir:
        os.makedirs(dst_dir, exist_ok=False)
    else:
        assert os.path.isdir(dst_dir), f"Destination directory {dst_dir} does not exist and create_dst_dir is False."
    for src_path in objects_to_copy:
        dst_path = os.path.join(dst_dir, os.path.basename(src_path))
        if os.path.islink(src_path):
            os.symlink(os.readlink(src_path), dst_path)
        elif os.path.isfile(src_path):
            copyfile(src_path, dst_path)
        elif os.path.isdir(src_path):
            copy_objects(listdir_full_paths(src_path), dst_path, create_dst_dir=True)
        else:
            raise NotImplementedError(f"Unknown object type: {src_path}")

def create_symlinks(symlinks_to_create: list[os.PathLike], dst_dir: os.PathLike, verbose: bool=False):
    """
    Creates symlinks to the given paths inside the destination directory.
    """
    for src_path in symlinks_to_create:
        dst_path = os.path.join(dst_dir, os.path.basename(src_path))

        if os.path.islink(src_path):
            # Let's not create symlinks to symlinks
            # Since dropping the current symlink will break the experiment
            os.symlink(os.readlink(src_path), dst_path)
        else:
            if verbose:
                print(f'Creating a symlink to {src_path}, so try not to delete it accidentally!')
            os.symlink(src_path, dst_path)


def read_env_vars(names: list[str]) -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k in names}

#----------------------------------------------------------------------------
# Git utils.

def is_git_repo(path: os.PathLike):
    try:
        _ = git.Repo(path).git_dir
        return True
    except git.exc.InvalidGitRepositoryError:
        return False

def get_git_hash() -> Optional[str]:
    if not is_git_repo(os.getcwd()):
        return None

    try:
        return subprocess \
            .check_output(['git', 'rev-parse', '--short', 'HEAD']) \
            .decode("utf-8") \
            .strip()
    except:
        return None

def get_git_hash_suffix() -> str:
    git_hash: Optional[str] = get_git_hash()
    git_hash_suffix = "-nogit" if git_hash is None else f"-{git_hash}"

    return git_hash_suffix

def are_there_uncommitted_changes() -> bool:
    try:
        return len(subprocess.check_output('git status -s'.split()).decode("utf-8")) > 0
    except:
        return False

def get_git_uncommited_changes_desc() -> str:
    return "-dirty" if are_there_uncommitted_changes() else ""

#----------------------------------------------------------------------------
# Hydra utils.

def dict_to_hydra_overrides(d, parent_key='') -> list[str]:
    HYDRA_SPECIAL_CHARS = ('+', '~')
    overrides = []
    for k, v in d.items():
        new_key = f"{k[0] if k.startswith(HYDRA_SPECIAL_CHARS) else ''}{parent_key}.{k.lstrip(''.join(HYDRA_SPECIAL_CHARS))}" if parent_key else k
        if isinstance(v, dict):
            overrides.extend(dict_to_hydra_overrides(v, new_key))
        else:
            overrides.append(f"{new_key}={maybe_none_to_null(v)}")
    return overrides

def maybe_none_to_null(v: Any) -> Any:
    return 'null' if v is None else v

def recursive_instantiate(cfg: DictConfig):
    for key in cfg:
        if isinstance(cfg[key], DictConfig):
            if '_target_' in cfg[key]:
                cfg[key] = instantiate(cfg[key])
            else:
                recursive_instantiate(cfg[key])

def diff(a: float, b: float):
    return a - b

def product(values):
    import numpy as np # pylint: disable=import-outside-toplevel
    return np.prod([x for x in values]).item()

def product_ab(a: float, b: float):
    """Hydra cannot overwrite `values` array so using this..."""
    return a * b

def divide(dividend, divisor):
    return dividend / divisor

def int_divide(dividend, divisor):
    return int(dividend / divisor)

def log2_divide(dividend, divisor):
    return int(math.log2(dividend / divisor))

def basename_without_ext(path: os.PathLike):
    return os.path.splitext(os.path.basename(path))[0].lower()

def compute_magnitude_ema_beta(batch_size: int) -> float:
    return 0.5 ** (batch_size / (20 * 1e3))

def linspace(val_from: float, val_to: float, num_steps: int) -> list[float]:
    # import numpy as np
    # return np.linspace(val_from, val_to, num_steps).tolist()
    assert num_steps > 1, f"Too small num_steps: {num_steps}"
    return [val_from + (val_to - val_from) * i / (num_steps - 1) for i in range(num_steps)]

def compute_breadth_coef(scale: float, resolution: tuple[int, int, int]) -> float:
    return scale * (resolution[1] * resolution[2]) ** 0.5

def slice_list(array: list[Any], start: Optional[int]=0, end: Optional[int]=None) -> list[Any]:
    return array[start:end]

def safe_get_list_value(array: list[Any], index: int) -> Any:
    return array[index] if index < len(array) else None

def str_replace(input_string: str, mapping: dict[str, str]) -> str:
    for old, new in mapping.items():
        input_string = input_string.replace(old, new)
    return input_string

def get_with_default(value: Optional[Any], default: Any) -> Any:
    return default if value is None else value


def cfg_to_args_str(cfg: DictConfig, use_dashes=True) -> str:
    dashes = '--' if use_dashes else ''

    return ' '.join([f'{dashes}{p}={cfg[p]}' for p in cfg])

def num_gpus_to_mem(num_gpus: int, mem_per_gpu: 64) -> str:
    # Doing it here since hydra config cannot do formatting for ${...}
    return f"{num_gpus * mem_per_gpu}G"

def get_resolution_str(resolution: tuple[int, int, int]) -> str:
    return "x".join([str(r) for r in resolution])

#----------------------------------------------------------------------------
# Experiment directory management utils.

def create_project_dir(
        project_dir: os.PathLike,
        output_dir: os.PathLike,
        objects_to_copy: list[os.PathLike],
        symlinks_to_create: list[os.PathLike],
        quiet: bool=False,
        ignore_uncommited_changes: bool=False,
        overwrite: bool=False,
    ):

    if is_git_repo(os.getcwd()) and are_there_uncommitted_changes():
        if ignore_uncommited_changes or click.confirm("There are uncommited changes. Continue?", default=False):
            pass
        else:
            raise PermissionError("Cannot created a dir when there are uncommited changes")

    if os.path.exists(project_dir):
        if overwrite or click.confirm(f'Dir {project_dir} already exists. Overwrite it?', default=False):
            shutil.rmtree(project_dir)
        else:
            print('User refused to delete an existing project dir.')
            raise PermissionError("There is an existing dir and I cannot delete it.")

    os.makedirs(project_dir)
    os.makedirs(output_dir)
    copy_objects(objects_to_copy, project_dir)
    create_symlinks(symlinks_to_create, project_dir, verbose=not quiet)

    if not quiet:
        print(f'Created a project dir: {project_dir}')


def maybe_increment_filename(filename: os.PathLike):
    """
    Checks if this file exists in the directory and adds a suffix if it does.
    Written by ChatGPT (do not judge me if it fails).
    """

    base_name, ext = os.path.splitext(filename)

    if not os.path.exists(filename):
        return filename

    # Find existing files in the format: base_name_{counter}.ext
    dir_name = os.path.dirname(filename)
    pattern = re.compile(fr"^{base_name}_\d{{6}}{ext}$")

    # Find the highest counter already used
    highest_counter = 0
    for file in os.listdir(dir_name):
        if pattern.match(file):
            counter = int(file[len(base_name) + 1 : -len(ext)])
            highest_counter = max(highest_counter, counter)

    return f"{base_name}_{highest_counter+1:06d}{ext}"


def get_next_experiment_id_str(experiments_dir: os.PathLike, **kwargs) -> str:
    return f"{get_next_experiment_id(experiments_dir, **kwargs):04d}"


def get_next_experiment_id(experiments_dir: os.PathLike, max_allowed_id: int | None=None) -> int:
    cur_experiments = listdir_full_paths(experiments_dir)
    cur_experiments = [e for e in cur_experiments if os.path.isdir(e) and len(os.path.basename(e)) >= 4 and os.path.basename(e).split('-')[0].isdigit()]
    experiment_ids = [int(os.path.basename(e).split('-')[0]) for e in cur_experiments]

    if max_allowed_id is not None:
        assert max_allowed_id not in experiment_ids, f"Max experiment ID is given ({max_allowed_id}) and it is already in use."
        experiment_ids = [eid for eid in experiment_ids if eid < max_allowed_id]

    latest_experiment_id = max([eid for eid in experiment_ids], default=0)

    return latest_experiment_id + 1


def hash_string(input_string, output_length: int=3):
    """Written by ChatGPT."""
    sha256 = hashlib.sha256() # Create a SHA-256 hash object
    sha256.update(input_string.encode('utf-8')) # Encode the input string as bytes and update the hash object
    digest_bytes = sha256.digest() # Get the digest as bytes

    # We'll reduce the hash to a 3-symbol lowercase English letter hash
    # by taking the modulo 26 of the sum of the digest bytes and mapping to 'a' through 'z'
    return ''.join([chr((byte % 26) + ord('a')) for byte in digest_bytes[:output_length]])

#----------------------------------------------------------------------------
# Constructing a large command for latents pre-extraction before training.

def construct_latents_extraction_cmd(no_preext_exp: str, dst: os.PathLike, ae_batch_gpu: int, resolution: tuple[int, int, int], hydra_args_str: str, ignore: bool=False, kwargs: dict={}) -> str:
    extraction_cmd = " ".join([
        f"torchrun --max-restarts=1 --nproc_per_node=auto scripts/data_scripts/extract_latents.py {no_preext_exp}",
        f"dst={dst}",
        # Note that the FPS is taken from the experiment config defined by "desc".
        f"dataset.resolution=[{','.join([str(r) for r in resolution])}] dataset.batch_gpu={ae_batch_gpu} dataset.random_offset=false dataset.shuffle=false",
        f"dataset.print_exceptions=true dataset.print_traceback=true",
        cfg_to_args_str(kwargs, use_dashes=False),
        hydra_args_str,
    ])

    if ignore:
        extraction_cmd = "# " + extraction_cmd + " # Ignored."

    return extraction_cmd

#----------------------------------------------------------------------------
