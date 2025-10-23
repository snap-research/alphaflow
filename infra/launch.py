"""
Run a __reproducible__ experiment on __allocated__ resources.
From a brief experiment description, generates a full experiment config, the training command and launches it.
"""
# import sys; sys.path.extend(['..', '.', 'src'])
import os
import socket
import argparse
from typing import Callable

from termcolor import colored
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from src.structs import EasyDict
from infra.utils import create_project_dir, recursive_instantiate, maybe_increment_filename, dict_to_hydra_overrides
from src.utils.os_utils import load_yaml, save_yaml
from src.train import train

#----------------------------------------------------------------------------

def compute_overrides(experiment_name: str, extra_overrides: list[str]=None, max_inheritance_depth: int=20) -> list[str]:
    assert max_inheritance_depth > 0, f"Reached the maximum inheritance depth for {experiment_name}."
    prefix = experiment_name.split('-')[0]
    experiments_cfgs = load_yaml(f"infra/experiments/experiments-{prefix}.yaml")

    assert experiment_name in experiments_cfgs, f"Unknown experiment name: {experiment_name}"
    overrides_cfg = experiments_cfgs[experiment_name]
    parents: list[str] | str = overrides_cfg.pop('_inherit_', [])
    parents = [parents] if isinstance(parents, str) else parents
    parent_overrides = [o for p in parents for o in compute_overrides(p, max_inheritance_depth=max_inheritance_depth - 1)]
    main_overrides = [f'{k}={v}' for k, v in overrides_cfg.items() if k != 'overrides']
    overrides = parent_overrides + main_overrides + [f'desc={experiment_name}'] + dict_to_hydra_overrides(overrides_cfg.get('overrides', {})) + (extra_overrides or [])

    return overrides

def launch_from_registry(fn: Callable, config_path: os.PathLike="../configs/train.yaml"): # pylint-ignore:redefined-outer-name
    """
    Executes the provided function using the config from the registry, provided as the first `experiment_name` argument (with potential overrides).
    """
    parser = argparse.ArgumentParser(description='Launch an experiment from the registry.')
    parser.add_argument('experiment_name', type=str, help='Name of the experiment to launch.')
    parser.add_argument('overrides', nargs=argparse.REMAINDER, help='Additional overrides for Hydra configuration')
    args = parser.parse_args()

    with initialize(config_path=os.path.dirname(config_path), version_base='1.2'):
        cfg = compose(os.path.basename(config_path), overrides=compute_overrides(args.experiment_name, args.overrides))
        fn(cfg)

#----------------------------------------------------------------------------

def launch(cfg: DictConfig):
    recursive_instantiate(cfg) # TODO: do we really need it?
    OmegaConf.set_struct(cfg, True)
    maybe_update_cfg_for_debug_run(cfg)

    cfg.env.project_path = str(cfg.env.project_path) # This is needed to evaluate ${hydra:runtime.cwd}
    is_running_from_scratch = determine_if_running_from_scratch(cfg)
    cfg = maybe_update_cfg_with_existing_exp(cfg, is_running_from_scratch)
    training_cmd = build_training_cmd(cfg)

    assert not (cfg.model.loss_scaling > 1 and cfg.model.autocast), f"Cannot use autocast with loss scaling: {cfg.model.loss_scaling}"

    if is_running_from_scratch and not cfg.print_only and not cfg.direct_launch:
        create_project_dir(
            cfg.experiment_dir,
            cfg.output_dir,
            cfg.env.objects_to_copy,
            cfg.env.symlinks_to_create,
            quiet=cfg.quiet_launch,
            ignore_uncommited_changes=cfg.ignore_uncommited_changes,
            overwrite=cfg.overwrite
        )

        with open(cfg.cfg_save_path, 'w') as f:
            OmegaConf.save(config=cfg, f=f)
            if not cfg.quiet_launch:
                print(f'Saved config in {cfg.cfg_save_path}')

    if cfg.direct_launch:
        cfg.experiment_dir = cfg.env.project_path
        os.makedirs(os.path.join(cfg.experiment_dir, 'output'), exist_ok=True)

    if not cfg.quiet_launch:
        print('<=== TRAINING COMMAND START ===>')
        print(training_cmd)
        print('<=== TRAINING COMMAND END ===>')

    if not cfg.print_only and not cfg.direct_launch:
        with open(cfg.training_cmd_save_path, 'w') as f:
            f.write(training_cmd + '\n')
            if not cfg.quiet_launch:
                print(f'Saved training command in {cfg.training_cmd_save_path}')

        os.chdir(cfg.experiment_dir)

    execute_training_cmd(cfg, training_cmd)

#----------------------------------------------------------------------------

def execute_training_cmd(cfg: DictConfig, training_cmd: str):
    if cfg.print_only:
        print(training_cmd)
    elif cfg.direct_launch:
        train(cfg)
    else:
        os.system(training_cmd)

#----------------------------------------------------------------------------

def maybe_update_cfg_with_existing_exp(cfg, is_running_from_scratch: bool) -> OmegaConf:
    if is_running_from_scratch:
        return cfg
    else:
        assert os.path.isfile(cfg.cfg_save_path), f"Couldn't find the config file: {cfg.cfg_save_path}"

    cfg_new = cfg
    cfg_old = OmegaConf.load(cfg_new.cfg_save_path)
    for k in ['num_nodes', 'num_gpus', 'quiet_launch', 'experiment_dir']:
        cfg_old[k] = cfg_new[k]
    cfg_old.training.max_steps = cfg_new.training.max_steps
    cfg_old.cfg_save_path = maybe_increment_filename(cfg_old.cfg_save_path)
    cfg_old.training_cmd_save_path = maybe_increment_filename(cfg_old.training_cmd_save_path)

    return cfg_old

#----------------------------------------------------------------------------

def build_training_cmd(cfg) -> str:
    python_paths_str = ':'.join([cfg.experiment_dir] + cfg.env.extra_python_paths)
    extra_env_vars: List[str] = [f'export {k}={v}' for k, v in cfg.wandb.env_args.items()] + [f'export PYTHONPATH={python_paths_str}']
    before_train_cmds_list: List[str] = extra_env_vars + cfg.env.before_train_commands + cfg.env.before_train_commands_common + [f'cd {cfg.experiment_dir}'] + [cfg.pre_extract_latents.cmd]
    before_train_cmd: str = '\n'.join(before_train_cmds_list)
    before_train_cmd: str = (before_train_cmd + '\n\n') if len(before_train_cmd) > 0 else ''
    executor = f'torchrun --monitor-interval 5 --max-restarts {cfg.max_restarts}{" --standalone" if cfg.standalone else ""} --nproc-per-node={cfg.num_gpus}' if cfg.num_gpus > 1 else cfg.env.python_bin
    training_cmd = f'{before_train_cmd}{executor} src/train.py {cfg.hydra_args_str}'

    return training_cmd

#----------------------------------------------------------------------------

def determine_if_running_from_scratch(cfg) -> bool:
    can_resume_training = os.path.isdir(cfg.experiment_dir) and os.path.isfile(cfg.training_cmd_save_path) and os.path.isfile(cfg.cfg_save_path) and not cfg.debug_run
    is_running_from_scratch = not can_resume_training

    if not cfg.quiet_launch:
        if is_running_from_scratch:
            print('Running from scratch...')
        else:
            print('Found an existing experiment. Will be resuming from it.')

    return is_running_from_scratch

#----------------------------------------------------------------------------

def maybe_update_cfg_for_debug_run(cfg):
    import torch # pylint: disable=import-outside-toplevel

    if cfg.debug_run:
        cfg.quiet_launch = False
        cfg.ignore_uncommited_changes = True
        cfg.overwrite = True
        cfg.training.seconds_per_tick = 30 if cfg.training.seconds_per_tick is not None else cfg.training.seconds_per_tick
        cfg.training.steps_per_tick = 2 if cfg.training.steps_per_tick is not None else cfg.training.steps_per_tick
        cfg.training.num_samples_for_per_sigma_loss = min(16, cfg.training.num_samples_for_per_sigma_loss) if isinstance(cfg.training.num_samples_for_per_sigma_loss, int) else cfg.training.num_samples_for_per_sigma_loss
        cfg.training.num_log_sigmas = min(3, cfg.training.num_log_sigmas) if isinstance(cfg.training.num_log_sigmas, int) else cfg.training.num_log_sigmas
        cfg.max_restarts = 0
        cfg.wandb.tags = cfg.wandb.tags + ['debug']
        cfg.sampling.num_steps = min(cfg.sampling.num_steps, 3)
        if 'num_steps_min' in cfg.sampling:
            cfg.sampling.num_steps_min = 3
        cfg.num_gpus = min(cfg.num_gpus, torch.cuda.device_count())
        if cfg.debug_run_rename:
            cfg.desc = f'debug-{socket.gethostname()}'
        cfg.experiment_dir = os.path.join(cfg.env.project_path, 'experiments', cfg.desc)

    if cfg.debug_no_freqs:
        for k in cfg.training.freqs:
            if k in cfg.debug_no_freqs_allow:
                continue
            cfg.training.freqs[k] = None

    if cfg.init_debugger:
        import debugpy
        debugpy.listen(5678)
        print("Waiting for debugger attach...")
        debugpy.wait_for_client()
        print("Debugger attached.")

#----------------------------------------------------------------------------

if __name__ == "__main__":
    launch_from_registry(launch)

#----------------------------------------------------------------------------
