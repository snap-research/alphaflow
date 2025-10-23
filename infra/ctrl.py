"""
Experiments control utilities.
"""

import os
import argparse
from typing import Sequence

from src.utils.os_utils import listdir_full_paths, parse_int_list

#----------------------------------------------------------------------------

EXPERIMENTS_DIR = './experiments'

#----------------------------------------------------------------------------

def execute_command(cmd: str, experiment_ids: Sequence[str]):
    assert cmd in ['stop', 'snapshot']
    all_experiments: list[os.PathLike] = listdir_full_paths(EXPERIMENTS_DIR)
    for exp_id in experiment_ids:
        matching_experiments = [e for e in all_experiments if f'{EXPERIMENTS_DIR}/{exp_id:04d}-' in e]
        assert len(matching_experiments) == 1, f'Too many experiments match: {matching_experiments}'
        with open(os.path.join(matching_experiments[0], 'state.json'), 'w') as f:
            state_str = f'{{"should_{cmd}": true}}'
            f.write(state_str)
        print(f'Added state {state_str} to {matching_experiments[0]}')

#----------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cmd', type=str)
    parser.add_argument('-e', '--experiment_ids', type=str)
    args = parser.parse_args()
    execute_command(args.cmd, parse_int_list(args.experiment_ids))

#----------------------------------------------------------------------------
