from typing import Any
from src.structs import EasyDict

#----------------------------------------------------------------------------

def legacy_patch_experiment_cfg(experiment_cfg: EasyDict | dict[str, Any]) -> EasyDict:
    # Here, we can patch a legacy experiment config to be compatible with the latest code.
    return experiment_cfg

#----------------------------------------------------------------------------
