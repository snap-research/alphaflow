import os
import re
import gc
import contextlib
import warnings
from functools import partial
from collections import OrderedDict
from typing import Any, Iterable, Optional, Union, Callable, ContextManager, Sequence


import numpy as np
import torch
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import einops
from fvcore.nn import FlopCountAnalysis

from src.structs import EasyDict, TensorGroup
from src.structs import ResumeConfig, TensorLike, DTYPE_STRING_TO_DTYPE

#----------------------------------------------------------------------------
# Cached construction of constant tensors. Avoids CPU=>GPU copy when the
# same constant is used multiple times.

_constant_cache = dict()

def constant(value, shape=None, dtype=None, device=None, memory_format=None):
    value = np.asarray(value)
    if shape is not None:
        shape = tuple(shape)
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device('cpu')
    if memory_format is None:
        memory_format = torch.contiguous_format

    key = (value.shape, value.dtype, value.tobytes(), shape, dtype, device, memory_format)
    tensor = _constant_cache.get(key, None)
    if tensor is None:
        tensor = torch.as_tensor(value.copy(), dtype=dtype, device=device)
        if shape is not None:
            tensor, _ = torch.broadcast_tensors(tensor, torch.empty(shape))
        tensor = tensor.contiguous(memory_format=memory_format)
        _constant_cache[key] = tensor
    return tensor

def const_like(ref, value, shape=None, dtype=None, device=None, memory_format=None):
    """Variant of constant() that inherits dtype and device from the given reference tensor by default."""
    if dtype is None:
        dtype = ref.dtype
    if device is None:
        device = ref.device
    return constant(value, shape=shape, dtype=dtype, device=device, memory_format=memory_format)

#----------------------------------------------------------------------------

@contextlib.contextmanager
def suppress_tracer_warnings():
    """
    Context manager to temporarily suppress known warnings in torch.jit.trace().
    Note: Cannot use catch_warnings because of https://bugs.python.org/issue29672
    """
    flt = ('ignore', None, torch.jit.TracerWarning, None, 0)
    warnings.filters.insert(0, flt)
    yield
    warnings.filters.remove(flt)

#----------------------------------------------------------------------------

try:
    symbolic_assert = torch._assert # 1.8.0a0 # pylint: disable=protected-access
except AttributeError:
    symbolic_assert = torch.Assert # 1.7.0

def assert_shape(tensor, ref_shape):
    """
    Assert that the shape of a tensor matches the given list of integers.
    None indicates that the size of a dimension is allowed to vary.
    Performs symbolic assertion when used in torch.jit.trace().
    """
    if tensor.ndim != len(ref_shape):
        raise AssertionError(f'Wrong number of dimensions: got {tensor.ndim}, expected {len(ref_shape)}')
    for idx, (size, ref_size) in enumerate(zip(tensor.shape, ref_shape)):
        if ref_size is None:
            pass
        elif isinstance(ref_size, torch.Tensor):
            with suppress_tracer_warnings(): # as_tensor results are registered as constants
                symbolic_assert(torch.equal(torch.as_tensor(size), ref_size), f'Wrong size for dimension {idx}')
        elif isinstance(size, torch.Tensor):
            with suppress_tracer_warnings(): # as_tensor results are registered as constants
                symbolic_assert(torch.equal(size, torch.as_tensor(ref_size)), f'Wrong size for dimension {idx}: expected {ref_size}')
        elif size != ref_size:
            raise AssertionError(f'Wrong size for dimension {idx}: got {size}, expected {ref_size}')

#----------------------------------------------------------------------------
# Some profiling utils.

def profiled_function(fn):
    """Function decorator that calls torch.profiler.record_function()."""
    def decorator(*args, **kwargs):
        with torch.profiler.record_function(fn.__name__):
            return fn(*args, **kwargs)
    decorator.__name__ = fn.__name__
    return decorator

def init_profiler(profiling_logs_dir: os.PathLike) -> None:
    return torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=2, active=2, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profiling_logs_dir),
        record_shapes=True,
        # profile_memory=True, # Results in errors in TensorBoard.
        # with_flops=True,
        with_stack=False, # TensorBoard does not like it either :|
    )

#----------------------------------------------------------------------------
# Gradient clieaning utils.

def grad_nan_to_num_(named_parameters: Iterable[tuple[str, torch.Tensor]], return_stats: bool=False) -> EasyDict:
    stats = EasyDict()
    for name, param in named_parameters:
        if param.grad is not None:
            if return_stats:
                num_nans = torch.isnan(param.grad).sum().item()
                num_infs = torch.isinf(param.grad).sum().item()
                ratio_nans = (num_nans / param.numel()) if param.numel() > 0 else 0
                ratio_infs = (num_infs / param.numel()) if param.numel() > 0 else 0
                stats[name] = EasyDict(num_nans=num_nans, num_infs=num_infs, num_params=param.numel(), ratio_nans=ratio_nans, ratio_infs=ratio_infs)
            inf_clamp = 1e5 if param.dtype == torch.float32 else 1e3
            torch.nan_to_num(param.grad, nan=0, posinf=inf_clamp, neginf=-inf_clamp, out=param.grad)

    if return_stats:
        assert 'total' not in stats, "`total` key is reserved in stats"
        stats.total = EasyDict(
            num_nans=sum(s.num_nans for s in stats.values()),
            num_infs=sum(s.num_infs for s in stats.values()),
            num_params=sum(s.num_params for s in stats.values()),
        )
        stats.total.ratio_nans = (stats.total.num_nans / stats.total.num_params) if stats.total.num_params > 0 else 0
        stats.total.ratio_infs = (stats.total.num_infs / stats.total.num_params) if stats.total.num_params > 0 else 0

    return stats

try:
    nan_to_num = torch.nan_to_num # 1.8.0a0
except AttributeError:
    def nan_to_num(input, nan=0.0, posinf=None, neginf=None, *, out=None): # pylint: disable=redefined-builtin
        """Replace NaN/Inf with specified numerical values."""
        assert isinstance(input, torch.Tensor)
        if posinf is None:
            posinf = torch.finfo(input.dtype).max
        if neginf is None:
            neginf = torch.finfo(input.dtype).min
        assert nan == 0
        return torch.clamp(input.unsqueeze(0).nansum(0), min=neginf, max=posinf, out=out)

#----------------------------------------------------------------------------
# Print summary table of module hierarchy.
# TODO: prints gibberish for FSDP modules.

@torch.no_grad()
def print_module_summary(module, inputs, max_nesting=4, skip_redundant=True, module_kwargs={}, verbose: bool=True): # pylint: disable=dangerous-default-value
    assert isinstance(module, torch.nn.Module)
    assert not isinstance(module, torch.jit.ScriptModule)
    assert isinstance(inputs, (tuple, list)), f"Expected inputs to be tuple or list, got {type(inputs)}"

    entries = []
    nesting = [0]

    def pre_hook(_mod, _inputs):
        nesting[0] += (1 if not isinstance(_mod, FSDP) else 0)

    def post_hook(mod, module_inputs, outputs):
        nesting[0] -= (1 if not isinstance(mod, FSDP) else 0)
        if nesting[0] <= max_nesting:
            module_inputs = [t for t in module_inputs if isinstance(t, torch.Tensor)]
            outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
            outputs = [t for t in outputs if isinstance(t, torch.Tensor)]
            entries.append(EasyDict(mod=mod, inputs=module_inputs, outputs=outputs))

    hooks = [mod.register_forward_pre_hook(pre_hook) for mod in module.modules()]
    hooks += [mod.register_forward_hook(post_hook) for mod in module.modules()]

    # Run model.
    is_training = module.training
    module.eval() # To make sure that batch norm layers are not updated
    outputs = module(*inputs, **module_kwargs)
    module.train(is_training)

    for hook in hooks:
        hook.remove()

    # Identify unique outputs, parameters, and buffers.
    tensors_seen = set()
    for e in entries:
        e.unique_params = [t for t in e.mod.parameters() if id(t) not in tensors_seen]
        e.unique_buffers = [t for t in e.mod.buffers() if id(t) not in tensors_seen]
        e.unique_outputs = [t for t in e.outputs if id(t) not in tensors_seen]
        tensors_seen |= {id(t) for t in e.unique_params + e.unique_buffers + e.unique_outputs}

    # Filter out redundant entries.
    if skip_redundant:
        entries = [e for e in entries if len(e.unique_params) or len(e.unique_buffers) or len(e.unique_outputs)]

    # Calculate FLOPs per submodule
    module_flops = {}
    for e in entries:
        try:
            if len(e.inputs) > 0:
                with open(os.devnull, 'w') as devnull:
                    with contextlib.redirect_stderr(devnull):
                        flops = FlopCountAnalysis(e.mod, tuple(e.inputs)).total()
                module_flops[e.mod] = flops
            else:
                module_flops[e.mod] = 0
        except:
            module_flops[e.mod] = 0

    rows = [[type(module).__name__, 'Parameters', 'Buffers', 'Input Shape', 'Output Shape', 'Datatype', 'GFLOPs']]
    rows += [['---'] * len(rows[0])]
    param_total = buffer_total = flops_total = 0
    submodule_names = {mod: name for name, mod in module.named_modules()}

    for e in entries:
        name = '<top-level>' if e.mod is module else re.sub(r'\.?_fsdp_wrapped_module\.?', '.', submodule_names[e.mod]).lstrip('.')
        param_size = sum(t.numel() for t in e.unique_params)
        buffer_size = sum(t.numel() for t in e.unique_buffers)
        input_shapes = ' + '.join([str(list(t.shape)) for t in e.inputs]) or '-'
        output_shapes = [str(list(t.shape)) for t in e.outputs]
        output_dtypes = [str(t.dtype).split('.')[-1] for t in e.outputs]
        flops = module_flops.get(e.mod, 0)
        gflops = f'{flops / 1e9:.3f}' if flops else '-'

        rows += [[
            name + (':0' if len(e.outputs) >= 2 else ''),
            f'{param_size:,}' if param_size else '-',
            f'{buffer_size:,}' if buffer_size else '-',
            input_shapes,
            (output_shapes + ['-'])[0],
            (output_dtypes + ['-'])[0],
            gflops,
        ]]
        for idx in range(1, len(e.outputs)):
            rows += [[name + f':{idx}', '-', '-', '-', output_shapes[idx], output_dtypes[idx], '-']]
        param_total += param_size
        buffer_total += buffer_size
        flops_total += flops

    rows += [['---'] * len(rows[0])]
    total_gflops_str = f'{flops_total / 1e9:.3f}'
    rows += [['Total', f'{param_total:,}', f'{buffer_total:,}', '-', '-', '-', total_gflops_str]]
    row_lengths = [len(r) for r in rows]
    assert len(set(row_lengths)) == 1, f"Summary table contains rows of different lengths: {row_lengths}"

    # Print table.
    widths = [max(len(cell) for cell in column) for column in zip(*rows)]
    if verbose:
        print()
        for row in rows:
            print('  '.join(cell.ljust(width) for cell, width in zip(row, widths)))
        print()

    return outputs

#----------------------------------------------------------------------------

def print_stats(prefix: str=None, x: Any=None, module_detailed: bool=False):
    sep = '' if prefix is None else ' '
    if x is None:
        print(f'{prefix}{sep}{x}')
    elif isinstance(x, torch.Tensor):
        dtype = x.dtype
        x_fp64 = x.detach().double()
        print(f'{prefix}{sep}avg: {x_fp64.mean().item()} | std: {x_fp64.std().item()} | min: {x_fp64.min().item()} | max: {x_fp64.max().item()} | shape: {list(x_fp64.shape)} | dtype: {dtype} | requires_grad: {x.requires_grad}')
    elif isinstance(x, torch.nn.Module):
        if module_detailed:
            for name, p in x.named_parameters():
                print_stats(f'{prefix}{sep}[{name}]', p)
        else:
            print_stats(prefix, torch.cat([p.view(-1) for p in x.parameters()]).double())
    elif isinstance(x, (float, int)):
        print(f'{prefix}{sep}[{type(x).__name__}]{x}')
    else:
        raise NotImplementedError(f"Uknown type: {type(x)}")

#----------------------------------------------------------------------------
# Model/optimizer state management utils.

def params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.parameters()) + list(module.buffers())

def params_and_buffers_vector(module):
    return torch.cat([p.view(-1) for p in params_and_buffers(module)])

def grad_vector(module):
    return torch.cat([p.grad.view(-1) for p in module.parameters() if p.grad is not None])

def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())

@torch.no_grad()
def copy_params_and_buffers(src, dst, require_all: bool=False, verbose: bool=False):
    if isinstance(src, torch.nn.Parameter):
        assert isinstance(dst, torch.nn.Parameter), f"Wrong type: {type(dst)}"
        dst.data.copy_(src.data)
    elif isinstance(src, torch.nn.Module):
        assert isinstance(dst, torch.nn.Module), f"Wrong type: {type(src)}"
        src_tensors = dict(named_params_and_buffers(src))
        trg_tensors = dict(named_params_and_buffers(dst))
        extra_keys = [k for k in src_tensors if not k in trg_tensors]
        if len(extra_keys) > 0 and verbose:
            print('extra keys:', extra_keys)
        for name, tensor in trg_tensors.items():
            assert (name in src_tensors) or (not require_all), f"{name} is missing among source tensors. Set require_all=False to suppress."
            if name in src_tensors:
                try:
                    tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad)
                except:
                    print('Couldnt copy', name)
                    raise
    elif src is None:
        assert dst is None or not require_all, f"Hyperaparameters mismatch: {type(dst)}"
    else:
        raise TypeError(f"Wrong type: {type(src)}")

def optimizer_to(optim, device):
    """Copy-pasted from https://github.com/pytorch/pytorch/issues/8741#issuecomment-402129385"""
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
    return optim

def load_state_dict_(module: torch.nn.Module, state_dict: dict[str, torch.Tensor], resume_cfg: ResumeConfig, module_log_name: str="", verbose: bool=False) -> torch.nn.Module:
    # Remapping the keys if needed.
    if resume_cfg.keys_aug is not None:
        state_dict.update({key_to: state_dict[key_from] for key_from, key_to in resume_cfg.keys_aug})

    # Removing the mismatched keys from the state dict.
    if resume_cfg.allow_shape_mismatch:
        cur_state_dict = module.state_dict()
        mismatched_keys = {k: (tuple(cur_state_dict[k].shape), tuple(v.shape)) for k, v in state_dict.items() if k in cur_state_dict and cur_state_dict[k].shape != v.shape}
        state_dict = {k: v for k, v in state_dict.items() if not k in mismatched_keys}
        if verbose:
            print(f"Shape mismatch in {module_log_name}: {mismatched_keys}. Loading state dict with shape mismatch.")

    # Loading the state.
    incompatible_keys = unwrap_module(module).load_state_dict(state_dict, strict=resume_cfg.is_strict_loading())

    # Checking what we loaded.
    missing_keys = incompatible_keys.missing_keys if resume_cfg.allow_missing_keys_regex is None else [k for k in incompatible_keys.missing_keys if not re.compile(resume_cfg.allow_missing_keys_regex).match(k)]
    assert len(missing_keys) == 0 or resume_cfg.allow_missing_keys, \
        f'Missing keys when loading "{module_log_name}" from "{resume_cfg.ckpt_path}": {missing_keys}. `allow_missing_keys` should be True to ignore (for resume or resume_from_start) or resume_cfg.allow_missing_keys_regex={resume_cfg.allow_missing_keys_regex} should filter them.'
    assert len(incompatible_keys.unexpected_keys) == 0 or resume_cfg.allow_unexpected_keys, \
        f'Unexpected keys when loading "{module_log_name}" from "{resume_cfg.ckpt_path}": {incompatible_keys.unexpected_keys}. `allow_unexpected_keys` should be True to ignore (for resume or resume_from_start).'
    return incompatible_keys

#----------------------------------------------------------------------------
# Profiled/slightly generalized versions of some common einops/pytorch functions.

@profiled_function
def rearrange(x: Any, *args, **kwargs):
    """A profiled version of einops.rearrange."""
    if isinstance(x, TensorGroup):
        return TensorGroup(**{k: einops.rearrange(v, *args, **kwargs) for k, v in x.items()})
    else:
        return einops.rearrange(x, *args, **kwargs)

@profiled_function
def reduce(*args, **kwargs):
    """A profiled version of einops.reduce."""
    return einops.reduce(*args, **kwargs)

@profiled_function
def repeat(*args, **kwargs):
    """A profiled version of einops.repeat."""
    return einops.repeat(*args, **kwargs)

@profiled_function
def grid_sample(*args, **kwargs):
    """A profiled version of F.grid_sample."""
    return F.grid_sample(*args, **kwargs)

def randn_like(x: TensorLike, *args, **kwargs) -> TensorLike:
    return TensorGroup.map(x, torch.randn_like, *args, **kwargs) if isinstance(x, TensorGroup) else torch.randn_like(x, *args, **kwargs)

def empty_like(x: TensorLike, *args, **kwargs) -> TensorLike:
    return TensorGroup.map(x, torch.empty_like, *args, **kwargs) if isinstance(x, TensorGroup) else torch.empty_like(x, *args, **kwargs)

def ones_like(x: TensorLike, *args, **kwargs) -> TensorLike:
    return TensorGroup.map(x, torch.ones_like, *args, **kwargs) if isinstance(x, TensorGroup) else torch.ones_like(x, *args, **kwargs)

def zeros_like(x: TensorLike, *args, **kwargs) -> TensorLike:
    return TensorGroup.map(x, torch.zeros_like, *args, **kwargs) if isinstance(x, TensorGroup) else torch.zeros_like(x, *args, **kwargs)

def randn(shape: Sequence[int] | torch.Size | dict[str, Any], *args, **kwargs) -> torch.Tensor:
    if isinstance(shape, dict):
        return TensorGroup({k: torch.randn(v, *args, **kwargs) for k, v in shape.items()})
    return torch.randn(shape, *args, **kwargs)

def stack(tensors: Sequence[TensorLike], *args, **kwargs) -> TensorLike:
    return TensorGroup.stack(tensors, *args, **kwargs) if isinstance(tensors[0], TensorGroup) else torch.stack(tensors, *args, **kwargs)

def cat(tensors: Sequence[TensorLike], *args, **kwargs) -> TensorLike:
    return TensorGroup.cat(tensors, *args, **kwargs) if isinstance(tensors[0], TensorGroup) else torch.cat(tensors, *args, **kwargs)

#----------------------------------------------------------------------------
# Random pytorch utils.

def safe_shape(x: Optional[torch.Tensor]) -> Union[None, torch.Size]:
    return None if x is None else x.shape

def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def unsqueeze_right(src: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
    # Adds dummy dimensions to the `src` tensor so it matches dimensions with `trg`
    # Useful for (implicit) broadcasting afterwards.
    assert all(d_src == d_trg or d_src == 1 for d_src, d_trg in zip(src.shape, trg.shape)), f"Shape prefixes do not align: {src.shape} vs {trg.shape}"
    return src.view(*src.shape, *([1] * (trg.ndim - src.ndim)))

def unsqueeze_left(src: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
    assert all(d_src == d_trg or d_src == 1 for d_src, d_trg in zip(src.shape[::-1], trg.shape[::-1])), f"Shape suffixes do not align: {src.shape} vs {trg.shape}"
    return src.view(*([1] * (trg.ndim - src.ndim)), *src.shape)

def maybe_checkpoint(flag: bool, layer: torch.nn.Module, *args, use_reentrant=None, **kwargs) -> Any:
    return torch.utils.checkpoint.checkpoint(layer, *args, use_reentrant=use_reentrant, **kwargs) if flag else layer(*args, **kwargs)

def to(*tensors, **kwargs):
    if len(tensors) == 1:
        return tensors[0].to(**kwargs)
    else:
        return tuple([t.to(**kwargs) for t in tensors])

def safe_gmean(x: torch.Tensor, *args, **kwargs):
    return x.log().mean(*args, **kwargs).exp()

def int_prod(values: Iterable) -> int:
    return int(np.prod(values))

def size(x: torch.Tensor, dim: int | Iterable[int]) -> int | tuple[int]:
    return x.size(dim) if isinstance(dim, int) else tuple(x.size(d) for d in dim)

@contextlib.contextmanager
def disable_grad(tensors: Iterable[torch.Tensor]):
    requires_grads = [t.requires_grad for t in tensors]
    for t in tensors:
        t.requires_grad = False
    yield
    for t, requires_grad in zip(tensors, requires_grads):
        t.requires_grad = requires_grad

def zero_grad(tensors: Iterable[torch.Tensor]):
    for t in tensors:
        t.grad = None

def full_tensor(x: Any) -> Any:
    return x.full_tensor() if hasattr(x, 'full_tensor') else x

def is_strictly_broadcastable(x: TensorLike, y: TensorLike) -> bool:
    """
    Checks if `y` is strictly broadcastable w.r.t. `x`.
    The logic is not fully correct actually..
    """
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        return len(x) == len(y) and x.ndim == y.ndim
    elif isinstance(x, TensorGroup) and isinstance(y, TensorGroup):
        assert x.keys() == y.keys(), "Keys do not match"
        return all(is_strictly_broadcastable(xk, yk) for xk, yk in zip(x.values(), y.values()))
    elif isinstance(x, TensorGroup) and isinstance(y, torch.Tensor):
        return all(is_strictly_broadcastable(xk, y) for xk in x.values())
    else:
        return False

#----------------------------------------------------------------------------
# DDP/FSDP utils.

def unwrap_module(module: torch.nn.Module) -> torch.nn.Module:
    assert isinstance(module, torch.nn.Module), f"Unsupported type: {type(module)}"
    # Somehow, FSDP wrapping does not break the access to model attributes and methods.
    return module.module if isinstance(module, torch.nn.parallel.DistributedDataParallel) else module

@contextlib.contextmanager
def summon_full_params(*args, **kwargs):
    """
    A context manager proxy so that we do not need to have a scary FSDP import everywhere
    (apparently, they seems to be including some bizarre torch.jit compilations)
    """
    with FSDP.summon_full_params(*args, **kwargs) as ctx:
        yield ctx

def set_fsdp_wrap_recursively_(module: torch.nn.Module, should_fsdp_wrap: bool) -> None:
    # Set's the `should_fsdp_wrap` flag for all modules in the hierarchy.
    for submodule in module.modules():
        if submodule is module:
            continue
        if isinstance(submodule, torch.nn.Module):
            set_fsdp_wrap_recursively_(submodule, should_fsdp_wrap)
    module.should_fsdp_wrap = should_fsdp_wrap

def check_ddp_consistency(model, ignore_regex=None, prefix: str='', atol: float=0.0, rtol: float=0.0):
    assert isinstance(model, torch.nn.Module), f"Unsupported type: {type(model)}"
    with summon_full_params(model):
        for name, tensor in named_params_and_buffers(model):
            fullname = type(model).__name__ + '.' + name
            if ignore_regex is not None and re.fullmatch(ignore_regex, fullname):
                continue
            lhs = tensor.detach()
            if lhs.is_floating_point():
                lhs = nan_to_num(lhs)
            rhs = lhs.clone()
            torch.distributed.broadcast(tensor=rhs, src=0)

            if not torch.allclose(lhs, rhs, atol=atol, rtol=rtol):
                error_msg = f'{prefix}{fullname} is not consistent across processes. Norm difference: {torch.norm(lhs.float() - rhs.float())}. LHS norm: {torch.norm(lhs.float())}. RHS norm: {torch.norm(rhs.float())}.' \
                                f'Num different elements: {(lhs != rhs).sum()}. Dtypes: {lhs.dtype}, {rhs.dtype}. Shapes: {lhs.shape}, {rhs.shape}'
                raise ValueError(error_msg)

@contextlib.contextmanager
def maybe_ddp_sync(*modules, force_sync=True):
    """Context manager for easily enabling/disabling DistributedDataParallel synchronization."""
    if force_sync:
        yield # Yielding with sync because we need to sync.
    elif not any(isinstance(m, torch.nn.parallel.DistributedDataParallel) for m in modules):
        yield # Yielding with sync because the module is not parallelized.
    else:
        with contextlib.ExitStack() as stack:
            for m in modules:
                if m is not None and isinstance(m, torch.nn.parallel.DistributedDataParallel):
                    stack.enter_context(m.no_sync())
            yield

def replace_parameters_with_buffers_(module: torch.nn.Module) -> torch.nn.Module:
    """Turns all parameters of a module into buffers."""
    modules = module.modules()
    module = next(modules)
    for name, param in OrderedDict(module.named_parameters(recurse=False)).items():
        delattr(module, name) # Unregister parameter
        module.register_buffer(name, param.data)
    for submodule in modules:
        replace_parameters_with_buffers_(submodule)
    return module

def find_submodule(parent_module: torch.nn.Module, module_name: str) -> torch.nn.Module:
    submodule = find_node(parent_module, module_name)
    assert isinstance(submodule, torch.nn.Module), f"{module_name} is not a module, but a {type(submodule)}"
    return submodule

def recursive_tensor_map(x: Any, fn: Callable[[torch.Tensor], torch.Tensor]) -> Any:
    if isinstance(x, torch.Tensor):
        return fn(x)
    elif isinstance(x, (list, tuple)):
        return type(x)(recursive_tensor_map(v, fn) for v in x)
    elif isinstance(x, dict):
        return {k: recursive_tensor_map(v, fn) for k, v in x.items()}
    else:
        raise ValueError(f"Unsupported container type: {type(x)}")

def recursive_detach(x: Any) -> Any:
    return recursive_tensor_map(x, lambda t: t.detach())

def recursive_add_noise(x: Any, std: float) -> Any:
    return recursive_tensor_map(x, lambda t: t + torch.randn_like(t) * std)

def register_activations_collection_hooks(net, submodule_names: list[str]=None, inputs_swap: dict[str, Any]=None, detach_acts: bool=False, input_noise_std: float=0.0) -> tuple[dict, dict, dict]:
    storage = {} # To store output activations from each submodule.

    def filter_tensor_list(x) -> list[torch.Tensor]:
        x = list(x) if isinstance(x, (tuple, list)) else [x]
        x = [t for t in x if isinstance(t, torch.Tensor)]
        return x

    def post_hook(module, inputs, outputs, module_name: str):
        assert module_name not in storage, f"Module name {module_name} has already been saved in storage."
        _ = module # Unused.
        storage[module_name] = (inputs, filter_tensor_list(outputs)[0])
        return recursive_detach(outputs) if detach_acts else outputs

    def pre_hook(module, inputs, module_name: str, input_noise_std: float):
        _ = module # Unused.
        # We want to swap the inputs of the current executation with the provided ones.
        # This is useful for a) injecting noise or b) swapping the inputs of a student with the ones from the teacher.
        if inputs_swap is not None:
            assert module_name in inputs_swap, f"Module name {module_name} not found in inputs_swap."
            inputs = inputs_swap[module_name]
        if input_noise_std > 0:
            inputs = recursive_add_noise(inputs, input_noise_std)
        return inputs

    submodule_names: list[str] = list(k for k, _ in net.named_modules()) if submodule_names is None else submodule_names
    param_counts = sum(count_parameters(find_submodule(net, k)) for k in submodule_names)
    net_param_counts = count_parameters(net)
    assert param_counts == net_param_counts, f"Parameter count mismatch: {param_counts} vs {net_param_counts}"
    pre_hooks = {k: find_submodule(net, k).register_forward_pre_hook(partial(pre_hook, module_name=k, input_noise_std=input_noise_std)) for k in submodule_names} if inputs_swap is not None or input_noise_std > 0 else {}
    post_hooks = {k: find_submodule(net, k).register_forward_hook(partial(post_hook, module_name=k)) for k in submodule_names}

    return pre_hooks, post_hooks, storage

#----------------------------------------------------------------------------

class StackedRandomGenerator:
    """
    Wrapper for torch.Generator that allows specifying a different random seed for each sample in a minibatch.
    """
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, x: torch.Tensor):
        return self.randn(x.shape, dtype=x.dtype, layout=x.layout, device=x.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

    def multinomial(self, probs, num_samples=1):
        assert probs.shape[0] == len(self.generators)
        return torch.stack([torch.multinomial(probs[i], num_samples, generator=gen) for i, gen in enumerate(self.generators)])

def randn42(*args, **kwargs) -> torch.Tensor:
    return torch.from_numpy(np.random.RandomState(42).randn(*args, **kwargs))

def rand42(*args, **kwargs) -> torch.Tensor:
    return torch.from_numpy(np.random.RandomState(42).rand(*args, **kwargs))

#----------------------------------------------------------------------------
# Image/video processing utils.

def pad_t(x: torch.Tensor, pad_size: int, mode: str):
    assert x.ndim == 5, f"Expected 5D tensor of [b, c, t, h, w], got {x.shape}"
    if pad_size == 0:
        return x # [b, c, t, h, w]
    if mode == 'replicate' and x.dtype == torch.bfloat16:
        # Doing this manually since it does not support bfloat16.
        return torch.cat([x[:, :, :1].expand(-1, -1, pad_size, -1, -1), x], dim=2) # [b, c, 1 + t, h, w]
    else:
        return torch.nn.functional.pad(x, pad=[0, 0, 0, 0, pad_size, 0], mode=mode, value=0.0) # [b, c, pad + f, h, w] pylint: disable=not-callable

def unpad_t(x: torch.Tensor, pad_size: int):
    """Removes the first `pad_size` frames from the temporal dimension of a 5D tensor."""
    assert x.ndim == 5, f"Expected 5D tensor of [b, c, t, h, w], got {x.shape}"
    if pad_size == 0:
        return x # [b, c, t, h, w]
    return x[:, :, pad_size:] # [b, c, t - pad, h, w]

def video_2d_interpolate(x: torch.Tensor, **kwargs) -> torch.Tensor:
    assert x.ndim == 5, f"Expected 5D tensor, got {x.ndim}"
    batch_size = x.shape[0]
    x = rearrange(x, 'b t c h w -> (b t) c h w') # [b * t, c, h, w]
    x = F.interpolate(x, **kwargs) # [b * t, c, h, w]
    x = rearrange(x, '(b t) c h w -> b t c h w', b=batch_size) # [b, t, c, h, w]
    return x

def maybe_rescale_uint8_to_fp32(x: torch.Tensor, always: bool=False):
    """
    Rescales [0, 255] uint8 videos to fp32 [-1, 1] range. Ignores when the data is already float.
    This implicitly omits normalization of latent datasets (TODO: non-processing latents implicitly might be a bad design choice...).
    """
    return x.float().mul_(2 / 255).sub_(1) if always or x.dtype == torch.uint8 else x

def maybe_rescale_fp32_to_uint8(x: torch.Tensor, always: bool=False):
    """Rescales fp32 videos from [-1, 1] to uint8 [0, 255] range."""
    return x.add_(1).mul_(255 / 2).clamp_(0, 255).byte() if always or x.dtype in (torch.float64, torch.float32, torch.float16, torch.bfloat16) else x

#----------------------------------------------------------------------------
# Random python utils.

def partial_last(func: Callable, last_arg: Any):
    def wrapped(*args, **kwargs):
        return func(*args, last_arg, **kwargs)
    return wrapped

def free_memory():
    gc.collect()
    torch.cuda.empty_cache()

def all_not_none(values: Iterable[Optional[Any]]) -> bool:
    return all(v is not None for v in values)

def find_node(struct: Any, path: str, separator: str='.') -> Any:
    """
    Finds a node in a nested structure by a path string.
    Useful for finding modules/parameters/buffers in a model.
    """
    cur_node = struct
    path_chunks = path.split(separator)
    for chunk in path_chunks:
        cur_node = getattr(cur_node, chunk)
    return cur_node

def filter_nones(d: EasyDict | dict) -> EasyDict | dict:
    container_cls = EasyDict if isinstance(d, EasyDict) else dict
    return container_cls(**{k: v for k, v in d.items() if v is not None})

#----------------------------------------------------------------------------
# Gradient computation utils.

def compute_param_grad(loss: torch.Tensor, param: torch.nn.Parameter) -> torch.Tensor:
    return torch.autograd.grad(outputs=loss, inputs=param, grad_outputs=torch.ones_like(loss), retain_graph=True)[0].detach()

def compute_param_grad_batched(loss: torch.Tensor, param: torch.nn.Parameter) -> torch.Tensor:
    """
    By default, torch.autograd.grad would compute a sum of gradients over the batch dimension (is_grads_batch=True does not help).
    Hence, this wrapper to compute the gradients for each sample in the batch separately.
    """
    return torch.stack([compute_param_grad(l, param) for l in loss])

#----------------------------------------------------------------------------
# Autocast utils.

def maybe_autocast(model_cfg: EasyDict) -> ContextManager:
    use_autocast = model_cfg.autocast and torch.cuda.is_available() and model_cfg.act_dtype in ['float16', 'bfloat16']
    autocast_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    autocast_ctx = torch.autocast(
        device_type=autocast_device,
        enabled=True,
        dtype=DTYPE_STRING_TO_DTYPE[model_cfg.act_dtype],
        cache_enabled=model_cfg.autocast_cache_enabled
    ) if use_autocast else contextlib.nullcontext() # Autocast is still applied even if enabled=False, so returning a null context.

    assert not (use_autocast and model_cfg.loss_scaling != 1), f"Loss scaling is not supported with autocast: {model_cfg.loss_scaling}."

    return autocast_ctx

#----------------------------------------------------------------------------
