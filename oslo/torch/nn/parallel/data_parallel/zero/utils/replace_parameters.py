from typing import Any, Dict, Iterator, Optional, Tuple, Union

import torch
from torch import nn

from oslo.torch.nn.parallel.data_parallel.zero.tensor import (
    DistributedParameter,
    DistributedTensor,
)
from oslo.torch.distributed.parallel_context import ParallelContext


# find named_params includes replica
def _named_params_with_replica(
    module: nn.Module,
    prefix: str = "",
    recurse: bool = True,
) -> Iterator[Tuple[str, Union[nn.Parameter, DistributedTensor]]]:
    """Get named parameters of a module, including replica parameters.

    Args:
        module (nn.Module): The module to get parameters from.
        prefix (str, optional): The prefix to prepend to all parameter names.
        recurse (bool, optional): If True, then yields parameters of this module

    Yields:
        Iterator[Tuple[str, Union[nn.Parameter, DistributedTensor]]]: The named parameters.
    """
    modules = module.named_modules(prefix=prefix) if recurse else [(prefix, module)]
    for mod_prefix, mod in modules:
        for name, val in mod._parameters.items():
            if val is None:
                continue
            name = mod_prefix + ("." if mod_prefix else "") + name
            yield name, val


def _convert_to_distparam(
    param: torch.nn.Parameter,
    device: torch.device,
    dtype: torch.dtype = torch.float,
    parallel_context: ParallelContext = None,
    dist_spec: Optional[Any] = None,
) -> DistributedParameter:
    """Convert a parameter to a DistributedParameter.

    Args:
        param (torch.nn.Parameter): The parameter to convert.
        device (torch.device): The device to place the parameter.
        dtype (torch.dtype, optional): The dtype of the parameter. Defaults to torch.float.
        parallel_context (ParallelContext, optional): The parallel context. Defaults to None.
        dist_spec (Optional[Any], optional): The default dist spec. Defaults to None.

    Returns:
        DistributedParameter: The converted DistributedParameter.
    """
    if type(param) is DistributedParameter:
        return param
    # detaching tensor is necessary for optimizers.
    requires_grad = param.requires_grad
    # param is the global tensor.
    if param.device.type == "meta":
        dist_param = DistributedParameter(param, requires_grad=requires_grad)
    else:
        dist_param = DistributedParameter(
            param.to(device=device, dtype=dtype), requires_grad=requires_grad
        )
    # if default_shard_plan exists, shard the param during initialization.
    # This can reduce the model size after initialization.
    # NOTE() embedding usually can not be correctly sharded. So I use except to handle
    # the param that can not be sharded by the default plan
    if parallel_context is not None:
        dist_param.set_parallel_context(parallel_context)

    if dist_spec is not None:
        try:
            dist_param.set_dist_spec(dist_spec)
        except:
            pass
    return dist_param


def replace_params_with_distributed(
    module: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float,
    parallel_context: ParallelContext = None,
    dist_spec: Optional[Any] = None,
):
    """
    Replace the parameters of a module with DistributedParameter.

    Args:
        module (torch.nn.Module): The module to replace parameters.
        device (torch.device): The device to place the parameters.
        dtype (torch.dtype, optional): The dtype of the parameters. Defaults to torch.float.
        parallel_context (ParallelContext, optional): The parallel context. Defaults to None.
        dist_spec (Optional[Any], optional): The default dist spec. Defaults to None.
    """
    name_list = []
    for name, param in _named_params_with_replica(module):
        if type(param) is DistributedParameter:
            continue

        split = name.rfind(".")
        if split >= 0:  # param in submodule
            module_name = name[:split]
            param_name = name[split + 1 :]
        else:
            module_name = ""  # param in current module
            param_name = name
        name_list.append((module_name, param_name))

    replaced_tensors = (
        dict()
    )  # record mapping between (torch.Tensor, ColoTensor) to distinguish the same reference
    for module_name, param_name in name_list:
        submodule = module.get_submodule(module_name)
        param = submodule.get_parameter(param_name)
        if param in replaced_tensors:
            dist_param = replaced_tensors[param]
        else:
            dist_param = _convert_to_distparam(
                param, device, dtype, parallel_context, dist_spec
            )
            replaced_tensors[param] = dist_param
        delattr(submodule, param_name)
        setattr(submodule, param_name, dist_param)
        dist_param.shared_param_modules.append(submodule)

    param_number = 0
    meta_param_number = 0
    buffer_number = 0
    meta_buffer_number = 0

    for param in module.parameters():
        param_number += 1
        meta_param_number += param.device.type == "meta"

    for buffer in module.buffers():
        buffer_number += 1
        meta_buffer_number += buffer.device.type == "meta"

    if meta_param_number > 0 and meta_param_number != param_number:
        raise ValueError(
            "Meta parameters and valued parameters can not  be in the same model"
        )
    if meta_buffer_number > 0 and meta_buffer_number != buffer_number:
        raise ValueError("Meta buffers and valued buffers can not be in the same model")

    if meta_buffer_number == 0:
        for buffer in module.buffers():
            buffer.data = buffer.data.to(device=dtype)
