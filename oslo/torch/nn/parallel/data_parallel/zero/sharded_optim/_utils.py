from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch.distributed as dist
from torch._six import inf
import math
import torch
from typing import Optional

# TODO
def is_model_parallel_parameter(p):
    return False


def flatten(input_):
    return _flatten_dense_tensors(input_)


def unflatten(flat, tensors):
    return _unflatten_dense_tensors(flat, tensors)


def calculate_global_norm_from_list(norm_list):
    """Compute total from a list of norms"""
    total_norm = 0.0
    for norm in norm_list:
        total_norm += norm ** 2.0
    return math.sqrt(total_norm)


def reduce_tensor_dp_group(
    tensor: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
    dst_local_rank: Optional[int] = None,
    dst_global_rank: Optional[int] = None,
    group: Optional[dist.ProcessGroup] = None,
):
    """
    Reduce the tensor in the data parallel process group
    :param tensor: A tensor object to reduce/all-reduce
    :param dtype: The data type used in communication
    :param dst_rank: The source rank for reduce. If dst_rank is None,
    :param parallel_mode: Communication parallel mode
    all-reduce will be used instead of reduce. Default is None.
    :type tensor: torch.Tensor
    :type dtype: torch.dtype, optional
    :type dst_rank: int, optional
    :type pg: ProcessGroup, optional
    """
    # use the original dtype
    if dtype is None:
        dtype = tensor.dtype

    # cast the data to specified dtype for reduce/all-reduce
    if tensor.dtype != dtype:
        tensor_to_reduce = tensor.to(dtype)
    else:
        tensor_to_reduce = tensor

    world_size = dist.get_world_size(group=group)
    tensor_to_reduce.div_(world_size)

    # if rank is None, all reduce will be used
    # else, reduce is used
    use_all_reduce = dst_local_rank is None

    if use_all_reduce:
        dist.all_reduce(tensor_to_reduce, group=group)
    else:
        dist.reduce(tensor=tensor_to_reduce, dst=dst_global_rank, group=group)

    # recover the original dtype
    if tensor.dtype != dtype and tensor is not tensor_to_reduce:
        local_rank = dist.get_rank(group=group)
        if use_all_reduce or dst_local_rank == local_rank:
            tensor.copy_(tensor_to_reduce)

    return tensor


def has_inf_or_nan(tensor):
    try:
        # if tensor is half, the .float() incurs an additional deep copy, but it's necessary if
        # Pytorch's .sum() creates a one-element tensor of the same type as tensor
        # (which is true for some recent version of pytorch).
        tensor_sum = float(tensor.float().sum())
        # More efficient version that can be used if .sum() returns a Python scalar
        # tensor_sum = float(tensor.sum())
    except RuntimeError as instance:
        # We want to check if inst is actually an overflow exception.
        # RuntimeError could come from a different error.
        # If so, we still want the exception to propagate.
        if "value cannot be converted" not in instance.args[0]:
            raise
        return True
    else:
        if (
            tensor_sum == float("inf")
            or tensor_sum == -float("inf")
            or tensor_sum != tensor_sum
        ):
            return True
        return False


def release_param_grad(tensor_list):
    for tensor in tensor_list:
        tensor.grad = None


def compute_norm(gradients, params, dp_group, mp_group, norm_type=2):
    """Clips gradient norm of an iterable of parameters.
    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters. Note that
    the gradients are modified in place.
    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
    Returns:
        Total norm of the parameters (viewed as a single vector).
    """

    if mp_group is None:
        mp_rank = 0
    else:
        mp_rank = dist.get_rank(mp_group)

    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(g.data.abs().max() for g in gradients)
        total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])
        dist.all_reduce(
            total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=dp_group
        )

        # Take max across all GPUs.
        if mp_group is not None:
            dist.all_reduce(tensor=total_norm_cuda, op=torch.distributed.ReduceOp.MAX)
        total_norm = total_norm_cuda[0].item()
    else:
        total_norm = 0.0
        # if dist.get_rank() == 0:
        #    logger.info(f"Total Norm beginning {total_norm}")

        for g, p in zip(gradients, params):
            # Pipeline parallelism may replicate parameters. Avoid multi-counting.
            if is_model_parallel_parameter(p) or mp_rank == 0:
                param_norm = g.data.double().norm(2)
                total_norm += param_norm.item() ** 2

        # Sum across all model parallel GPUs.
        total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])
        torch.distributed.all_reduce(
            total_norm_cuda, op=torch.distributed.ReduceOp.SUM, group=dp_group
        )

        if mp_group is not None:
            dist.all_reduce(tensor=total_norm_cuda, op=torch.distributed.ReduceOp.SUM)

        total_norm = total_norm_cuda[0].item() ** (1.0 / norm_type)

    if (
        total_norm == float("inf")
        or total_norm == -float("inf")
        or total_norm != total_norm
    ):
        total_norm = -1

    return total_norm


def split_half_float_double(tensor_list):
    dtypes = [
        "torch.cuda.HalfTensor",
        "torch.cuda.FloatTensor",
        "torch.cuda.DoubleTensor",
        "torch.cuda.BFloat16Tensor",
    ]
    buckets = []
    for i, dtype in enumerate(dtypes):
        bucket = [t for t in tensor_list if t.type() == dtype]
        if bucket:
            buckets.append(bucket)
    return


def sync_param(flat_tensor, tensor_list):
    """
    Synchronize the flattened tensor and unflattened tensor list. When
    a list of tensor are flattened with `torch._utils._unflatten_dense_tensors`,
    a new tensor is created. Thus, the flat tensor and original tensor list do not
    share the same memory space. This function will update the tensor list so that
    they point to the same value.
    :param flat_tensor: A flat tensor obtained by calling `torch._utils._unflatten_dense_tensors` on a tensor lsit
    :param tensor_list: A list of tensors corresponding to the flattened tensor
    :type flat_tensor: torch.Tensor
    :type tensor_list: List[torch.Tensor]
    """
    updated_params = unflatten(flat_tensor, tensor_list)

    # update the tensor data
    for p, q in zip(tensor_list, updated_params):
        p.data = q.data
