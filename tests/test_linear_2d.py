import torch

from oslo.torch.distributed import ParallelContext
from oslo.torch.nn import Linear2D


parallel_context = ParallelContext.from_torch(
    data_parallel_size=1,
    pipeline_parallel_size=1,
    tensor_parallel_size=4,
    tensor_parallel_mode="2d",
)

layer_2d = Linear2D(4, 4, parallel_context)
input_ = torch.Tensor([[0.1, 0.2], [0.3, 0.4]])
out = layer_2d(input_.cuda())
print(out, out.shape)