# Copyright 2021 HPC-AI Technology Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by EleutherAI on 2023.

from collections import deque
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple

import torch

from .chunk import Chunk, ChunkFullError, TensorState
from oslo.torch.nn.parallel.data_parallel.zero.utils import get_current_device

from oslo.torch.distributed.parallel_context import ParallelContext


class ChunkManager:
    """
    A manager class to manipulate the tensors in chunks.

    Args:
        chunk_configuration (Dict[int, Dict]): the configuration dictionary of this chunk manager.
        init_device (torch.device): optional, the device on which the chunk is initialized. The default is None.
    """

    def __init__(
        self,
        chunk_configuration: Dict[int, Dict],
        init_device: Optional[torch.device] = None,
    ) -> None:

        self.device = init_device or get_current_device()
        self.dp_degree_chunk_size_dict: Dict[int, int] = dict()
        self.kwargs_config = chunk_configuration
        for k, v in self.kwargs_config.items():
            self.dp_degree_chunk_size_dict[k] = v.pop("chunk_size")
            v["init_device"] = self.device
        self.chunk_groups: Dict[str, Deque] = dict()
        self.tensor_chunk_map: Dict[torch.Tensor, Chunk] = dict()
        self.accessed_chunks: Set[Chunk] = set()
        self.accessed_mem: int = 0
        self.total_mem: Dict[str, int] = {"cpu": 0, "cuda": 0}

    def register_tensor(
        self,
        tensor: torch.Tensor,
        group_type: str,
        config_key: int,
        parallel_context: ParallelContext,
        cpu_offload: bool = False,
        pin_memory: bool = False,
    ) -> None:
        """
        Register a tensor to the chunk manager.
        Then, the tensor should be accessed by `get_chunks`.

        Args:
            tensor: the tensor appended to the chunk
            group_type: the data type of the group.
            config_key: the key of the group's name, the size of the dp world
            cpu_offload: if True, the chunk will be closed on CPU
            parallel_context: the parallel context
            pin_memory: whether the chunk is pinned in the cpu memory
        """
        assert tensor not in self.tensor_chunk_map
        assert isinstance(
            tensor, torch.Tensor
        ), "Please feed Tensor to this ChunkManager"
        assert config_key in self.dp_degree_chunk_size_dict

        chunk_size = self.dp_degree_chunk_size_dict[config_key]
        chunk_kwargs = self.kwargs_config[config_key]
        group_name = "{}_{}".format(group_type, config_key)
        chunk_group = self.__get_chunk_group(group_name)

        try:
            # append the tensor to the last chunk
            chunk_group[-1].append_tensor(tensor)
        except (IndexError, ChunkFullError):
            # the except statement will be triggered when there is no chunk or
            # the last chunk in the chunk group is full
            # this will create a new chunk and allocate this chunk to its corresponding process
            if chunk_group:
                # the chunk group is not empty
                # close the last chunk
                self.__close_one_chunk(chunk_group[-1])

            if tensor.numel() > chunk_size:
                chunk_size = tensor.numel()
                dp_size = tensor.get_dp_world_size()
                chunk_size = chunk_size + (-chunk_size % dp_size)

            chunk = Chunk(
                chunk_size=chunk_size,
                parallel_context=parallel_context,
                dtype=tensor.dtype,
                cpu_shard_init=cpu_offload,
                pin_memory=pin_memory,
                **chunk_kwargs,
            )

            chunk_group.append(chunk)
            chunk.append_tensor(tensor)
            self.__add_memory_usage(chunk.memory_usage)

        self.tensor_chunk_map[tensor] = chunk_group[-1]

    def close_all_groups(self):
        """Close all the chunks of all groups."""
        for group_name in self.chunk_groups:
            self.__close_one_chunk(self.chunk_groups[group_name][-1])

    def access_chunk(self, chunk: Chunk) -> None:
        """Make the chunk can be used for calculation."""
        if chunk in self.accessed_chunks:
            return
        self.__sub_memroy_usage(chunk.memory_usage)
        if chunk.device_type == "cpu":
            chunk.shard_move(get_current_device())
        self.__add_accessed_chunk(chunk)
        self.__add_memory_usage(chunk.memory_usage)

    def release_chunk(self, chunk: Chunk) -> None:
        """Scatter the chunk in CUDA."""
        if chunk not in self.accessed_chunks:
            return
        if chunk.can_release:
            self.__sub_memroy_usage(chunk.memory_usage)
            self.__sub_accessed_chunk(chunk)
            self.__add_memory_usage(chunk.memory_usage)

    def move_chunk(
        self, chunk: Chunk, device: torch.device, force_copy: bool = False
    ) -> None:
        """Move the shard of the chunk to the target device."""
        if not chunk.can_move or chunk.device_type == device.type:
            return
        self.__sub_memroy_usage(chunk.memory_usage)
        chunk.shard_move(device, force_copy)
        self.__add_memory_usage(chunk.memory_usage)

    def trans_tensor_state(self, tensor: torch.Tensor, state: TensorState) -> None:
        """Transit tensor state according to pre-defined state machine."""
        chunk = self.tensor_chunk_map[tensor]
        chunk.tensor_trans_state(tensor, state)

    def reduce_chunk(self, chunk: Chunk) -> bool:
        """Reduce or all reduce the chunk."""
        if not chunk.can_reduce:
            return False
        self.__sub_memroy_usage(chunk.memory_usage)
        chunk.reduce()
        self.__sub_accessed_chunk(chunk)
        self.__add_memory_usage(chunk.memory_usage)
        return True

    def fake_release_chunk(self, chunk: Chunk) -> None:
        """Release gathered chunk in a fake mode.
        This function is used for keep-gathered chunk in the inference mode.
        """
        assert chunk.keep_gathered
        assert chunk.tensor_state_cnter[TensorState.HOLD] == chunk.num_tensors
        self.__sub_accessed_chunk(chunk)

    def copy_tensor_to_chunk_slice(
        self, tensor: torch.Tensor, data: torch.Tensor
    ) -> None:
        """
        Copy data to the chunk.

        Args:
            tensor (torch.Tensor): the tensor used to retrieve meta information
            data (torch.Tensor): the tensor to be copied to the chunk
        """
        chunk = self.tensor_chunk_map[tensor]
        chunk.copy_tensor_to_chunk_slice(tensor, data)

    def get_chunk(self, tensor: torch.Tensor) -> Chunk:
        """
        Return the chunk owning the tensor.

        Args:
            tensor (torch.Tensor): a torch tensor object
        """
        return self.tensor_chunk_map[tensor]

    def get_cuda_movable_chunks(self) -> List[Chunk]:
        """
        Get all chunks that can be moved.
        """
        chunk_list = []
        for chunk in self.accessed_chunks:
            if chunk.can_release:
                chunk_list.append(chunk)
        chunk_list.sort(key=lambda x: x.count_id)
        return chunk_list

    def get_chunks(self, tensors: Iterable[torch.Tensor]) -> Tuple[Chunk, ...]:
        """
        Get all chunks owning the input tensors.

        Args:
            tensors (Iterable[torch.Tensor]): the tensors used to look for chunks
        """
        chunks = []
        for tensor in tensors:
            chunk = self.get_chunk(tensor)
            if chunk not in chunks:
                chunks.append(chunk)
        return tuple(chunks)

    def add_extern_static_tensor(self, tensor: torch.Tensor) -> None:
        """Add extern static tensor to chunk manager.
        Those tensors won't be managed by chunk manager, but we want to monitor memory usage of them.
        They are "static", which means their shape, dtype, device never change.
        Thus, their memory usage never changes.

        Args:
            tensor (torch.Tensor): An extern static tensor. E.g. optimizer state.
        """
        assert tensor not in self.tensor_chunk_map
        self.total_mem[tensor.device.type] += tensor.numel() * tensor.element_size()

    def __repr__(self) -> str:
        msg = [
            "Chunk Manager Information:\n",
            "Total memory: "
            + ", ".join([f"{k}={v}B" for k, v in self.total_mem.items()])
            + "\n",
        ]
        for group_name, group in self.chunk_groups.items():
            msg.append(f"Group {group_name}:\n")
            for i, chunk in enumerate(group):
                msg.append(f"[{i}] {chunk}\n")
        return "".join(msg)

    def __get_chunk_group(self, group_name: str) -> Deque:
        """Register a chunk group."""
        if group_name not in self.chunk_groups:
            self.chunk_groups[group_name] = deque()
        return self.chunk_groups[group_name]

    def __close_one_chunk(self, chunk: Chunk):
        """Close one chunk."""
        self.__sub_memroy_usage(chunk.memory_usage)
        chunk.close_chunk()
        self.__add_memory_usage(chunk.memory_usage)

    def __sub_memroy_usage(self, usage: Dict[str, int]):
        """Subtract memory usage from the total memory usage."""
        for k, v in usage.items():
            self.total_mem[k] -= v

    def __add_memory_usage(self, usage: Dict[str, int]):
        """Add memory usage to the total memory usage."""
        for k, v in usage.items():
            self.total_mem[k] += v

    def __add_accessed_chunk(self, chunk: Chunk):
        """Add a chunk to the accessed chunk list."""
        chunk.access_chunk()
        self.accessed_chunks.add(chunk)
        self.accessed_mem += chunk.chunk_mem

    def __sub_accessed_chunk(self, chunk: Chunk):
        """Subtract a chunk from the accessed chunk list."""
        chunk.release_chunk()
        self.accessed_chunks.remove(chunk)
        self.accessed_mem -= chunk.chunk_mem
