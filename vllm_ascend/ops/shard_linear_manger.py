import torch
import typing
from torch.nn import Parameter
from typing import List, Dict, Optional
import torch.distributed as dist
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.models.utils import extract_layer_index
from vllm.logger import logger
from vllm.distributed.parallel_state import get_tp_group
from vllm_ascend.distributed.utils import all_gather_async
from vllm.distributed.parallel_state import GroupCoordinator

# Global registry: manages full-weight restoration for layers under the "shard linear" feature
SHARDED_LINEAR_FULL_WEIGHT_RESTORE_MANAGERS: Optional[List["ShardedLinearFullWeightRestorer"]] = None


def get_sharded_linear_full_weight_restore_managers():
    return SHARDED_LINEAR_FULL_WEIGHT_RESTORE_MANAGERS


def dispose_tensor(x: torch.Tensor):
    """Release the underlying storage of a tensor if it's valid."""
    if x is not None and x.numel() > 0:
        x.set_(torch.empty([], device=x.device, dtype=x.dtype))
        
        

def init_sharded_linear_full_weight_restore(model, sharded_linear_patterns: List[str]):
    """
    Initialize full-weight restore logic for linear layers participating in the 'shard linear' feature.
    
    Args:
        model: The model instance to inspect.
        sharded_linear_patterns: List of suffixes (e.g., ["o_proj"]) identifying target linear layers.
    """
    global SHARDED_LINEAR_FULL_WEIGHT_RESTORE_MANAGERS
    if SHARDED_LINEAR_FULL_WEIGHT_RESTORE_MANAGERS is not None:
        return  # Already initialized

    SHARDED_LINEAR_FULL_WEIGHT_RESTORE_MANAGERS = []
    for pattern in sharded_linear_patterns:
        logger.info(f"Initializing layer sharding with pattern: {pattern}")
        restorer = ShardedLinearFullWeightRestorer(pattern)
        for name, module in model.named_modules():
            if name.endswith(pattern):
                restorer.register_sharded_linear(module)
        if torch.distributed.get_rank() == 0:
            print(f"zzh-debug: init restorer {restorer.__dict__}")
        SHARDED_LINEAR_FULL_WEIGHT_RESTORE_MANAGERS.append(restorer)




def trigger_sharded_linear_full_weight_reconstruction(curr_layer_idx: int, comm_group):
    """
    Trigger asynchronous all-gather to reconstruct full weights for current layer's sharded linear modules.
    """
    if SHARDED_LINEAR_FULL_WEIGHT_RESTORE_MANAGERS is None:
        return
    for restorer in SHARDED_LINEAR_FULL_WEIGHT_RESTORE_MANAGERS:
        print(f"start reconstruction for pattern {restorer.sharded_linear_pattern}")
        restorer.initiate_full_weight_reconstruction(curr_layer_idx, comm_group)


class ShardedLinearFullWeightRestorer:
    def __init__(self, sharded_linear_pattern: str):
        self.sharded_linear_pattern = sharded_linear_pattern
        self.tp_size = get_tp_group().world_size
        self.layer_idx_to_sharded_linear: Dict[int, LinearBase] = {}

        # Runtime state during full-weight reconstruction cycle
        self.sharded_weight: Optional[torch.Tensor] = None      # local TP shard of weight
        self.full_weight: Optional[torch.Tensor] = None         # reconstructed full weight
        self.gather_work: Optional[dist.Work] = None            # async all-gather handle
        self.gather_dim: int = -1                               # dimension along which to gather

    def register_sharded_linear(self, linear: LinearBase):
        layer_idx = extract_layer_index(linear.prefix)
        self.layer_idx_to_sharded_linear[layer_idx] = linear

        # Wrap forward to temporarily use full weight (core of "shard linear" feature)
        linear._original_forward = linear.forward
        linear.forward = self._create_full_weight_wrapped_forward(linear)

        # Extend quantization-related tensors if present (replicate across TP ranks)
        if hasattr(linear, 'aclnn_input_scale'):
            linear.aclnn_input_scale = Parameter(
                linear.aclnn_input_scale.repeat(self.tp_size),
                requires_grad=False
            )
        if hasattr(linear, 'aclnn_input_scale_reciprocal'):
            linear.aclnn_input_scale_reciprocal = linear.aclnn_input_scale_reciprocal.repeat(self.tp_size)
        if hasattr(linear, 'aclnn_input_offset'):
            linear.aclnn_input_offset = linear.aclnn_input_offset.repeat(self.tp_size)

        # Disable result reduction for output projection (e.g., o_proj) under shard linear
        if linear.prefix.endswith("o_proj") and hasattr(linear, 'reduce_results'):
            linear.reduce_results = False

        # Determine gather dimension based on how the linear layer is partitioned
        if self.tp_size > 1:
            weight_shape = linear.weight.shape
            if linear.input_size != linear.input_size_per_partition:
                # Input channel is partitioned → gather along input dim (e.g., dim=1 for [out, in])
                self.gather_dim = next(
                    i for i, s in enumerate(weight_shape) if s == linear.input_size_per_partition
                )
            elif linear.output_size != linear.output_size_per_partition:
                # Output channel is partitioned → gather along output dim (e.g., dim=0)
                self.gather_dim = next(
                    i for i, s in enumerate(weight_shape) if s == linear.output_size_per_partition
                )
            else:
                raise ValueError(
                    f"Cannot determine gather dimension for {linear.prefix}: "
                    "no partitioned dimension found in weight shape {weight_shape}."
                )

    def initiate_full_weight_reconstruction(self, curr_layer_idx: int, comm_group):
        linear = self.layer_idx_to_sharded_linear[curr_layer_idx]
        self.sharded_weight = linear.weight.clone().detach()
        assert self.sharded_weight is not None, "sharded weight should be clone before reconstruction"
        if comm_group.world_size == 1:
            self.full_weight = self.sharded_weight
            self.gather_work = None
            return

        assert self.full_weight is None, "Previous full_weight not cleaned up"
        self.full_weight, self.gather_work = all_gather_async_for_sharded_linear(
            input_=self.sharded_weight,
            comm_group=comm_group,
            dim=self.gather_dim,
            async_op=False
        )
        print(f"zzh-debug-2: all_gather_event {self.gather_work}, output_size {self.full_weight.size()}")
        
    def _sync_and_reshape_full_weight(self):
        """Wait for async gather and reshape into original full tensor layout."""
        if self.gather_work is not None:
            self.gather_work.wait()
            self.gather_work = None

        assert self.full_weight is not None, "Full weight not gathered"
        if self.tp_size > 1:
            input_tensor = self.sharded_weight
            dim = self.gather_dim % input_tensor.dim()  # normalize negative index

            # Reshape from flat gathered tensor back to logical full shape
            shard_size = input_tensor.size(dim)
            full_size = self.tp_size * shard_size

            reshaped = self.full_weight.view((self.tp_size,) + input_tensor.shape)
            reshaped = reshaped.movedim(0, dim)
            self.full_weight = reshaped.reshape(
                input_tensor.shape[:dim] + (full_size,) + input_tensor.shape[dim + 1:]
            )


    def _create_full_weight_wrapped_forward(self, linear: LinearBase):
        def forward_with_full_weight(*args, **kwargs):
            self._sync_and_reshape_full_weight()
            # Temporarily replace sharded weight with full version
            assert self.full_weight is not None, f"Full weight not reconstructed for layer {linear.prefix}"
            linear.weight.set_(self.full_weight)
            result = linear._original_forward(*args, **kwargs)
            # Restore original sharded weight
            linear.weight.set_(self.sharded_weight)
            # Clean up full weight to avoid memory leak
            dispose_tensor(self.full_weight)
            self.full_weight = None
            return result
        return forward_with_full_weight
               
        

def all_gather_async_for_sharded_linear(input_: torch.Tensor,
                     comm_group: GroupCoordinator,
                     output_: Optional[torch.Tensor] = None,
                     dim: int = -1,
                     async_op: bool = True):
    if comm_group.world_size == 1:
        return input_, None
    if dim < 0:
        dim += input_.dim()
    input_size = input_.size()
    if output_ is None:
        output_size = (input_size[0] * comm_group.world_size, ) + input_size[1:]
        output_ = torch.empty(output_size,
                             dtype=input_.dtype,
                             device=input_.device)
    all_gather_event = dist.all_gather_into_tensor(output_,
                                               input_,
                                               group=comm_group.device_group,
                                               async_op=async_op)
    print(f"zzh-debug: all_gather_event {all_gather_event}, output_size {output_.size()}")
    return output_, all_gather_event


    
