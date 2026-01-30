import torch
import typing
from torch.nn import Parameter
from typing import List, Dict, Optional
import torch.distributed as dist
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.models.utils import extract_layer_index
from vllm.logger import logger
from vllm.distributed.parallel_state import get_tp_group
from vllm.distributed.parallel_state import GroupCoordinator

# Global registry: manages full-weight restoration for layers under the "shard linear" feature
SHARDED_LINEAR_MANAGERS: Optional[List["ShardedLinearFullWeightRestorer"]] = None


def get_sharded_linear_managers():
    return SHARDED_LINEAR_MANAGERS


def dispose_tensor(x: torch.Tensor):
    """Release the underlying storage of a tensor if it's valid."""
    if x is not None and x.numel() > 0:
        x.set_(torch.empty([], device=x.device, dtype=x.dtype))
        
        

def init_sharded_linear_mangers(model, layer_shard_config: List[str]):
    """
    Initialize full-weight restore logic for linear layers participating in the 'shard linear' feature.
    
    Args:
        model: The model instance to inspect.
        layer_shard_config: List of suffixes (e.g., ["o_proj"]) identifying target linear layers.
    """
    global SHARDED_LINEAR_MANAGERS
    if SHARDED_LINEAR_MANAGERS is not None:
        return  # Already initialized

    SHARDED_LINEAR_MANAGERS = []
    for module_name in layer_shard_config:
        logger.info(f"Initializing layer sharding with module name: {module_name}")
        restorer = ShardedLinearFullWeightRestorer(module_name)
        for name, module in model.named_modules():
            if name.endswith(module_name):
                restorer.register_sharded_linear(module)

        SHARDED_LINEAR_MANAGERS.append(restorer)


def trigger_sharded_linear_full_weight_prefetch(curr_layer_idx: int, comm_group):
    """
    Trigger asynchronous all-gather to prefetch full weights for current layer's sharded linear modules.
    """
    if SHARDED_LINEAR_MANAGERS is None:
        return
    for restorer in SHARDED_LINEAR_MANAGERS:
        restorer.prefetch_next_layer_full_weight(curr_layer_idx, comm_group)


class _LayerState:
    def __init__(self, linear_module: LinearBase, 
                 tp_sharded_weight: torch.Tensor, 
                 full_weight: Optional[torch.Tensor] = None, 
                 gather_work = None):
        self.linear_module = linear_module
        self.tp_sharded_weight = tp_sharded_weight
        self.full_weight = full_weight
        self.gather_work = gather_work

class ShardedLinearFullWeightRestorer:
    def __init__(self, sharded_linear_module: str):
        self.sharded_linear_module = sharded_linear_module
        self.tp_size = get_tp_group().world_size
        self.layer_idx_to_state: Dict[int, _LayerState] = {}
        self.gather_dim: int = -1                               # dimension along which to gather

    def register_sharded_linear(self, linear: LinearBase):
             
        layer_idx = extract_layer_index(linear.prefix)
        tp_sharded_weight = linear.weight.clone().detach()
        self.layer_idx_to_state[layer_idx] = _LayerState(linear, tp_sharded_weight)
        
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
        # prefetch full weight for first layer
        if layer_idx == 0:
            self.prefetch_next_layer_full_weight(layer_idx-1, get_tp_group(), async_op=False)
        

    def prefetch_next_layer_full_weight(self, curr_layer_idx: int, comm_group:GroupCoordinator, async_op: bool = True):
        # prefetch full weight for next layer
        prefetch_layer_idx = (curr_layer_idx+1) % len(self.layer_idx_to_state)
        state = self.layer_idx_to_state[prefetch_layer_idx]
        assert state.tp_sharded_weight is not None, "sharded weight should be clone before prefetch"
        if comm_group.world_size == 1:
            state.full_weight = state.tp_sharded_weight
            state.gather_work = None
            return

        assert state.full_weight is None, "Previous full_weight not cleaned up"
        state.full_weight, state.gather_work = all_gather_async_for_sharded_linear(
            input_=state.tp_sharded_weight,
            comm_group=comm_group,
            dim=self.gather_dim,
            async_op=async_op
        )
        
    def _sync_and_reshape_full_weight(self, layer_idx: int):
        """Wait for async gather and reshape into original full tensor layout."""
        state = self.layer_idx_to_state[layer_idx]
        if state.gather_work is not None:
            state.gather_work.wait()
            state.gather_work = None

        assert state.full_weight is not None, "Full weight not gathered"
        
        if self.tp_size > 1:
            input_tensor = state.tp_sharded_weight
            dim = self.gather_dim % input_tensor.dim()  # normalize negative index

            # Reshape from flat gathered tensor back to logical full shape
            shard_size = input_tensor.size(dim)
            full_size = self.tp_size * shard_size

            reshaped = state.full_weight.view((self.tp_size,) + input_tensor.shape)
            reshaped = reshaped.movedim(0, dim)
            state.full_weight = reshaped.reshape(
                input_tensor.shape[:dim] + (full_size,) + input_tensor.shape[dim + 1:]
            )


    def _create_full_weight_wrapped_forward(self, linear: LinearBase):
        def forward_with_full_weight(*args, **kwargs):
            layer_idx = extract_layer_index(linear.prefix)
            state = self.layer_idx_to_state[layer_idx]
            self._sync_and_reshape_full_weight(layer_idx)
            # Temporarily replace sharded weight with full version
            assert state.full_weight is not None, f"Full weight not reconstructed for layer {linear.prefix}"
            linear.weight.set_(state.full_weight)
            result = linear._original_forward(*args, **kwargs)
            # Restore original sharded weight
            linear.weight.set_(state.tp_sharded_weight)
            # Clean up full weight to avoid memory leak
            dispose_tensor(state.full_weight)
            state.full_weight = None
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
    return output_, all_gather_event


    
