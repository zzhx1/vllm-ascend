import torch
import typing
from typing import List, Dict, Optional

from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.models.utils import extract_layer_index

from vllm.distributed.parallel_state import get_tp_group
from vllm_ascend.distributed.utils import all_gather_async
from vllm.distributed.parallel_state import GroupCoordinator

SHARD_LINEAR_MANAGER = None



def get_shard_linear_manager():
    return SHARD_LINEAR_MANAGER

def dispose_tensor(x: torch.Tensor):
    if x is not None and x.numel() > 0:
        x.set_(torch.empty([], device=x.device, dtype=x.dtype))


def init_shard_linear_manager(model, layer_sharding_config):
    global SHARD_LINEAR_MANAGER
    if SHARD_LINEAR_MANAGER is None:
        SHARD_LINEAR_MANAGER = []
    for series_name in layer_sharding_config:
        curr_series_manager = ShardLinearManager(series_name)
        for name, module in model.named_modules():
            if name.endswith(series_name):
                curr_series_manager.register_linear(module)
    
        SHARD_LINEAR_MANAGER.append(curr_series_manager)

def shard_linear_reach_full_weight(curr_layer_idx: int, comm_group: GroupCoordinator):
    global SHARD_LINEAR_MANAGER
    for series_manager in SHARD_LINEAR_MANAGER:
        series_manager.reach_full_weight(curr_layer_idx, comm_group)

class ShardLinearManager:
    series_collect: Dict[int, LinearBase] = {} # eg: 61 * o_proj
    full_series_weight: torch.Tensor = None
    full_tensor_ready_event: Optional[torch.distributed.Work] = None
    tp_mode_tensor: torch.Tensor = None

    def __init__(self, series_name):
        self.series_name = series_name
        self.tp_size = get_tp_group().world_size
    def register_linear(self, curr_linear: LinearBase):
        layer_idx = extract_layer_index(curr_linear.prefix)
        self.series_collect[layer_idx] = curr_linear
        # create a warp forward function for shard linear
        curr_linear._original_forward = curr_linear.forward
        curr_linear.forward = self._create_warp_forward(curr_linear)

        # set full tensor related attributes
        if hasattr(curr_linear, 'aclnn_weight_scale'):
            curr_linear.aclnn_input_scale = curr_linear.aclnn_input_scale.repeat(self.tp_size)
        if hasattr(curr_linear, 'aclnn_weight_scale_reciprocal'):    
            curr_linear.aclnn_input_scale_reciprocal = curr_linear.aclnn_weight_scale.repeat(self.tp_size)
        if hasattr(curr_linear, 'aclnn_input_offset'):
            curr_linear.aclnn_input_offset = curr_linear.aclnn_input_offset.repeat(self.tp_size)
        if hasattr(curr_linear, 'reduce_results'):
            curr_linear.reduce_results = False
        
        
    def reach_full_weight(self, curr_layer_idx: int, comm_group: GroupCoordinator):
        curr_linear = self.series_collect[curr_layer_idx]
        # assert curr_linear.weight is None, f"layer {curr_layer_idx} {self.series_name} weight should be None"
        self.tp_mode_tensor = curr_linear.weight.clone().detach()
        self.full_series_weight = comm_group.all_gather(self.tp_mode_tensor)
        
        # self.full_series_weight = torch.empty(
        #     (curr_linear.weight.size(0), curr_linear.weight.size(1) * comm_group.world_size),
        #     dtype=curr_linear.weight.dtype,
        #     device=curr_linear.weight.device
        # )
        # _, self.full_tensor_ready_event = all_gather_async(
        #     curr_linear.weight,
        #     comm_group,
        #     self.full_series_weight,
        #     async_op=False
        # )
        
        
    def wait_full_weight_handle(self):
        if self.full_tensor_ready_event is not None:
            self.full_tensor_ready_event.wait()
            self.full_tensor_ready_event = None


    def _create_warp_forward(self, curr_linear: LinearBase):
        def warp_forward(*args, **kwargs):
            self.wait_full_weight_handle()
            
            curr_linear.weight.set_(self.full_series_weight)
            print(f"zzh-debug: layer {curr_linear.prefix} use full weight for forward, weight shape: {curr_linear.weight.shape}", flush=True)
            res = curr_linear._original_forward(*args, **kwargs)
            curr_linear.weight.set_(self.tp_mode_tensor)
            dispose_tensor(self.full_series_weight)
            return res
            
        return warp_forward