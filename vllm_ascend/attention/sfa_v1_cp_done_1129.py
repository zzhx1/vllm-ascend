from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Optional, Tuple, Type, TypeVar

import torch
import torch_npu
from torch import nn
from vllm.attention.backends.abstract import (AttentionBackend,
                                              AttentionMetadata,
                                              MLAAttentionImpl)
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size, get_tp_group, get_dp_group
from vllm.model_executor.layers.linear import (LinearBase, ReplicatedLinear,
                                               UnquantizedLinearMethod)
from vllm.v1.attention.backends.utils import AttentionCGSupport

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.mla_v1 import MAX_O_PROJ_PREFETCH_SIZE
from vllm_ascend.attention.utils import (AscendCommonAttentionMetadata,
                                         wait_for_kv_layer_from_connector)
from vllm_ascend.ops.weight_prefetch import maybe_npu_prefetch
from vllm_ascend.utils import (ACL_FORMAT_FRACTAL_ND, ACL_FORMAT_FRACTAL_NZ,
                               is_enable_nz)
from vllm_ascend.worker.npu_input_batch import InputBatch
from vllm_ascend.utils import dispose_tensor
from vllm_ascend.distributed.sfa_sp_context import (set_sfa_sp_context_none, set_sfa_sp_context, get_sfa_sp_context,
                                                get_sfa_tmp_result_context, set_sfa_tmp_result_context, check_diff)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


class AscendSFABackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "ASCEND_SFA"

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return AscendSFAMetadata

    @staticmethod
    def get_builder_cls():
        return AscendSFAMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(num_blocks: int, block_size: int, num_kv_heads: int,
                           head_size: int) -> tuple[int, ...]:
        return (num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_impl_cls() -> Type["AscendSFAImpl"]:
        return AscendSFAImpl


@dataclass
class AscendSFAMetadata:
    """Metadata for MLACommon.

    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|
    has_prefill: bool
    num_actual_tokens: int  # Number of tokens excluding padding.
    slot_mapping: torch.Tensor
    seq_lens: torch.Tensor
    cum_query_lens: torch.Tensor
    block_tables: torch.Tensor
    sin: torch.Tensor
    cos: torch.Tensor

    # For logging.
    num_input_tokens: int = 0  # Number of tokens including padding.
    # The dimension of the attention heads
    head_dim: Optional[int] = None
    attn_mask: torch.Tensor = None
    # chunked prefill by default if no attn_states passed
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill


M = TypeVar("M", bound=AscendSFAMetadata)


class AscendSFAMetadataBuilder:
    # Does this backend/builder support ACL Graphs for attention (default: no).
    aclgraph_support: ClassVar[AttentionCGSupport] = \
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    # _attn_mask_builder = None
    def __init__(self,
                 kv_cache_spec,
                 layer_names,
                 vllm_config: VllmConfig,
                 device: torch.device,
                 metadata_cls: Optional[AscendSFAMetadata] = None):
        self.metadata_cls: Optional[AscendSFAMetadata] = metadata_cls \
            if metadata_cls is not None else AscendSFAMetadata  # type: ignore
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.device = device
        self.block_size = vllm_config.cache_config.block_size
        self.max_blocks = (vllm_config.model_config.max_model_len +
                           self.block_size - 1) // self.block_size

        self.speculative_config = vllm_config.speculative_config
        self.decode_threshold = 1
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            self.decode_threshold += spec_token_num
            assert self.decode_threshold <= 16, f"decode_threshold exceeded \
                npu_fused_infer_attention_score TND layout's limit of 16, \
                got {self.decode_threshold}"

        self.rope_dim = self.model_config.hf_text_config.qk_rope_head_dim
        self.cos_cache = None
        self.sin_cache = None

    def reorder_batch(self, input_batch: "InputBatch",
                      scheduler_output: "SchedulerOutput") -> bool:
        # No need to reorder for Ascend SFA
        return False

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        model: nn.Module,
    ) -> AscendSFAMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        device = self.device

        block_table = (common_attn_metadata.block_table_tensor[:num_reqs])
        slot_mapping = common_attn_metadata.slot_mapping[:
                                                         num_actual_tokens].to(
                                                             device,
                                                             non_blocking=True)
        input_positions = common_attn_metadata.positions[:
                                                         num_actual_tokens].long(
                                                         )
        query_start_loc = common_attn_metadata.query_start_loc
        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        has_prefill = any(query_lens > self.decode_threshold)

        if self.cos_cache is None:
            self.cos_cache = model.model.layers[
                model.model.start_layer].self_attn.rotary_emb.cos_cached
            self.sin_cache = model.model.layers[
                model.model.start_layer].self_attn.rotary_emb.sin_cached
        if self.cos_cache.dtype != self.model_config.dtype:  # type: ignore
            self.cos_cache = self.cos_cache.to(  # type: ignore
                self.model_config.dtype)  # type: ignore
            self.sin_cache = self.sin_cache.to(  # type: ignore
                self.model_config.dtype)  # type: ignore

        cum_query_lens = query_start_loc_cpu[1:num_reqs + 1].to(
            torch.int32).to(device, non_blocking=True)
        seq_lens = common_attn_metadata.seq_lens_cpu[:num_reqs].to(
            torch.int32).to(device, non_blocking=True)

        cos = self.cos_cache[input_positions].unsqueeze(  # type: ignore
            1).unsqueeze(2)
        sin = self.sin_cache[input_positions].unsqueeze(  # type: ignore
            1).unsqueeze(2)

        return self.metadata_cls(  # type: ignore
            has_prefill=has_prefill,
            num_input_tokens=common_attn_metadata.num_input_tokens,
            num_actual_tokens=num_actual_tokens,
            cum_query_lens=cum_query_lens,
            seq_lens=seq_lens,
            slot_mapping=slot_mapping,
            head_dim=self.model_config.get_head_size(),
            attn_mask=common_attn_metadata.attn_mask,
            attn_state=common_attn_metadata.attn_state,
            block_tables=block_table,
            sin=sin,
            cos=cos)

    def build_for_graph_capture(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        attn_state: AscendAttentionState = AscendAttentionState.DecodeOnly,
        model: Optional[nn.Module] = None,
    ):
        if attn_state == AscendAttentionState.DecodeOnly:
            attn_metadata = self.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
                model=model,
            )
        else:
            raise NotImplementedError(
                "Currently we only support building dummy metadata for DecodeOnly state"
            )

        attn_metadata.attn_state = attn_state
        return attn_metadata


class AscendSFAImpl(MLAAttentionImpl):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        logits_soft_cap: Optional[float],
        attn_type: str,
        kv_sharing_target_layer_name: Optional[str],
        **kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        # MLA Args
        self.q_lora_rank = kwargs['q_lora_rank']
        self.kv_lora_rank = kwargs['kv_lora_rank']
        self.qk_nope_head_dim = kwargs['qk_nope_head_dim']
        self.qk_rope_head_dim = kwargs['qk_rope_head_dim']
        self.qk_head_dim = kwargs['qk_head_dim']
        self.v_head_dim = kwargs['v_head_dim']
        self.rotary_emb = kwargs['rotary_emb']
        self.q_proj = kwargs['q_proj'] if self.q_lora_rank is None else kwargs[
            'q_b_proj']
        self.fused_qkv_a_proj = kwargs.get('fused_qkv_a_proj', None)
        self.kv_b_proj = kwargs['kv_b_proj']
        self.o_proj = kwargs['o_proj']
        self.indexer = kwargs['indexer']
        self.kv_a_proj_with_mqa = kwargs.get('kv_a_proj_with_mqa', None)
        self.kv_a_layernorm = kwargs.get('kv_a_layernorm', None)
        self.q_a_layernorm = kwargs.get('q_a_layernorm', None)
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tp_group().rank_in_group
        self.num_heads_per_rank = self.num_heads // self.tp_size
        self.q_b_proj = kwargs['q_b_proj']

        ascend_config = get_ascend_config()
        self.enable_shared_expert_dp = ascend_config.enable_shared_expert_dp
        self.enable_prefetch = ascend_config.weight_prefetch_config.enabled
        self.enable_kv_nz = ascend_config.torchair_graph_config.enable_kv_nz

        assert self.indexer is not None, "Indexer is required for DSA."
        
        self.enable_sfa_cp = ascend_config.enable_sfa_cp
        if self.enable_sfa_cp:
            vllm_config = get_current_vllm_config()
            # Dispose tensor from the original q_proj
            for attr_name in dir(self.q_proj):
                attr_value = getattr(self.q_proj, attr_name)
                if isinstance(attr_value, torch.Tensor):
                    dispose_tensor(attr_value)
            # Construct the new q_proj using ReplicatedLinear
            new_q_proj = ReplicatedLinear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim * self.tp_size,
                bias=False,
                quant_config=vllm_config.quant_config,
                prefix=self.q_proj.prefix)
            # Replace the q_proj with the new one
            self.q_proj.__class__ = new_q_proj.__class__
            self.q_proj.__dict__ = new_q_proj.__dict__

            # Dispose tensor from the original kv_b_proj
            for attr_name in dir(self.kv_b_proj):
                attr_value = getattr(self.kv_b_proj, attr_name)
                if isinstance(attr_value, torch.Tensor):
                    dispose_tensor(attr_value)
            # Construct the new kv_b_proj using ReplicatedLinear
            new_kv_b_proj = ReplicatedLinear(
                self.kv_lora_rank,
                self.num_heads * self.tp_size * (self.qk_nope_head_dim + self.v_head_dim),
                bias=False,
                quant_config=vllm_config.quant_config,
                prefix=self.kv_b_proj.prefix)
            # Replace the kv_b_proj with the new one
            self.kv_b_proj.__class__ = new_kv_b_proj.__class__
            self.kv_b_proj.__dict__ = new_kv_b_proj.__dict__

        # indexer param
        self.n_head: int = self.indexer.n_head  # 64
        self.head_dim: int = self.indexer.head_dim  # 128
        self.wq_b = self.indexer.wq_b
        self.wk = self.indexer.wk
        self.weights_proj = self.indexer.weights_proj
        self.k_norm = self.indexer.k_norm

        self.cp_size = 1

    def process_weights_after_loading(self, act_dtype: torch.dtype):

        def get_layer_weight(layer):
            WEIGHT_NAMES = ("weight", "qweight", "weight_packed")
            for attr in WEIGHT_NAMES:
                if hasattr(layer, attr):
                    return getattr(layer, attr)
            raise AttributeError(
                f"Layer '{layer}' has no recognized weight attribute:"
                f" {WEIGHT_NAMES}.")

        def get_and_maybe_dequant_weights(layer: LinearBase):
            if not isinstance(layer.quant_method, UnquantizedLinearMethod):
                # NOTE: This should only be used offline, since it's O(N^3)
                eye = torch.eye(layer.input_size_per_partition,
                                dtype=act_dtype,
                                device=get_layer_weight(layer).device)
                dequant_weights = layer.quant_method.apply(layer,
                                                           eye,
                                                           bias=None)
                del eye
                # standardize to (output, input)
                return dequant_weights.T
            # Weight will be reshaped next. To be on the safe side, the format
            # of the weight should be reverted to FRACTAL_AND.
            layer.weight.data = torch_npu.npu_format_cast(
                layer.weight.data, ACL_FORMAT_FRACTAL_ND)
            return layer.weight

        # we currently do not have quantized bmm's which are needed for
        # `W_UV` and `W_UK_T`, we we just store fp16/bf16 copies and perform
        # the bmm's in 16-bit, the extra memory overhead of this is fairly low
        kv_b_proj_weight = get_and_maybe_dequant_weights(self.kv_b_proj).T
        local_num_heads = self.num_heads
        if self.enable_sfa_cp:
            local_num_heads = self.num_heads * self.tp_size
        assert kv_b_proj_weight.shape == (
                self.kv_lora_rank,
                local_num_heads * (self.qk_nope_head_dim + self.v_head_dim)), (
                    f"{kv_b_proj_weight.shape=}, "
                    f"{self.kv_lora_rank=}, "
                    f"{local_num_heads=}, "
                    f"{self.qk_nope_head_dim=}, "
                    f"{self.v_head_dim=}")
        kv_b_proj_weight = kv_b_proj_weight.view(
            self.kv_lora_rank,
            local_num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )

        W_UK, W_UV = kv_b_proj_weight.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # Convert from (L, N, V) to (N, L, V)
        self.W_UV = W_UV.transpose(0, 1).contiguous()
        # Convert from (L, N, P) to (N, P, L)
        self.W_UK_T = W_UK.permute(1, 2, 0).contiguous()
        if self.enable_sfa_cp:
            self.W_UV_new = self.W_UV
            self.W_UK_T_new = self.W_UK_T
            #print("===1,",self.W_UV.shape,self.W_UK_T.shape,flush=True)
            chunk_size = self.W_UV.shape[0] // self.tp_size
            w_uv_chunks = torch.split(self.W_UV, chunk_size, dim=0)
            self.W_UV = w_uv_chunks[self.tp_rank]
            w_uk_chunks = torch.split(self.W_UK_T, chunk_size, dim=0)
            self.W_UK_T = w_uk_chunks[self.tp_rank]
            #print("===2,",self.W_UV.shape,self.W_UK_T.shape,chunk_size,flush=True)
        # Function `get_and_maybe_dequant_weights` will cast the weights to
        # FRACTAL_AND. So we need to cast to FRACTAL_NZ again.
        if is_enable_nz():
            self.kv_b_proj.weight.data = torch_npu.npu_format_cast(
                self.kv_b_proj.weight.data, ACL_FORMAT_FRACTAL_NZ)

        # Waiting for BMM NZ support
        # self.W_UV.data = torch_npu.npu_format_cast(self.W_UV.data, 29)
        # self.W_UK_T.data = torch_npu.npu_format_cast(self.W_UK_T.data, 29)

    def _v_up_proj(self, x):
        if self.W_UV.shape[0] * self.W_UV.shape[1] < 65536:
            x = x.view(-1, self.num_heads, self.kv_lora_rank)
            x = torch_npu.npu_transpose_batchmatmul(x,
                                                    self.W_UV,
                                                    perm_x1=[1, 0, 2],
                                                    perm_x2=[0, 1, 2],
                                                    perm_y=[1, 0, 2])
            x = x.reshape(-1, self.num_heads * self.v_head_dim)
        else:
            # Convert from (B, N, L) to (N, B, L)
            x = x.view(-1, self.num_heads, self.kv_lora_rank).transpose(0, 1)
            # # Multiply (N, B, L) x (N, L, V) -> (N, B, V)
            x = torch.bmm(x, self.W_UV)
            # # Convert from (N, B, V) to (B, N * V)
            x = x.transpose(0, 1).reshape(-1, self.num_heads * self.v_head_dim)
        return x

    # Return `ql_nope`, `q_pe`
    def _q_proj_and_k_up_proj(self, x):
        q_all=self.q_proj(x)[0]
        if self.enable_sfa_cp:
            chunk_size = q_all.shape[1] // self.tp_size
            q_chunks=torch.split(q_all, chunk_size, dim=1)
            q_all = q_chunks[self.tp_rank]
            #print("===0,y=",q_all.shape,q_chunks[tp_rank].shape,tp_rank,flush=True)
        q_nope, q_pe = q_all\
            .view(-1, self.num_heads, self.qk_head_dim)\
            .split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        #print("===1,y=",q_all.shape,q_chunks[tp_rank].shape,q_nope.shape,q_pe.shape,tp_rank,flush=True)
        # Convert from (B, N, P) to (N, B, P)
        q_nope = q_nope.transpose(0, 1)
        # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
        ql_nope = torch.bmm(q_nope, self.W_UK_T)
        # Convert from (N, B, L) to (B, N, L)
        return ql_nope.transpose(0, 1), q_pe

    def _q_proj_and_k_up_proj_new(self, x):
        q_all=self.q_proj(x)[0]
        # if self.enable_sfa_cp:
        #     chunk_size = q_all.shape[1] // self.tp_size
        #     q_chunks=torch.split(q_all, chunk_size, dim=1)
        #     q_all = q_chunks[self.tp_rank]
        #     #print("===0,y=",q_all.shape,q_chunks[tp_rank].shape,tp_rank,flush=True)
        q_nope, q_pe = q_all\
            .view(-1, self.num_heads * self.tp_size, self.qk_head_dim)\
            .split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        #print("===1,y=",q_all.shape,q_nope.shape,q_pe.shape,self.W_UK_T_new.shape,self.tp_rank,flush=True)
        # Convert from (B, N, P) to (N, B, P)
        q_nope = q_nope.transpose(0, 1)
        #print("===2,y=",q_all.shape,q_nope.shape,q_pe.shape,self.W_UK_T_new.shape,self.tp_rank,flush=True)
        # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
        ql_nope = torch.bmm(q_nope, self.W_UK_T_new)
        # Convert from (N, B, L) to (B, N, L)
        return ql_nope.transpose(0, 1), q_pe

    def exec_kv(
        self,
        kv_no_split: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Tuple,
        slots: torch.Tensor,
    ):
        B = kv_no_split.shape[0]
        N = self.num_kv_heads
        S = 1
        # npu_kv_rmsnorm_rope_cache needs [B, N, S, D]
        kv_no_split = kv_no_split.view(
            B, N, S, self.kv_lora_rank + self.qk_rope_head_dim)
        cache_mode = "PA_NZ" if self.enable_kv_nz else "PA"
        #k_pe, k_nope, _, _ 
        _, _, k_pe, k_nope = torch_npu.npu_kv_rmsnorm_rope_cache(
            kv_no_split,
            self.kv_a_layernorm.weight,
            cos,
            sin,
            slots.to(torch.int64),
            kv_cache[1],
            kv_cache[0],
            epsilon=self.kv_a_layernorm.variance_epsilon,
            cache_mode=cache_mode,
            is_output_kv=True,
        )
        #print("===1",torch.equal(k_pe, kv_cache[1]),torch.equal(k_nope, kv_cache[0]),flush=True)
        return k_pe, k_nope

    def exec_kv_new(
        self,
        kv_no_split: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Tuple,
        slots: torch.Tensor,
        all_slots: torch.Tensor,
    ):
        #before = kv_cache[0].detach().clone()

        B = kv_no_split.shape[0]
        N = self.num_kv_heads
        S = 1
        # npu_kv_rmsnorm_rope_cache needs [B, N, S, D]
        kv_no_split = kv_no_split.view(
            B, N, S, self.kv_lora_rank + self.qk_rope_head_dim)
        cache_mode = "PA_NZ" if self.enable_kv_nz else "PA"

        _, _, k_pe, k_nope = torch_npu.npu_kv_rmsnorm_rope_cache(
            kv_no_split,
            self.kv_a_layernorm.weight,
            cos,
            sin,
            slots.to(torch.int64),
            kv_cache[1],
            kv_cache[0],
            epsilon=self.kv_a_layernorm.variance_epsilon,
            cache_mode=cache_mode,
            is_output_kv=True,
        )
        #pos = all_slots.view(-1, 1)[0, 0].item()
        #val = k_nope.view(-1, k_nope.shape[-1])[0]


        k_pe = get_tp_group().all_gather(k_pe, 0)
        k_nope = get_tp_group().all_gather(k_nope, 0)
        #k_pe = k_pe * 3
        #k_nope = k_nope * 4
        #k_pe = k_pe[:attn_metadata.num_actual_tokens]
        #k_nope = k_nope[:attn_metadata.num_actual_tokens]


        if kv_cache is not None:
            torch_npu.npu_scatter_nd_update_(kv_cache[0].view(-1, k_nope.shape[-1]),
                                             all_slots.view(
                                                 -1, 1),
                                             k_nope.view(-1,
                                                    k_nope.shape[-1]))  # b, s, n, d
            torch_npu.npu_scatter_nd_update_(kv_cache[1].view(-1, k_pe.shape[-1]),
                                             all_slots.view(
                                                 -1, 1),
                                             k_pe.view(-1,
                                                    k_pe.shape[-1]))  # b, s, n, d
        return k_pe, k_nope

    def rope_single(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        B, N, D = x.shape
        S = 1
        x = x.view(B, N, S, D)
        x = torch_npu.npu_interleave_rope(x, cos, sin)
        return x.view(B, N, D)

    def forward(
        self,
        layer_name,
        hidden_states: torch.Tensor,  # query in unified attn
        kv_cache: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        attn_metadata: M,
        need_gather_q_kv: bool = False,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)
        
        should_cp = self.enable_sfa_cp and "model.layers.0.self_attn.attn" in layer_name and attn_metadata.has_prefill and attn_metadata.num_actual_tokens == hidden_states.shape[0]
        if "model.layers.0.self_attn.attn" in layer_name and self.tp_rank==0 and attn_metadata.has_prefill:
            print("===0,hidden_states",hidden_states.shape,flush=True)
            print("===0,cos",attn_metadata.cos.shape,flush=True)
            print("===0,slot_mapping",attn_metadata.slot_mapping[:attn_metadata.num_actual_tokens].shape,flush=True)
            print("===0,meta,",attn_metadata,attn_metadata.seq_lens.shape,attn_metadata.cum_query_lens.shape,attn_metadata.block_tables.shape,flush=True)
        
        pad_size = 0
        start_index = 0
        end_index = 0
        set_sfa_sp_context_none()
        if should_cp:
            '''
            remainder = hidden_states.shape[0] % self.tp_size
            if remainder == 0:
                pad_size = 0
            else:
                pad_size = self.tp_size - remainder
            set_sfa_tmp_result_context(pad_size=pad_size)
            hidden_states1 =  hidden_states
            if pad_size > 0:
                hidden_states1 = nn.functional.pad(hidden_states,
                                                  (0, 0, 0, pad_size))
            set_sfa_sp_context(hidden_states1)
            sfa_sp_context = get_sfa_sp_context()
            dp_rank = get_dp_group().rank_in_group
            if sfa_sp_context is not None:
                start_index = sfa_sp_context.dp_device_sp_start_token[dp_rank][self.tp_rank]
                end_index = sfa_sp_context.dp_device_sp_end_token[dp_rank][self.tp_rank]
                cp_states = hidden_states1[start_index:end_index]
                set_sfa_tmp_result_context(tp_rank=self.tp_rank,cp_hidden_states=cp_states)
                #print("===1,sfa_sp_context=",sfa_sp_context,hidden_states1.shape,sp_states.shape,self.tp_rank,dp_rank,layer_name,flush=True)
            '''
            hidden_states_cp = hidden_states
            set_sfa_sp_context(hidden_states_cp)
            sfa_sp_context = get_sfa_sp_context()
            if sfa_sp_context.pad_size > 0:
                hidden_states_cp = nn.functional.pad(hidden_states_cp, (0, 0, 0, sfa_sp_context.pad_size))
            hidden_states_cp = hidden_states_cp[sfa_sp_context.local_start:sfa_sp_context.local_end_with_pad]

        has_prefill = attn_metadata.has_prefill
        num_actual_tokens = attn_metadata.num_actual_tokens
        hidden_states = hidden_states[:num_actual_tokens]
        # Inputs and outputs may be padded for CUDA graphs
        output_padded = output
        output = output[:num_actual_tokens]
        assert self.fused_qkv_a_proj is not None, "q lora is required for DSA."
        maybe_npu_prefetch(inputs=self.fused_qkv_a_proj.weight,
                           dependency=hidden_states,
                           enabled=self.enable_prefetch)
        qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
        
        q_c, kv_no_split = qkv_lora.split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            dim=-1,
        )
        q_c = self.q_a_layernorm(q_c)
        
        # Process for Flash Comm V1
        q_c = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            q_c.contiguous(), need_gather_q_kv)
        kv_no_split = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            kv_no_split.contiguous(), need_gather_q_kv)
        
        ### execute cp 
        #new_qkv_lora = self.fused_qkv_a_proj(get_sfa_tmp_result_context().hidden_states[self.tp_rank])[0]
        #new_q_c, new_kv_no_split = new_qkv_lora.split(
        #    [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
        #    dim=-1,
        #)
        #new_q_c = self.q_a_layernorm(new_q_c)
        #set_sfa_tmp_result_context(tp_rank=self.tp_rank,qkv_lora=new_qkv_lora)
        #set_sfa_tmp_result_context(tp_rank=self.tp_rank,q_c=new_q_c)
        #new_q_c = get_tp_group().all_gather(new_q_c, 0)
        #if "model.layers.0.self_attn.attn" in layer_name and self.tp_rank==0 and has_prefill:
        #    print("===q_c",q_c.shape,q_c,need_gather_q_kv,flush=True)
        #    print("===new_q_c",new_q_c.shape,new_q_c,need_gather_q_kv,flush=True)

        if has_prefill:
            wait_for_kv_layer_from_connector(layer_name) # TODO

        slot_mapping = attn_metadata.slot_mapping[:num_actual_tokens]
        ql_nope, q_pe = \
            self._q_proj_and_k_up_proj(q_c)
        q_pe = self.rope_single(q_pe, attn_metadata.cos, attn_metadata.sin)

        kv0_cp = kv_cache[0].detach().clone()
        kv1_cp = kv_cache[1].detach().clone()
        kv2_cp = kv_cache[2].detach().clone()
        kv0_init = kv_cache[0].detach().clone()
        kv1_init = kv_cache[1].detach().clone()

        k_pe, k_nope = self.exec_kv(kv_no_split, attn_metadata.cos,
                                    attn_metadata.sin, kv_cache, slot_mapping)
        
        ### execute cp
        #hidden_states_cp = get_sfa_tmp_result_context().hidden_states[self.tp_rank]
        if should_cp:
            new_qkv_lora = self.fused_qkv_a_proj(hidden_states_cp)[0]
            #new_qkv_lora = self.fused_qkv_a_proj(get_sfa_tmp_result_context().hidden_states[self.tp_rank])[0]
            new_q_c, new_kv_no_split = new_qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            new_q_c = self.q_a_layernorm(new_q_c)
            #set_sfa_tmp_result_context(tp_rank=self.tp_rank,qkv_lora=new_qkv_lora)
            #set_sfa_tmp_result_context(tp_rank=self.tp_rank,q_c=new_q_c)

            #TODO: wait_for_kv_layer_from_connector

            new_ql_nope, new_q_pe = \
                self._q_proj_and_k_up_proj_new(new_q_c)
            cos = attn_metadata.cos
            sin = attn_metadata.sin
            slot_mapping_padded = slot_mapping
            if sfa_sp_context.pad_size > 0:
                cos = nn.functional.pad(cos, (0, 0, 0, 0, 0, 0, 0, sfa_sp_context.pad_size))
                sin = nn.functional.pad(sin, (0, 0, 0, 0, 0, 0, 0, sfa_sp_context.pad_size))
                slot_mapping_padded = nn.functional.pad(slot_mapping, (0, sfa_sp_context.pad_size), value=-1)
            cos = cos[sfa_sp_context.local_start:sfa_sp_context.local_end_with_pad]
            sin = sin[sfa_sp_context.local_start:sfa_sp_context.local_end_with_pad]
            slot_mapping_cp = slot_mapping_padded[sfa_sp_context.local_start:sfa_sp_context.local_end_with_pad]
            new_q_pe = self.rope_single(new_q_pe, cos, sin)

            new_k_pe, new_k_nope = self.exec_kv_new(new_kv_no_split, cos,
                                         sin, (kv0_cp, kv1_cp, kv2_cp), slot_mapping_cp, slot_mapping_padded)
            #                            sin, kv_cache, slot_mapping_cp, slot_mapping_padded)
            
            new_ql_nope_dbg = get_tp_group().all_gather(new_ql_nope.contiguous(), 0)[:num_actual_tokens]
            old_ql_nope = get_tp_group().all_gather(ql_nope.contiguous(), 1)
            new_q_pe_dbg = get_tp_group().all_gather(new_q_pe.contiguous(), 0)[:num_actual_tokens]
            old_q_pe = get_tp_group().all_gather(q_pe.contiguous(), 1)
            new_k_pe = new_k_pe[:num_actual_tokens]
            new_k_nope = new_k_nope[:num_actual_tokens]
            #new_k_pe = get_tp_group().all_gather(new_k_pe.contiguous(), 0)
            #new_k_nope = get_tp_group().all_gather(new_k_nope.contiguous(), 0)

            if "model.layers.0.self_attn.attn" in layer_name and self.tp_rank==0 and has_prefill:
                '''
                print("===cos",cos.shape,sin.shape,flush=True)
                print("===new_ql_nope",new_ql_nope.shape,new_ql_nope,need_gather_q_kv,flush=True)
                print("===ql_nope",ql_nope.shape,ql_nope,need_gather_q_kv,flush=True)
                print("===new_q_pe",new_q_pe.shape,new_q_pe,need_gather_q_kv,flush=True)
                print("===q_pe",q_pe.shape,q_pe,need_gather_q_kv,flush=True)

                print("===new_k_pe",new_k_pe.shape,new_k_pe,need_gather_q_kv,flush=True)
                print("===k_pe",k_pe.shape,k_pe,need_gather_q_kv,flush=True)
                print("===new_k_nope",new_k_nope.shape,new_k_nope,need_gather_q_kv,flush=True)
                print("===k_nope",k_nope.shape,k_nope,need_gather_q_kv,flush=True)
                '''
                print("!!!!!stageResult", check_diff(new_ql_nope_dbg, old_ql_nope),
                      check_diff(new_q_pe_dbg, old_q_pe),
                      check_diff(new_k_pe, k_pe),
                      check_diff(new_k_nope, k_nope),
                      check_diff(kv_cache[0], kv0_init), # answer compare with init
                      check_diff(kv_cache[1], kv1_init),
                      check_diff(kv_cache[0], kv0_cp), # answer compare with cp
                      check_diff(kv_cache[1], kv1_cp),
                      attn_metadata.slot_mapping[:num_actual_tokens].shape,
                      slot_mapping_padded.shape,
                      slot_mapping_padded,
                      flush=True)

        topk_indices = self.indexer_select(x=hidden_states,
                                           qr=q_c,
                                           kv_cache=kv_cache,
                                           attn_metadata=attn_metadata,
                                           need_gather_q_kv=need_gather_q_kv,
                                           layer_name=layer_name)
        
        if should_cp:
            cum_query_lens = attn_metadata.cum_query_lens
            seq_lens = attn_metadata.seq_lens
            num_segs = cum_query_lens.shape[0]
            actual_seq_lengths_query = torch.empty_like(cum_query_lens) # 每个请求这轮处理几个token的前缀和
            actual_seq_lengths_key = torch.empty_like(seq_lens) # 每个请求这轮最后一个token能看见几个token 包括自己
            last_token = 0
            cum = 0
            for i in range(0, num_segs):
                global_start = last_token
                global_end = cum_query_lens[i].item()
                last_token = global_end

                local_start = max(global_start, sfa_sp_context.local_start)
                local_end = min(global_end, sfa_sp_context.local_end_with_pad)
                num_local_tokens = local_end - local_start

                if num_local_tokens > 0:
                    cum += num_local_tokens
                    actual_seq_lengths_query[i] = cum

                    offset = global_end - local_end
                    actual_seq_lengths_key[i] = seq_lens[i].item() - offset
                else:
                    actual_seq_lengths_query[i] = cum
                    actual_seq_lengths_key[i] = 0
            
            actual_seq_lengths_query_dbg = get_tp_group().all_gather(actual_seq_lengths_query.unsqueeze(0), 0)
            actual_seq_lengths_key_dbg = get_tp_group().all_gather(actual_seq_lengths_key.unsqueeze(0), 0)

            if "model.layers.0.self_attn.attn" in layer_name and self.tp_rank==0 and has_prefill:
                print("!!!!!watch", attn_metadata.cum_query_lens, attn_metadata.seq_lens, attn_metadata.block_tables.shape, flush=True)
                print("!!!!!cons", actual_seq_lengths_query_dbg, actual_seq_lengths_key_dbg, flush=True)

            new_topk_indices = self.indexer_select_new(x=hidden_states_cp,
                                           qr=new_q_c,
                                           cos=cos,
                                           sin=sin,
                                           all_slots=slot_mapping_padded,
                                           actual_seq_lengths_query=actual_seq_lengths_query,
                                           actual_seq_lengths_key=actual_seq_lengths_key,
                                           block_tables=attn_metadata.block_tables,
                                           kv_cache=(kv0_cp, kv1_cp, kv2_cp),
                                           layer_name=layer_name)
            new_topk_indices_dbg = get_tp_group().all_gather(new_topk_indices, 0)[:,:,:10].view(-1, 10)[:num_actual_tokens]
            old_topk_indices = topk_indices[:,:,:10].view(-1, 10)
            if "model.layers.0.self_attn.attn" in layer_name and self.tp_rank==0 and has_prefill:
                print("!!!!!topk", check_diff(kv_cache[2], kv2_cp), check_diff(old_topk_indices, new_topk_indices_dbg), old_topk_indices.shape, old_topk_indices, new_topk_indices_dbg.shape, new_topk_indices_dbg, flush=True)
        
        '''
        topk_indices_new = None
        if has_prefill:
            topk_indices_new = self.indexer_select_new(x=hidden_states_cp,
                                           qr=new_q_c,
                                           cos=cos,
                                           sin=sin,
                                           kv_cache=kv_cache,
                                           attn_metadata=attn_metadata,
                                           need_gather_q_kv=need_gather_q_kv,
                                           layer_name=layer_name)
        #topk_indices_new = get_tp_group().all_gather(topk_indices_new, 0)
        if "model.layers.0.self_attn.attn" in layer_name and self.tp_rank<6 and has_prefill:
            print("===topk",topk_indices.shape,topk_indices,flush=True)
            if topk_indices_new is not None:
                print("===topk_new",self.tp_rank,topk_indices_new.shape,topk_indices_new,flush=True)

        if "model.layers.0.self_attn.attn" in layer_name and self.tp_rank == 0:
                print("===12,",ql_nope.shape,topk_indices.shape,attn_metadata.cum_query_lens.shape,attn_metadata.cum_query_lens,attn_metadata.seq_lens.shape,attn_metadata.seq_lens,q_pe.shape,flush=True)
        '''
        attn_output = torch.ops.custom.npu_sparse_flash_attention(
            query=ql_nope,
            key=kv_cache[0],#k_nope,
            value=kv_cache[0],#k_nope,
            sparse_indices=topk_indices,
            scale_value=self.scale,
            sparse_block_size=1,
            block_table=attn_metadata.block_tables,
            actual_seq_lengths_query=attn_metadata.cum_query_lens,
            actual_seq_lengths_kv=attn_metadata.seq_lens,
            query_rope=q_pe,
            key_rope=kv_cache[1],#k_pe,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
        )
        old_attn_output = get_tp_group().all_gather(attn_output, 1)

        if should_cp:
            new_attn_output = torch.ops.custom.npu_sparse_flash_attention(
                query=new_ql_nope,
                key=kv_cache[0],#k_nope,
                value=kv_cache[0],#k_nope,
                sparse_indices=new_topk_indices,
                scale_value=self.scale,
                sparse_block_size=1,
                block_table=attn_metadata.block_tables,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_kv=actual_seq_lengths_key,
                query_rope=new_q_pe,
                key_rope=kv_cache[1],#k_pe,
                layout_query="TND",
                layout_kv="PA_BSND",
                sparse_mode=3,
            )
            new_attn_output_dbg = get_tp_group().all_gather(new_attn_output, 0)[:num_actual_tokens]
            if "model.layers.0.self_attn.attn" in layer_name and self.tp_rank==0 and has_prefill:
                print("!!!!!attn", check_diff(old_attn_output, new_attn_output_dbg), old_attn_output.shape, new_attn_output_dbg.shape, flush=True)
        
        '''
        attn_output_new = None
        if has_prefill and start_index < num_actual_tokens:
            seq_lens_cp = attn_metadata.seq_lens.detach().clone()
            cum_query_lens_cp = attn_metadata.cum_query_lens.detach().clone()
            seq_lens_cp[0] = end_index
            cum_query_lens_cp[0] = 1
            
            if "model.layers.0.self_attn.attn" in layer_name:
                print("===11,",self.tp_rank,new_ql_nope.shape,topk_indices_new.shape,cum_query_lens_cp.shape,cum_query_lens_cp,seq_lens_cp.shape,seq_lens_cp,new_q_pe.shape,flush=True)

            attn_output_new = torch.ops.custom.npu_sparse_flash_attention(
                query=new_ql_nope,
                key=kv_cache[0],
                value=kv_cache[0],
                sparse_indices=topk_indices_new,
                scale_value=self.scale,
                sparse_block_size=1,
                block_table=attn_metadata.block_tables,
                actual_seq_lengths_query=cum_query_lens_cp,
                actual_seq_lengths_kv=seq_lens_cp,
                query_rope=new_q_pe,
                key_rope=kv_cache[1],
                layout_query="TND",
                layout_kv="PA_BSND",
                sparse_mode=3,
            )

            if False and "model.layers.0.self_attn.attn" in layer_name:
                print("===attn_output",attn_output.shape,attn_output,flush=True)
                print("===attn_output_new",attn_output_new.shape,attn_output_new,flush=True)

        if has_prefill and attn_output_new is None:
            attn_output_new = torch.empty(1,128,512,
                               dtype=attn_output.dtype,
                               device=attn_output.device)
        if has_prefill:
            attn_output_new = get_tp_group().all_gather(attn_output_new, 0)
            attn_output_new = attn_output_new[:num_actual_tokens]

            if "model.layers.0.self_attn.attn" in layer_name:
                print("===attn_output_new",attn_output_new.shape,attn_output_new,flush=True)
                print("===attn_output_old",attn_output_old.shape,attn_output_old,flush=True)
                print("===attn_output_equal",torch.equal(attn_output_old,attn_output_new),flush=True)
        '''


        attn_output = self._v_up_proj(attn_output)
        maybe_npu_prefetch(inputs=self.o_proj.weight,
                           dependency=attn_output,
                           max_size=MAX_O_PROJ_PREFETCH_SIZE,
                           enabled=self.enable_prefetch)
        output[...] = self.o_proj(attn_output)[0]
        return output_padded

    def indexer_select(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        attn_metadata: M,
        need_gather_q_kv: bool = False,
        layer_name: str = "",
    ):
        cos = attn_metadata.cos
        sin = attn_metadata.sin

        cos_q, sin_q = cos, sin
        cos = cos.view(-1, 1, 1, self.qk_rope_head_dim)
        sin = sin.view(-1, 1, 1, self.qk_rope_head_dim)

        # q process in new stream
        q, _ = self.wq_b(qr)  # [b,s,1536] @ [1536,64*128] = [b,s,64*128]
        q = q.view(-1, self.n_head, self.head_dim)  # [b,s,64,128]
        q_pe, q_nope = torch.split(
            q, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim],
            dim=-1)  # [b,s,64,64+64]

        q_pe = q_pe.unsqueeze(2)
        q_pe = torch_npu.npu_interleave_rope(q_pe, cos_q, sin_q)
        q_pe = q_pe.squeeze(2)
        q = torch.cat([q_pe, q_nope], dim=-1)  # [b*s,64,128]

        k_proj, _ = self.wk(x)  # [b,s,7168] @ [7168,128] = [b,s,128]
        k_proj = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            k_proj, need_gather_q_kv)
        k = self.k_norm(k_proj).unsqueeze(1)
        k_pe, k_nope = torch.split(
            k, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim],
            dim=-1)  # [b,s,64+64]

        k_pe = k_pe.unsqueeze(2)
        k_pe = torch_npu.npu_interleave_rope(k_pe, cos, sin)
        k_pe = k_pe.squeeze(2)

        k = torch.cat([k_pe, k_nope], dim=-1)  # [b*s,128]
        if False and "model.layers.0.self_attn.attn" in layer_name and self.tp_rank==0:
            print("===k1,",k.shape,k,flush=True)
            print("===q1,",q.shape,q,flush=True)
            print("===meta1,",attn_metadata.seq_lens,attn_metadata.cum_query_lens,attn_metadata.block_tables,flush=True)
        if kv_cache is not None:
            torch_npu.npu_scatter_nd_update_(kv_cache[2].view(-1, k.shape[-1]),
                                             attn_metadata.slot_mapping.view(
                                                 -1, 1),
                                             k.view(-1,
                                                    k.shape[-1]))  # b, s, n, d

        weights, _ = self.weights_proj(x)
        weights = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            weights, need_gather_q_kv)

        block_table = attn_metadata.block_tables
        seq_lens = attn_metadata.seq_lens
        cum_query_lens = attn_metadata.cum_query_lens

        '''
        if "model.layers.0.self_attn.attn" in layer_name and self.tp_rank==0:
            print("===1-q,",q.shape,q,flush=True)
            print("===1-k,",kv_cache[2].shape,kv_cache[2],flush=True)
            print("===1-seq_lens,",seq_lens.shape,seq_lens,flush=True)
            print("===1-cum_query_lens,",cum_query_lens.shape,cum_query_lens,flush=True)
        '''

        topk_indices = torch.ops.custom.npu_lightning_indexer(
            query=q,
            key=kv_cache[2],
            weights=weights,
            actual_seq_lengths_query=cum_query_lens,
            actual_seq_lengths_key=seq_lens,
            block_table=block_table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=2048,
            sparse_mode=3)
        return topk_indices

    def indexer_select_new(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        all_slots: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        block_tables: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        #attn_metadata: M,
        layer_name: str = "",
    ):
        # cos = attn_metadata.cos
        # sin = attn_metadata.sin

        cos_q, sin_q = cos, sin
        cos = cos.view(-1, 1, 1, self.qk_rope_head_dim)
        sin = sin.view(-1, 1, 1, self.qk_rope_head_dim)

        # q process in new stream
        q, _ = self.wq_b(qr)  # [b,s,1536] @ [1536,64*128] = [b,s,64*128]
        q = q.view(-1, self.n_head, self.head_dim)  # [b,s,64,128]
        q_pe, q_nope = torch.split(
            q, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim],
            dim=-1)  # [b,s,64,64+64]

        q_pe = q_pe.unsqueeze(2)
        q_pe = torch_npu.npu_interleave_rope(q_pe, cos_q, sin_q)
        q_pe = q_pe.squeeze(2)
        q = torch.cat([q_pe, q_nope], dim=-1)  # [b*s,64,128]

        k_proj, _ = self.wk(x)  # [b,s,7168] @ [7168,128] = [b,s,128]
        k = self.k_norm(k_proj).unsqueeze(1)
        k_pe, k_nope = torch.split(
            k, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim],
            dim=-1)  # [b,s,64+64]

        k_pe = k_pe.unsqueeze(2)
        k_pe = torch_npu.npu_interleave_rope(k_pe, cos, sin)
        k_pe = k_pe.squeeze(2)

        k = torch.cat([k_pe, k_nope], dim=-1)  # [b*s,128]

        # cp k all_gather
        k = get_tp_group().all_gather(k, 0)
        #k = k[:attn_metadata.num_actual_tokens]
        if False and "model.layers.0.self_attn.attn" in layer_name and self.tp_rank==0:
            print("===k-cp,",k.shape,k,flush=True)
            print("===q-cp,",q.shape,q,flush=True)
            print("===meta-cp,",attn_metadata.seq_lens,attn_metadata.cum_query_lens,attn_metadata.block_tables,flush=True)
        #print("===11-cp,",attn_metadata.slot_mapping.shape,k.shape,k,flush=True)
        if kv_cache is not None:
            torch_npu.npu_scatter_nd_update_(kv_cache[2].view(-1, k.shape[-1]),
                                             all_slots.view(
                                                 -1, 1),
                                             k.view(-1,
                                                    k.shape[-1]))  # b, s, n, d

        weights, _ = self.weights_proj(x)

        #block_table = attn_metadata.block_tables

        '''
        pad_size = get_sfa_tmp_result_context().pad_size
        sfa_sp_context = get_sfa_sp_context()
        dp_rank = get_dp_group().rank_in_group
        seq_lens_cp = attn_metadata.seq_lens.detach().clone()
        cum_query_lens_cp = attn_metadata.cum_query_lens.detach().clone()

        start_index = sfa_sp_context.dp_device_sp_start_token[dp_rank][self.tp_rank]
        num_actual_tokens = attn_metadata.num_actual_tokens
        if start_index >= num_actual_tokens:
            return None
        end_index = sfa_sp_context.dp_device_sp_end_token[dp_rank][self.tp_rank]
        seq_lens_cp[0] = end_index 
        cum_query_lens_cp[0] = 1
        if "model.layers.0.self_attn.attn" in layer_name and self.tp_rank<6:
            print("===cp-q,",q.shape,q,flush=True)
            print("===cp-k,",kv_cache[2].shape,kv_cache[2],flush=True)
            print("===cp-seq_lens,",seq_lens_cp.shape,seq_lens_cp,flush=True)
            print("===cp-cum_query_lens,",cum_query_lens_cp.shape,cum_query_lens_cp,flush=True)
            print("===cp-meta,",attn_metadata.block_tables.shape,attn_metadata.block_tables,flush=True)
        '''
        
        topk_indices = torch.ops.custom.npu_lightning_indexer(
            query=q,
            key=kv_cache[2],
            weights=weights,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key,
            block_table=block_tables,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=2048,
            sparse_mode=3)
        return topk_indices
