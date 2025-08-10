#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/vllm/worker/gpu_model_runner.py
#

import copy
import gc
import math
import os
import time
import types
import weakref
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt
import torch
import torch._dynamo.cache_size
import torch.distributed as dist
import torch.nn as nn
import torchair
from torchair import patch_for_hcom
from vllm.attention import AttentionType, get_attn_backend
from vllm.attention.layer import Attention
from vllm.config import CompilationLevel, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorBase_V1
from vllm.distributed.parallel_state import get_dp_group, get_pp_group
from vllm.forward_context import get_forward_context
from vllm.inputs import INPUT_REGISTRY
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.model_executor.model_loader import get_model
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalKwargs, PlaceholderRange
from vllm.multimodal.utils import group_mm_inputs_by_modality
from vllm.sampling_params import SamplingType
from vllm.sequence import IntermediateTensors
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, DeviceMemoryProfiler,
                        LayerBlockType, LazyLoader, cdiv)
from vllm.v1.core.encoder_cache_manager import compute_encoder_budget
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec)
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.spec_decode.utils import is_spec_decode_supported
from vllm.v1.utils import bind_kv_cache
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin
from vllm.v1.worker.utils import (gather_mm_placeholders,
                                  sanity_check_mm_encoder_outputs,
                                  scatter_mm_placeholders)

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.attention import AttentionMaskBuilder
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import \
    AscendCommonAttentionMetadata as CommonAttentionMetadata
from vllm_ascend.eplb.adaptor.vllm_adaptor import VllmEplbAdaptor
from vllm_ascend.eplb.eplb_updator import EplbUpdator
from vllm_ascend.multistream.ms_split import compute_split_seq_index
from vllm_ascend.platform import NPUPlatform
from vllm_ascend.sample.rejection_sampler import AscendRejectionSampler
from vllm_ascend.utils import ProfileExecuteDuration, _enable_lmhead_tp
from vllm_ascend.worker.mtp_proposer_v1 import MtpProposer

if TYPE_CHECKING:
    import xgrammar as xgr  # type: ignore[import-untyped]
    from vllm.v1.core.sched.output import SchedulerOutput
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")

import vllm.envs as envs_vllm

import vllm_ascend.envs as envs_ascend


@dataclass
class GraphCaptureContext:
    stream: torch.npu.Stream


@contextmanager
def graph_capture(device: torch.device):
    """
    `graph_capture` is a context manager which should surround the code that
    is capturing the NPU graph. Its main purpose is to ensure that the
    some operations will be run after the graph is captured, before the graph
    is replayed. It returns a `GraphCaptureContext` object which contains the
    necessary data for the graph capture. Currently, it only contains the
    stream that the graph capture is running on. This stream is set to the
    current NPU stream when the context manager is entered and reset to the
    default stream when the context manager is exited. This is to ensure that
    the graph capture is running on a separate stream from the default stream,
    in order to explicitly distinguish the kernels to capture
    from other kernels possibly launched on background in the default stream.
    """
    graph_capture_context = GraphCaptureContext(
        torch.npu.Stream(device=device))
    stream = graph_capture_context.stream

    # we use nullcontext now
    maybe_ca_context = nullcontext()

    # ensure all initialization operations complete before attempting to
    # capture the graph on another stream
    curr_stream = torch.npu.current_stream()
    if curr_stream != stream:
        stream.wait_stream(curr_stream)

    with torch.npu.stream(stream), maybe_ca_context:
        yield graph_capture_context


class NPUModelRunner(LoRAModelRunnerMixin):

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        ascend_config = get_ascend_config()
        if ascend_config.ascend_scheduler_config.enabled:
            self.chunked_prefill_enabled = self.scheduler_config.chunked_prefill_enabled
        else:
            self.chunked_prefill_enabled = True
        self.device = device

        self.is_multimodal_model = self.model_config.is_multimodal_model
        self.block_size = vllm_config.cache_config.block_size

        self.max_num_blocks_per_req = cdiv(self.model_config.max_model_len,
                                           self.block_size)
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_num_reqs = self.scheduler_config.max_num_seqs

        # Model-related.
        self.num_attn_layers = self.model_config.get_num_layers_by_block_type(
            vllm_config.parallel_config, LayerBlockType.attention)
        self.hidden_size = self.model_config.get_hidden_size()
        self.dtype = self.model_config.dtype
        cache_config = vllm_config.cache_config
        if cache_config.cache_dtype == "auto":
            self.kv_cache_dtype = self.dtype
        else:
            self.kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                cache_config.cache_dtype]

        self.head_size = self.model_config.get_head_size()
        self.attn_backend = get_attn_backend(
            self.head_size,
            self.dtype,
            self.kv_cache_dtype,
            self.block_size,
            self.model_config.is_attention_free,
            use_mla=self.model_config.use_mla,
        )
        if self.attn_backend is None:
            error_msg = (
                f"Error with get_att_backend: {self.head_size=}, "
                f"{self.dtype=}, {self.kv_cache_dtype=}, {self.block_size=}, "
                f"{self.model_config.is_attention_free=}, "
                f"{self.model_config.use_mla=}")
            logger.error(error_msg)
            raise NotImplementedError(
                "Non-Attention backend is not supported by V1 NPUModelRunner.")

        self.attn_metadata_builder = self.attn_backend.get_builder_cls()(
            weakref.proxy(self))

        # Multi-modal data support
        self.input_registry = INPUT_REGISTRY
        self.mm_registry = MULTIMODAL_REGISTRY
        self.uses_mrope = self.model_config.uses_mrope

        self.max_num_encoder_input_tokens, self.encoder_cache_size = compute_encoder_budget(
            model_config=self.model_config,
            scheduler_config=self.scheduler_config,
            mm_registry=self.mm_registry)

        # Lazy initialization
        # self.model: nn.Module  # Set after load_model
        self.kv_caches: List[torch.Tensor] = []
        # req_id -> (input_id -> encoder_output)
        self.encoder_cache: Dict[str, Dict[int, torch.Tensor]] = {}

        # Set up speculative decoding.
        self.use_spec_decode = False
        self.spec_attn_mask = None
        self.actual_seq_lengths_q = []
        self.spec_token_num = 0
        self.decode_token_per_req = 1
        if self.speculative_config:
            self.use_spec_decode = True
            self.spec_token_num = self.speculative_config.num_speculative_tokens
            assert self.spec_token_num > 0
            self.spec_attn_mask = torch.triu(torch.ones(2048,
                                                        2048,
                                                        dtype=torch.bool),
                                             diagonal=1).to("npu")
            if get_pp_group().is_last_rank:
                if self.speculative_config.method == "ngram":
                    self.drafter = NgramProposer(self.vllm_config)
                elif self.speculative_config.method == "eagle":
                    self.drafter = EagleProposer(self.vllm_config,
                                                 self.device)  # type: ignore
                elif self.speculative_config.method == 'deepseek_mtp':
                    self.drafter = MtpProposer(self.vllm_config, self)
                    self.decode_token_per_req = 1 + self.spec_token_num
                    self.actual_seq_lengths_q = [
                        len for len in
                        range(self.decode_token_per_req, self.max_num_tokens +
                              1, self.decode_token_per_req)
                    ]
                else:
                    raise ValueError("Unknown speculative decoding method: "
                                     f"{self.speculative_config.method}")
                self.rejection_sampler = AscendRejectionSampler()

        # Request states.
        self.requests: Dict[str, CachedRequestState] = {}
        # Persistent batch.

        self.input_ids = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int32,
                                     device=self.device)
        self.positions = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int64,
                                     device=self.device)
        self.query_start_loc = torch.zeros(self.max_num_reqs + 1,
                                           dtype=torch.int32,
                                           device=self.device)
        self.seq_lens = torch.zeros(self.max_num_reqs,
                                    dtype=torch.int32,
                                    device=self.device)
        self.slot_mapping = torch.zeros(self.max_num_tokens,
                                        dtype=torch.int32,
                                        device=self.device)
        self.query_lens = torch.zeros(self.max_num_reqs,
                                      dtype=torch.int32,
                                      device=self.device)
        # None in the first PP rank. The rest are set after load_model.
        self.intermediate_tensors: Optional[IntermediateTensors] = None

        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            # NOTE: `mrope_positions` is implemented with one additional dummy
            # position on purpose to make it non-contiguous so that it can work
            # with torch compile.
            # See detailed explanation in https://github.com/vllm-project/vllm/pull/12128#discussion_r1926431923

            # NOTE: When M-RoPE is enabled, position ids are 3D regardless of
            # the modality of inputs. For text-only inputs, each dimension has
            # identical position IDs, making M-RoPE functionally equivalent to
            # 1D-RoPE.
            # See page 5 of https://arxiv.org/abs/2409.12191
            self.mrope_positions = torch.zeros((3, self.max_num_tokens + 1),
                                               dtype=torch.int64,
                                               device=self.device)
            self.mrope_positions_cpu = torch.zeros(
                (3, self.max_num_tokens + 1),
                dtype=torch.int64,
                device="cpu",
                pin_memory=True)

        if self.is_multimodal_model:
            self.inputs_embeds = torch.zeros(
                (self.max_num_tokens, self.hidden_size),
                dtype=self.dtype,
                device=self.device)

        # OPTIMIZATION: Cache the tensors rather than creating them every step.
        self.arange_np: npt.NDArray[np.int32] = np.arange(max(
            self.max_num_reqs + 1, self.model_config.max_model_len,
            self.max_num_tokens),
                                                          dtype=np.int32)
        # NOTE(woosuk): These tensors are "stateless", i.e., they are literally
        # a faster version of creating a new tensor every time. Thus, we should
        # not make any assumptions about the values in these tensors.
        self.input_ids_cpu = torch.zeros(self.max_num_tokens,
                                         dtype=torch.int32,
                                         device="cpu",
                                         pin_memory=True)
        self.positions_cpu = torch.zeros(self.max_num_tokens,
                                         dtype=torch.int64,
                                         device="cpu",
                                         pin_memory=True)
        self.positions_np = self.positions_cpu.numpy()

        self.slot_mapping_cpu = torch.zeros(self.max_num_tokens,
                                            dtype=torch.int32,
                                            device="cpu",
                                            pin_memory=True)
        self.slot_mapping_np = self.slot_mapping_cpu.numpy()

        self.query_start_loc_cpu = torch.zeros(self.max_num_reqs + 1,
                                               dtype=torch.int32,
                                               device="cpu",
                                               pin_memory=True)
        self.query_start_loc_np = self.query_start_loc_cpu.numpy()

        self.seq_lens_cpu = torch.zeros(self.max_num_reqs,
                                        dtype=torch.int32,
                                        device="cpu",
                                        pin_memory=True)
        self.seq_lens_np = self.seq_lens_cpu.numpy()

        self.input_positions_cpu = torch.arange(0,
                                                self.max_num_tokens,
                                                device="cpu")
        self.attn_mask = None
        self.attn_state = None
        self.use_aclgraph = (self.vllm_config.compilation_config.level
                             == CompilationLevel.PIECEWISE
                             and not self.model_config.enforce_eager)
        self.aclgraph_batch_sizes = list(
            reversed(
                self.vllm_config.compilation_config.cudagraph_capture_sizes))

        # NOTE: Pre-construct a mask matrix to improve the efficiency of
        # attention mask construction during inference.
        # Note that the length of the matrix needs to be carefully balanced: a
        # matrix that is too large will consume excessive VRAM, while a matrix
        # that is too small will require dynamic concatenation during inference,
        # leading to performance degradation.
        # Therefore, an environment variable is added here to dynamically set
        # the size of the pre-constructed mask matrix based on requirements.
        mask_len = os.getenv("PAGED_ATTENTION_MASK_LEN", 10000)
        self.attn_mask_len = min(self.model_config.max_model_len,
                                 int(mask_len))
        self.attn_mask_builder = AttentionMaskBuilder.initialize_from_len(
            self.attn_mask_len, self.dtype)

        self.sampler = Sampler()

        self.torchair_compiled_model = None  # type: ignore
        self.torchair_compiled_models = {}  # type: ignore
        ascend_config = get_ascend_config()
        self.torchair_graph_enabled = ascend_config.torchair_graph_config.enabled
        self.use_cached_npu_graph = ascend_config.torchair_graph_config.use_cached_graph
        self.torchair_graph_batch_sizes = ascend_config.torchair_graph_config.graph_batch_sizes
        self.use_ring_mla = ascend_config.chunked_prefill_for_mla

        if ascend_config.torchair_graph_config.graph_batch_sizes_init:
            self.init_torchair_graph_batch_sizes()

        self.check_torchair_graph_batch_sizes()

        # graph_block_tables shape: [num_request, cell(max_model_len / block_size)]
        self.graph_block_tables = np.zeros(
            (self.torchair_graph_batch_sizes[-1] // self.decode_token_per_req,
             (self.model_config.max_model_len + self.block_size - 1) //
             self.block_size),
            dtype=np.int32)

        torch._dynamo.cache_size.config.cache_size_limit += len(
            self.torchair_graph_batch_sizes)
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        torch._logging.set_logs(
            recompiles=envs_ascend.VLLM_ASCEND_TRACE_RECOMPILES)

        self.dp_size = vllm_config.parallel_config.data_parallel_size
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank

        #EPLB
        self.dynamic_eplb = ascend_config.dynamic_eplb
        if self.dynamic_eplb:
            self.eplb_adaptor: Optional[VllmEplbAdaptor] = None
            self.is_eplb_warmuped = False
            self.eplb_updator = EplbUpdator(ascend_config.expert_map_path)

        # NOTE: we need to use `in_profile_run` to determine whether `enable_force_load_balance` is True
        self.in_profile_run = False

        # kv role
        self.is_kv_producer = False
        self.is_kv_consumer = False
        if vllm_config.kv_transfer_config is not None:
            self.is_kv_producer = vllm_config.kv_transfer_config.is_kv_producer
            self.is_kv_consumer = vllm_config.kv_transfer_config.is_kv_consumer

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The SamplingMetadata is updated and copied to the NPU if there is a
        new/resumed/paused/finished request in the batch.
        """
        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            self.encoder_cache.pop(req_id, None)
        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        removed_req_indices: List[int] = []
        for req_id in scheduler_output.finished_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            if req_index is not None:
                removed_req_indices.append(req_index)

        # Free the cached encoder outputs.
        for req_id, input_id in scheduler_output.free_encoder_input_ids:
            encoder_outputs = self.encoder_cache.get(req_id)
            if encoder_outputs is not None:
                encoder_outputs.pop(input_id, None)
                if not encoder_outputs:
                    self.encoder_cache.pop(req_id, None)

        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states since
        # they will be scheduled again sometime in the future.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        unscheduled_req_ids = cached_req_ids - scheduled_req_ids
        # NOTE(woosuk): The persistent batch optimization assumes that
        # consecutive batches contain mostly the same requests. If batches
        # have low request overlap (e.g., alternating between two distinct
        # sets of requests), this optimization becomes very inefficient.
        for req_id in unscheduled_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            assert req_index is not None
            removed_req_indices.append(req_index)

        req_ids_to_add: List[str] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            sampling_params = new_req_data.sampling_params
            if sampling_params.sampling_type == SamplingType.RANDOM_SEED:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                mm_inputs=new_req_data.mm_inputs,
                mm_positions=new_req_data.mm_positions,
                sampling_params=sampling_params,
                generator=generator,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )

            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            if self.uses_mrope:
                image_grid_thw = []
                video_grid_thw = []
                second_per_grid_ts = []
                audio_feature_lengths = []
                use_audio_in_video = False
                for mm_input in self.requests[req_id].mm_inputs:
                    if mm_input.get("image_grid_thw") is not None:
                        image_grid_thw.extend(
                            mm_input["image_grid_thw"].tolist())
                    if mm_input.get("video_grid_thw") is not None:
                        video_grid_thw.extend(
                            mm_input["video_grid_thw"].tolist())
                    if mm_input.get("second_per_grid_ts") is not None:
                        second_per_grid_ts.extend(
                            mm_input["second_per_grid_ts"])
                    if mm_input.get("audio_feature_lengths") is not None:
                        audio_feature_lengths.extend(
                            mm_input["audio_feature_lengths"])
                    if mm_input.get("use_audio_in_video") is True:
                        use_audio_in_video = True

                hf_config = self.model_config.hf_config

                self.requests[req_id].mrope_positions, \
                    self.requests[req_id].mrope_position_delta = \
                    MRotaryEmbedding.get_input_positions_tensor(
                        self.requests[req_id].prompt_token_ids,
                        hf_config=hf_config,
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw,
                        second_per_grid_ts=second_per_grid_ts,
                        audio_feature_lengths=audio_feature_lengths,
                        use_audio_in_video=use_audio_in_video,
                    )

            req_ids_to_add.append(req_id)

        # Update the states of the running/resumed requests.
        for req_data in scheduler_output.scheduled_cached_reqs:
            req_id = req_data.req_id
            req_state = self.requests[req_id]

            # Update the cached states.
            num_computed_tokens = req_data.num_computed_tokens
            req_state.num_computed_tokens = num_computed_tokens
            # Add the sampled token(s) from the previous step (if any).
            # This doesn't include "unverified" tokens like spec decode tokens.
            num_new_tokens = (num_computed_tokens +
                              len(req_data.new_token_ids) -
                              req_state.num_tokens)
            if num_new_tokens == 1:
                # Avoid slicing list in most common case.
                req_state.output_token_ids.append(req_data.new_token_ids[-1])
            elif num_new_tokens > 0:
                req_state.output_token_ids.extend(
                    req_data.new_token_ids[-num_new_tokens:])
            # Update the block IDs.
            if not req_data.resumed_from_preemption:
                # Append the new blocks to the existing block IDs.
                for block_ids, new_block_ids in zip(  # type: ignore[call-overload]
                        req_state.block_ids,
                        req_data.new_block_ids,
                        strict=True):
                    block_ids.extend(new_block_ids)
            else:
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = req_data.new_block_ids

            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.
                req_ids_to_add.append(req_id)
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = (
                num_computed_tokens)

            start_index = (len(req_state.block_ids) -
                           len(req_data.new_block_ids))
            self.input_batch.block_table.append_row(req_data.new_block_ids,
                                                    req_index)
            # Add new_token_ids to token_ids_cpu.
            start_token_index = num_computed_tokens
            end_token_index = num_computed_tokens + len(req_data.new_token_ids)
            self.input_batch.token_ids_cpu[
                req_index,
                start_token_index:end_token_index] = req_data.new_token_ids
            self.input_batch.num_tokens_no_spec[req_index] = end_token_index
            # Add spec_token_ids to token_ids_cpu.
            spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(
                req_id, ())
            if spec_token_ids:
                start_index = end_token_index
                end_token_index += len(spec_token_ids)
                self.input_batch.token_ids_cpu[
                    req_index, start_index:end_token_index] = spec_token_ids
            # NOTE(woosuk): `num_tokens` here may include spec decode tokens.
            self.input_batch.num_tokens[req_index] = end_token_index

        # Check if the batch has changed. If not, we can skip copying the
        # sampling metadata from CPU to GPU.
        batch_changed = len(removed_req_indices) > 0 or len(req_ids_to_add) > 0

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        removed_req_indices = sorted(removed_req_indices, reverse=True)
        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            if removed_req_indices:
                # Fill the empty index.
                req_index = removed_req_indices.pop()
            else:
                # Append to the end.
                req_index = None
            self.input_batch.add_request(req_state, req_index)
            spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(
                req_id, ())
            if spec_token_ids:
                req_index = self.input_batch.num_reqs - 1
                start_index = len(req_state.prompt_token_ids) + len(
                    req_state.output_token_ids)
                end_token_index = start_index + len(spec_token_ids)
                self.input_batch.token_ids_cpu[
                    req_index, start_index:end_token_index] = spec_token_ids
                self.input_batch.num_tokens[req_index] = end_token_index

        # Condense the batched states if there are empty indices.
        if removed_req_indices:
            self.input_batch.condense(removed_req_indices)

        if batch_changed:
            self.input_batch.refresh_sampling_metadata()

    def _get_forward_metadata_across_dp(
            self, maybe_padded_num_tokens: int, num_tokens: int,
            with_prefill: bool, enable_dbo: bool
    ) -> tuple[int, Optional[torch.Tensor], bool, bool]:
        if self.dp_size == 1:
            return maybe_padded_num_tokens, None, with_prefill, enable_dbo
        if self.is_kv_producer and not envs_ascend.VLLM_ASCEND_ENABLE_CHUNK_MC2:
            num_tokens_across_dp = torch.tensor([num_tokens] * self.dp_size,
                                                device="cpu",
                                                dtype=torch.int32)
            return num_tokens, num_tokens_across_dp, True, enable_dbo
        if self.is_kv_consumer and self.torchair_graph_enabled and len(
                self.torchair_graph_batch_sizes
        ) == 1 and not self.in_profile_run:
            max_num_decode_tokens = self.torchair_graph_batch_sizes[0]
            num_tokens_across_dp = torch.tensor([max_num_decode_tokens] *
                                                self.dp_size,
                                                device="cpu",
                                                dtype=torch.int32)
            return max_num_decode_tokens, num_tokens_across_dp, False, enable_dbo

        num_tokens_across_dp = [0] * self.dp_size * 2
        num_tokens_across_dp[self.dp_rank] = maybe_padded_num_tokens
        num_tokens_across_dp[self.dp_size + self.dp_rank] = num_tokens
        forward_metadata = torch.tensor(num_tokens_across_dp +
                                        [with_prefill, not enable_dbo],
                                        device="cpu",
                                        dtype=torch.int32)
        dist.all_reduce(forward_metadata, group=get_dp_group().cpu_group)
        with_prefill = bool(forward_metadata[-2])

        # NOTE: when with_prefill is false before all_reduce and true after all_reduce, we need to revert pad.
        if with_prefill:
            num_tokens_across_dp = forward_metadata[self.dp_size:self.dp_size *
                                                    2]
            maybe_padded_num_tokens = num_tokens
        else:
            num_tokens_across_dp = forward_metadata[:self.dp_size]

        # NOTE: when in torchair_graph_mode, we need to pad local_num_tokens to
        # `max_num_tokens_across_dp`, in other situation it is not necessary.
        if self.torchair_graph_enabled and not with_prefill:
            maybe_padded_num_tokens = torch.max(num_tokens_across_dp).item()
            num_tokens_across_dp = torch.tensor([maybe_padded_num_tokens] *
                                                self.dp_size,
                                                device="cpu",
                                                dtype=torch.int32)

        return maybe_padded_num_tokens, num_tokens_across_dp, with_prefill, not bool(
            forward_metadata[-1])

    def _check_dbo_is_valid(self, query_lens: torch.Tensor,
                            attn_state: AscendAttentionState,
                            num_tokens: int) -> bool:
        # do the checks for dp + dbo
        if attn_state in [
                AscendAttentionState.DecodeOnly,
                AscendAttentionState.SpecDecoding
        ]:
            return False
        # considering the case that one dp rank may enable dbo while others may not
        if not envs_ascend.VLLM_ASCEND_ENABLE_DBO:
            return False
        # TODO: remove it if token-level microbatch is enabled
        [token_index,
         seq_index] = compute_split_seq_index(query_lens, attn_state,
                                              num_tokens)
        if token_index == 0 or seq_index == 0 or seq_index == len(
                query_lens) or num_tokens < 256:
            return False
        return True

    def get_model(self) -> nn.Module:
        return self.model

    def _make_attention_mask(self, seq_lens, query_lens, position,
                             attn_state) -> torch.Tensor:
        # Chunk Prefill situation.
        if attn_state == AscendAttentionState.ChunkedPrefill:
            return self.attn_mask_builder.get_splitfuse_attn_mask(
                seq_lens, query_lens, position, self.dtype, self.device)
        # Prefill without cache situation.
        elif attn_state == AscendAttentionState.PrefillNoCache:
            max_seq_len = max(seq_lens, default=0)
            return self.attn_mask_builder.get_attn_mask(
                max_seq_len, self.dtype, self.device)
        # Prefill with cache hit.
        elif attn_state == AscendAttentionState.PrefillCacheHit:
            return self.attn_mask_builder.get_attn_mask(
                128, self.dtype, self.device)
        # Decode-only situation.
        else:
            return None

    def _calc_mrope_positions(self, scheduler_output: "SchedulerOutput"):
        mrope_pos_ptr = 0
        for index, req_id in enumerate(self.input_batch.req_ids):
            req = self.requests[req_id]
            assert req.mrope_positions is not None

            num_computed_tokens = \
                self.input_batch.num_computed_tokens_cpu[index]
            num_scheduled_tokens = \
                scheduler_output.num_scheduled_tokens[req_id]
            num_prompt_tokens = len(req.prompt_token_ids)

            if num_computed_tokens + num_scheduled_tokens > num_prompt_tokens:
                prompt_part_len = max(0,
                                      num_prompt_tokens - num_computed_tokens)
                completion_part_len = max(
                    0, num_scheduled_tokens - prompt_part_len)
            else:
                prompt_part_len = num_scheduled_tokens
                completion_part_len = 0

            assert num_scheduled_tokens == prompt_part_len + completion_part_len

            if prompt_part_len > 0:
                # prompt's mrope_positions are pre-computed
                dst_start = mrope_pos_ptr
                dst_end = mrope_pos_ptr + prompt_part_len
                src_start = num_computed_tokens
                src_end = num_computed_tokens + prompt_part_len

                self.mrope_positions_cpu[:, dst_start:dst_end] = \
                    req.mrope_positions[:,src_start:src_end]

                mrope_pos_ptr += prompt_part_len

            if completion_part_len > 0:
                # compute completion's mrope_positions on-the-fly
                dst_start = mrope_pos_ptr
                dst_end = mrope_pos_ptr + completion_part_len

                self.mrope_positions_cpu[:, dst_start:dst_end] = \
                    MRotaryEmbedding.get_next_input_positions_tensor(
                        req.mrope_position_delta,
                        context_len=num_computed_tokens +
                        prompt_part_len,
                        seq_len=num_computed_tokens +
                        prompt_part_len +
                        completion_part_len,
                    )

                mrope_pos_ptr += completion_part_len

    def _execute_mm_encoder(self, scheduler_output: "SchedulerOutput"):
        scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
        if not scheduled_encoder_inputs:
            return

        # Batch the multi-modal inputs.
        mm_inputs = list[MultiModalKwargs]()
        req_ids_pos = list[tuple[str, int, PlaceholderRange]]()
        for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
            req_state = self.requests[req_id]

            for mm_input_id in encoder_input_ids:
                mm_inputs.append(req_state.mm_inputs[mm_input_id])
                req_ids_pos.append(
                    (req_id, mm_input_id, req_state.mm_positions[mm_input_id]))

        # Batch mm inputs as much as we can: if a request in the batch has
        # multiple modalities or a different modality than the previous one,
        # we process it separately to preserve item order.
        # FIXME(ywang96): This is a hacky way to deal with multiple modalities
        # in the same batch while still being able to benefit from batching
        # multimodal inputs. The proper solution should be reordering the
        # encoder outputs.
        grouped_mm_inputs_list = group_mm_inputs_by_modality(mm_inputs)

        encoder_outputs = []
        for grouped_mm_inputs in grouped_mm_inputs_list:
            batched_mm_inputs = MultiModalKwargs.batch(grouped_mm_inputs)
            batched_mm_inputs = MultiModalKwargs.as_kwargs(batched_mm_inputs,
                                                           device=self.device)

            # Run the encoder.
            # `curr_group_outputs` is either of the following:
            # 1. A tensor of shape (num_items, feature_size, hidden_size)
            # in case feature_size is fixed across all multimodal items.
            # 2. A list or tuple (length: num_items) of tensors, each of shape
            # (feature_size, hidden_size) in case the feature size is dynamic
            # depending on the input multimodal items.
            curr_group_outputs = self.model.get_multimodal_embeddings(
                **batched_mm_inputs)

            sanity_check_mm_encoder_outputs(
                curr_group_outputs,
                expected_num_items=len(grouped_mm_inputs),
            )

            for output in curr_group_outputs:
                encoder_outputs.append(output)

        # Cache the encoder outputs.
        for (req_id, input_id, pos_info), output in zip(
                req_ids_pos,
                encoder_outputs,
        ):
            if req_id not in self.encoder_cache:
                self.encoder_cache[req_id] = {}

            self.encoder_cache[req_id][input_id] = scatter_mm_placeholders(
                output,
                is_embed=pos_info.is_embed,
            )

    def _gather_mm_embeddings(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> list[torch.Tensor]:
        mm_embeds: list[torch.Tensor] = []
        for req_id in self.input_batch.req_ids:
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[
                req_id]
            req_state = self.requests[req_id]
            num_computed_tokens = req_state.num_computed_tokens
            mm_positions = req_state.mm_positions
            for i, pos_info in enumerate(mm_positions):
                start_pos = pos_info.offset
                num_encoder_tokens = pos_info.length

                # The encoder output is needed if the two ranges overlap:
                # [num_computed_tokens,
                #  num_computed_tokens + num_scheduled_tokens) and
                # [start_pos, start_pos + num_encoder_tokens)
                if start_pos >= num_computed_tokens + num_scheduled_tokens:
                    # The encoder output is not needed in this step.
                    break
                if start_pos + num_encoder_tokens <= num_computed_tokens:
                    # The encoder output is already processed and stored
                    # in the decoder's KV cache.
                    continue

                start_idx = max(num_computed_tokens - start_pos, 0)
                end_idx = min(
                    num_computed_tokens - start_pos + num_scheduled_tokens,
                    num_encoder_tokens)
                assert start_idx < end_idx
                assert req_id in self.encoder_cache
                assert i in self.encoder_cache[req_id]
                encoder_output = self.encoder_cache[req_id][i]

                if (is_embed := pos_info.is_embed) is not None:
                    is_embed = is_embed[start_idx:end_idx]

                mm_embeds_item = gather_mm_placeholders(
                    encoder_output[start_idx:end_idx],
                    is_embed=is_embed,
                )
                mm_embeds.append(mm_embeds_item)
        return mm_embeds

    def _process_reqs(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> tuple[SpecDecodeMetadata, torch.Tensor, SpecDecodeMetadata,
               torch.Tensor, int, torch.Tensor, Optional[set[str]],
               Optional[set[str]]]:
        # Check input valid
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0
        if (self.use_aclgraph and
                total_num_scheduled_tokens <= self.aclgraph_batch_sizes[-1]):
            # Add padding to the batch size.
            num_input_tokens = self.vllm_config.pad_for_cudagraph(
                total_num_scheduled_tokens)
        else:
            # Eager mode.
            num_input_tokens = total_num_scheduled_tokens

        modified_batch = self.attn_metadata_builder.reorder_batch(
            self.input_batch, scheduler_output)
        if modified_batch:
            self.input_batch.refresh_sampling_metadata()

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        self.input_batch.block_table.commit(num_reqs)

        # Get the number of scheduled tokens for each request.
        # TODO: The Python loop can be slow. Optimize.
        num_scheduled_tokens = np.empty(num_reqs, dtype=np.int32)
        num_valid_tokens = np.empty(num_reqs, dtype=np.int32)
        max_num_scheduled_tokens = 0
        for i, req_id in enumerate(self.input_batch.req_ids):
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_scheduled_tokens[i] = num_tokens
            num_valid_tokens[i] = num_tokens - \
                len(scheduler_output.scheduled_spec_decode_tokens.get(req_id, []))
            max_num_scheduled_tokens = max(max_num_scheduled_tokens,
                                           num_tokens)

        # Hot-Swap lora model
        if self.lora_config:
            self.set_active_loras(self.input_batch, num_scheduled_tokens)

        # Prepare positions
        req_indices = np.repeat(self.arange_np[:num_reqs],
                                num_scheduled_tokens)
        cu_num_tokens = np.cumsum(num_scheduled_tokens)
        cumsums_offsets = np.repeat(cu_num_tokens - num_scheduled_tokens,
                                    num_scheduled_tokens)
        sample_indices = cu_num_tokens - 1
        sample_indices = torch.from_numpy(sample_indices).to(self.device,
                                                             non_blocking=True)
        arange = self.arange_np[:total_num_scheduled_tokens] - cumsums_offsets

        positions_np = self.positions_np[:total_num_scheduled_tokens]
        np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
               arange,
               out=positions_np)

        # Calculate M-RoPE positions.
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            self._calc_mrope_positions(scheduler_output)

        if self.uses_mrope:
            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            self.mrope_positions[:, :total_num_scheduled_tokens].copy_(
                self.mrope_positions_cpu[:, :total_num_scheduled_tokens],
                non_blocking=True)

        self.positions[total_num_scheduled_tokens:num_input_tokens].zero_()
        self.positions[:total_num_scheduled_tokens].copy_(
            self.positions_cpu[:total_num_scheduled_tokens], non_blocking=True)
        positions = self.positions[:num_input_tokens]
        self.query_lens = torch.from_numpy(num_scheduled_tokens)

        self.seq_lens_np[:num_reqs] = (
            self.input_batch.num_computed_tokens_cpu[:num_reqs] +
            num_scheduled_tokens)
        seq_lens = self.seq_lens_cpu[:num_reqs]

        block_table_indices = (req_indices * self.max_num_blocks_per_req +
                               positions_np // self.block_size)

        block_table_cpu = self.input_batch.block_table[0].get_cpu_tensor()
        block_numbers = block_table_cpu.flatten()[block_table_indices].numpy()
        block_offsets = positions_np % self.block_size
        np.add(block_numbers * self.block_size,
               block_offsets,
               out=self.slot_mapping_np[:total_num_scheduled_tokens])

        ascend_config = get_ascend_config()
        use_spec_decode = len(
            scheduler_output.scheduled_spec_decode_tokens) > 0
        if np.array_equal(self.seq_lens_np[:num_reqs], num_scheduled_tokens):
            attn_state = AscendAttentionState.PrefillNoCache
        # We assume it is the decode stage, where prefill occurs but only one token is not hit in cache.
        elif np.all(num_scheduled_tokens == 1):
            attn_state = AscendAttentionState.DecodeOnly
            if self.speculative_config and self.speculative_config.method == 'deepseek_mtp':
                # SpecDecoding now supports seq_len=1 and seq_len=2
                attn_state = AscendAttentionState.SpecDecoding
        # Speculative decoding.
        elif np.all(num_valid_tokens == 1):
            attn_state = AscendAttentionState.SpecDecoding
        # splitfuse
        elif not ascend_config.ascend_scheduler_config.enabled or self.chunked_prefill_enabled:
            attn_state = AscendAttentionState.ChunkedPrefill
        else:
            attn_state = AscendAttentionState.PrefillCacheHit

        # NOTE: when use ring_mla, attn_mask don't need to generate here.
        if not self.use_ring_mla or attn_state == AscendAttentionState.PrefillNoCache:
            attn_mask = self._make_attention_mask(
                seq_lens=seq_lens,
                query_lens=num_scheduled_tokens,
                position=positions,
                attn_state=attn_state)
            self.attn_mask = attn_mask
        self.attn_state = attn_state  # type: ignore

        extra_builder_kwargs = {}

        self.query_start_loc_np[0] = 0
        self.query_start_loc_np[1:num_reqs + 1] = cu_num_tokens
        self.query_start_loc[:num_reqs + 1].copy_(
            self.query_start_loc_cpu[:num_reqs + 1], non_blocking=True)
        self.seq_lens[:num_reqs].copy_(self.seq_lens_cpu[:num_reqs],
                                       non_blocking=True)
        self.slot_mapping[:total_num_scheduled_tokens].copy_(
            self.slot_mapping_cpu[:total_num_scheduled_tokens],
            non_blocking=True)

        # Fill unused with -1. Needed for reshape_and_cache
        self.slot_mapping[total_num_scheduled_tokens:].fill_(-1)
        self.seq_lens[num_reqs:].fill_(0)
        self.query_start_loc[num_reqs + 1:].fill_(-1)

        query_start_loc = self.query_start_loc[:num_reqs + 1]
        # Use host tensor, other wise error: tensor.hostData is null
        common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            seq_lens=self.seq_lens_cpu[:num_reqs])
        with_prefill = attn_state not in [
            AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding
        ]

        is_only_prefill = bool(np.all(num_valid_tokens != 1))
        extra_builder_kwargs['is_only_prefill'] = is_only_prefill

        enable_dbo = self._check_dbo_is_valid(self.query_lens.tolist(),
                                              attn_state,
                                              total_num_scheduled_tokens)

        maybe_padded_num_tokens = total_num_scheduled_tokens
        if self.torchair_graph_enabled and not with_prefill:
            maybe_padded_num_tokens = self.select_torchair_padded_batch_size(
                total_num_scheduled_tokens)
        (padded_num_tokens_across_dp, num_tokens_across_dp, with_prefill,
         enable_dbo) = self._get_forward_metadata_across_dp(
             maybe_padded_num_tokens, total_num_scheduled_tokens, with_prefill,
             enable_dbo)
        extra_builder_kwargs['enable_dbo_across_dp'] = enable_dbo

        # TODO(zzzzwwjj): this code need to refactor afterwards.
        self.with_prefill = with_prefill
        # Add num_token_pad_size and num_reqs_pad_size here for torchair graph mode
        if self.torchair_graph_enabled and not with_prefill:
            num_token_pad_size = padded_num_tokens_across_dp - total_num_scheduled_tokens
            num_reqs_pad_size = (
                padded_num_tokens_across_dp // self.decode_token_per_req -
                num_reqs)
            assert num_token_pad_size >= 0 and num_reqs_pad_size >= 0

            extra_builder_kwargs['num_token_pad_size'] = num_token_pad_size
            extra_builder_kwargs['num_reqs_pad_size'] = num_reqs_pad_size
            self.num_reqs_pad_size = num_reqs_pad_size
            self.num_token_pad_size = num_token_pad_size
        self.extra_builder_kwargs = extra_builder_kwargs
        self.num_tokens_across_dp = num_tokens_across_dp

        if self.vllm_config.model_config.use_mla:
            attn_metadata = self.attn_metadata_builder.build(  # type: ignore
                num_reqs=num_reqs,
                num_actual_tokens=total_num_scheduled_tokens,
                max_query_len=max_num_scheduled_tokens,
                common_attn_metadata=common_attn_metadata,
                common_prefix_len=None,
                **extra_builder_kwargs,
            )
        else:
            attn_metadata = self.attn_metadata_builder.build(  # type: ignore
                num_reqs=num_reqs,
                num_actual_tokens=total_num_scheduled_tokens,
                max_query_len=max_num_scheduled_tokens,
                common_attn_metadata=common_attn_metadata,
                common_prefix_len=None,
                **extra_builder_kwargs,
            )
        attn_metadata.num_input_tokens = num_input_tokens

        # Prepare input_ids
        token_indices = (positions_np +
                         req_indices * self.input_batch.token_ids_cpu.shape[1])
        torch.index_select(self.input_batch.token_ids_cpu_tensor.flatten(),
                           0,
                           torch.from_numpy(token_indices),
                           out=self.input_ids_cpu[:total_num_scheduled_tokens])
        # Copy the tensors to the NPU.
        self.input_ids[:total_num_scheduled_tokens].copy_(
            self.input_ids_cpu[:total_num_scheduled_tokens], non_blocking=True)

        # _prepare_inputs may reorder the batch, so we must gather multi
        # modal outputs after that to ensure the correct order
        if self.is_multimodal_model:
            # Run the multimodal encoder if any.
            self._execute_mm_encoder(scheduler_output)
            mm_embeds = self._gather_mm_embeddings(scheduler_output)
        else:
            mm_embeds = []

        if self.is_multimodal_model:
            # NOTE(woosuk): To unify token ids and soft tokens (vision
            # embeddings), we always use embeddings (rather than token ids)
            # as input to the multimodal model, even when the input is text.
            input_ids = self.input_ids[:total_num_scheduled_tokens]
            if mm_embeds:
                inputs_embeds = self.model.get_input_embeddings(
                    input_ids, mm_embeds)
            else:
                inputs_embeds = self.model.get_input_embeddings(input_ids)
            # TODO(woosuk): Avoid the copy. Optimize.
            self.inputs_embeds[:total_num_scheduled_tokens].copy_(
                inputs_embeds)
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
            input_ids = None
        else:
            # For text-only models, we use token ids as input.
            # While it is possible to use embeddings as input just like the
            # multimodal models, it is not desirable for performance since
            # then the embedding layer is not included in the ACL Graph.
            input_ids = self.input_ids[:num_input_tokens]
            inputs_embeds = None
        if self.uses_mrope:
            positions = self.mrope_positions[:, :num_input_tokens]

        if self.torchair_graph_enabled and not with_prefill:
            input_ids = self.input_ids[:padded_num_tokens_across_dp]
            positions = self.positions[:padded_num_tokens_across_dp]

        # Run forward pass
        # TODO(zzzzwwjj): check param `num_tokens_across_dp` later.
        with set_ascend_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=padded_num_tokens_across_dp,
                num_tokens_across_dp=num_tokens_across_dp,
                with_prefill=with_prefill,
                num_actual_tokens=total_num_scheduled_tokens):
            with ProfileExecuteDuration().capture_async("forward"):
                self.maybe_setup_kv_connector(scheduler_output)
                model_kwargs = {}
                if self.torchair_graph_enabled:
                    model_kwargs["kv_caches"] = self.kv_caches
                    model_kwargs["attn_metadata"] = attn_metadata
                if envs_ascend.VLLM_ASCEND_ENABLE_DBO and with_prefill:
                    model_kwargs["graph_enable"] = False  # type: ignore
                if self.torchair_graph_enabled and not with_prefill:
                    torch._dynamo.mark_static(input_ids)
                    torch._dynamo.mark_static(positions)
                    if not self.vllm_config.model_config.use_mla:
                        torch._dynamo.mark_static(attn_metadata.block_tables)
                    else:
                        torch._dynamo.mark_static(
                            attn_metadata.decode.block_table)
                        torch._dynamo.mark_static(
                            attn_metadata.decode.input_positions)
                        torch._dynamo.mark_static(attn_metadata.decode.sin)
                        torch._dynamo.mark_static(attn_metadata.decode.cos)
                    torch._dynamo.mark_static(attn_metadata.slot_mapping)
                    for kv in self.kv_caches:
                        assert isinstance(kv,
                                          tuple), "kv_cache must be a tuple"
                        if isinstance(kv, tuple):
                            torch._dynamo.mark_static(kv[0])
                            torch._dynamo.mark_static(kv[1])
                    compiled_model = self._get_torchair_lazy_compiled_model(
                        padded_num_tokens_across_dp)
                    hidden_states = compiled_model(
                        input_ids=input_ids,
                        positions=positions,
                        intermediate_tensors=intermediate_tensors,
                        inputs_embeds=inputs_embeds,
                        **model_kwargs)
                else:
                    assert self.model is not None
                    hidden_states = self.model(
                        input_ids=input_ids,
                        positions=positions,
                        intermediate_tensors=intermediate_tensors,
                        inputs_embeds=inputs_embeds,
                        **model_kwargs)

        self.maybe_wait_for_kv_save()
        finished_sending, finished_recving = self.get_finished_kv_transfer(
            scheduler_output)
        use_spec_decode = len(
            scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            # NOTE(woosuk): Due to chunked prefills, the batch may contain
            # partial requests. While we should not sample any token
            # from these partial requests, we do so for simplicity.
            # We will ignore the sampled tokens from the partial requests.
            # TODO: Support prompt logprobs.
            spec_decode_metadata = None
        else:
            # Get the number of draft tokens for each request.
            # Iterate over the dictionary rather than all requests since not all
            # requests have draft tokens.
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            for req_id, draft_token_ids in (
                    scheduler_output.scheduled_spec_decode_tokens.items()):
                req_idx = self.input_batch.req_id_to_index[req_id]
                num_draft_tokens[req_idx] = len(draft_token_ids)

            spec_decode_metadata = self._calc_spec_decode_metadata(
                num_draft_tokens, cu_num_tokens)
            sample_indices = spec_decode_metadata.logits_indices
            if _enable_lmhead_tp(): # 
                if not with_prefill:
                    max_num_reqs_across_dp = padded_num_tokens_across_dp
                else:
                    max_num_reqs_across_dp = self.max_num_reqs
                sample_indices = nn.functional.pad(
                    sample_indices,
                    (0, max_num_reqs_across_dp - sample_indices.shape[0]))

        return (attn_metadata, hidden_states, spec_decode_metadata, positions,
                total_num_scheduled_tokens, sample_indices, finished_sending,
                finished_recving)

    def _calc_spec_decode_metadata(
        self,
        num_draft_tokens: np.ndarray,
        cu_num_scheduled_tokens: np.ndarray,
    ) -> SpecDecodeMetadata:
        # Inputs:
        # cu_num_scheduled_tokens:  [  4, 104, 107, 207, 209]
        # num_draft_tokens:         [  3,   0,   2,   0,   1]
        # Outputs:
        # cu_num_draft_tokens:      [  3,   3,   5,   5,   6]
        # logits_indices:           [  0,   1,   2,   3, 103, 104, 105, 106,
        #                            206, 207, 208]
        # target_logits_indices:    [  0,   1,   2,   5,   6,   9]
        # bonus_logits_indices:     [  3,   4,   7,   8,  10]

        # Compute the logits indices.
        # [4, 1, 3, 1, 2]
        num_sampled_tokens: Any = num_draft_tokens + 1
        # Step 1. [4, 5, 8, 9, 11]
        cu_num_sampled_tokens = np.cumsum(num_sampled_tokens, dtype=np.int32)
        total_num_sampled_tokens = cu_num_sampled_tokens[-1]
        # Step 2. [0, 0, 0, 0, 4, 5, 5, 5, 8, 9, 9]
        cumsums_offsets = np.repeat(cu_num_sampled_tokens - num_sampled_tokens,
                                    num_sampled_tokens)
        # Step 3. [0, 1, 2, 3, 0, 0, 1, 2, 0, 0, 1]
        arange = self.arange_np[:total_num_sampled_tokens] - cumsums_offsets
        # Step 4. [0, 0, 0, 0, 103, 104, 104, 104, 206, 207, 207]
        logits_indices = np.repeat(
            cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens)
        # Step 5. [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208]
        logits_indices += arange

        # Compute the bonus logits indices.
        bonus_logits_indices = cu_num_sampled_tokens - 1

        # Compute the draft logits indices.
        # [3, 3, 5, 5, 6]
        cu_num_draft_tokens = np.cumsum(num_draft_tokens, dtype=np.int32)
        total_num_draft_tokens = cu_num_draft_tokens[-1]
        # [0, 0, 0, 3, 3, 5]
        cumsums_offsets = np.repeat(cu_num_draft_tokens - num_draft_tokens,
                                    num_draft_tokens)
        # [0, 1, 2, 0, 1, 0]
        arange = self.arange_np[:total_num_draft_tokens] - cumsums_offsets
        # [0, 0, 0, 5, 5, 9]
        target_logits_indices = np.repeat(
            cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens)
        # [0, 1, 2, 5, 6, 9]
        target_logits_indices += arange

        # TODO: Optimize the CPU -> NPU copy.
        cu_num_draft_tokens = torch.from_numpy(cu_num_draft_tokens).to(
            self.device, non_blocking=True)
        logits_indices = torch.from_numpy(logits_indices).to(self.device,
                                                             non_blocking=True)
        target_logits_indices = torch.from_numpy(target_logits_indices).to(
            self.device, non_blocking=True)
        bonus_logits_indices = torch.from_numpy(bonus_logits_indices).to(
            self.device, non_blocking=True)

        # Compute the draft token ids.
        # draft_token_indices:      [  1,   2,   3, 105, 106, 208]
        draft_token_ids = self.input_ids[logits_indices]
        draft_token_ids = draft_token_ids[target_logits_indices + 1]

        metadata = SpecDecodeMetadata(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens.tolist(),
            cu_num_draft_tokens=cu_num_draft_tokens,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )
        return metadata

    def apply_grammar_bitmask(
        self,
        scheduler_output: "SchedulerOutput",
        logits: torch.Tensor,
    ) -> torch.Tensor:
        # Serialization of np.ndarray is much more efficient than a tensor,
        # so we receive it in that format.
        grammar_bitmask = scheduler_output.grammar_bitmask
        if grammar_bitmask is None:
            return

        # We receive the structured output bitmask from the scheduler, but the
        # indices of the requests in the batch may not match the indices of
        # the bitmask since the scheduler doesn't know how the gpu runner is
        # ordering the requests in the batch. We need to sort the bitmask to
        # match the order of the requests used here.
        struct_out_req_batch_indices: dict[str, int] = {}
        indices_match = True
        for req_id in self.input_batch.req_ids:
            mask_index = scheduler_output.structured_output_request_ids.get(
                req_id)
            if mask_index is None:
                # not a structured output request
                continue
            batch_index = self.input_batch.req_id_to_index[req_id]
            if batch_index != mask_index:
                indices_match = False
            struct_out_req_batch_indices[req_id] = batch_index

        if not indices_match:
            # Sort the bitmask to match the order of the requests
            sorted_bitmask = np.zeros_like(grammar_bitmask)
            for req_id, batch_index in struct_out_req_batch_indices.items():
                orig_index = scheduler_output.structured_output_request_ids[
                    req_id]
                sorted_bitmask[batch_index] = grammar_bitmask[orig_index]
            grammar_bitmask = sorted_bitmask

        grammar_bitmask = torch.from_numpy(grammar_bitmask)

        # TODO: compatibility with spec decode.
        # NOTE:
        # 1. XGrammar bitmask applying only supports CPU and GPU.
        # 2. The logits and bitmask should be on the same device.
        # 3. XGrammar logits on CPU only supports float32 dtype.
        logits_dtype = logits.dtype
        logits = logits.to("cpu").float()
        xgr.apply_token_bitmask_inplace(
            logits,
            grammar_bitmask,
            indices=list(struct_out_req_batch_indices.values()),
        )
        return logits.to(self.device).to(logits_dtype)

    def _get_spec_token_ids(
        self,
        valid_sampled_token_ids: list[list[int]],
        sampling_metadata: SamplingMetadata,
        scheduler_output: "SchedulerOutput",
        spec_decode_metadata: SpecDecodeMetadata,
        positions: torch.Tensor,
        num_scheduled_tokens: int,
        hidden_states: torch.Tensor,
        attn_metadata: SpecDecodeMetadata,
    ) -> Optional[list[list[int]]]:
        if not self.use_spec_decode:
            # Speculative decoding is not enabled.
            spec_token_ids = None
        elif self.speculative_config.method == "ngram":
            assert isinstance(self.drafter, NgramProposer)
            spec_token_ids = self._generate_draft_token_ids(
                valid_sampled_token_ids, sampling_metadata)
        elif self.speculative_config.method == "eagle":
            raise NotImplementedError(
                "eagle method for spec decode doesn't work on vllm-ascend currently"
            )
        elif self.speculative_config.method == 'deepseek_mtp':
            assert isinstance(self.drafter, MtpProposer)
            spec_token_ids = self._generate_mtp_token_ids(
                valid_sampled_token_ids, sampling_metadata, scheduler_output,
                spec_decode_metadata, positions, num_scheduled_tokens,
                hidden_states, attn_metadata)
        return spec_token_ids

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, torch.Tensor]:
        with ProfileExecuteDuration().capture_async(
                "prepare input and forward"):
            self._update_states(scheduler_output)
            if not scheduler_output.total_num_scheduled_tokens:
                if not has_kv_transfer_group():
                    logger.debug(
                        "skip this step for we receive the data from remote disaggregate prefill node"
                    )
                    # Return empty ModelRunnerOuptut if there's no work to do.
                    return EMPTY_MODEL_RUNNER_OUTPUT
                return self.kv_connector_no_forward(scheduler_output)

            if self.dynamic_eplb:
                self.eplb_updator.forward_before()

            (attn_metadata, hidden_states, spec_decode_metadata, positions,
             num_scheduled_tokens, sample_indices, finished_sending,
             finished_recving) = (self._process_reqs(scheduler_output,
                                                     intermediate_tensors))

            if self.dynamic_eplb:
                self.eplb_updator.take_update_info_from_eplb_process()

        with ProfileExecuteDuration().capture_async("post process"):
            logits = self.model.compute_logits(hidden_states[sample_indices],
                                               None)

            # Apply structured output bitmasks if present
            if scheduler_output.grammar_bitmask is not None:
                logits = self.apply_grammar_bitmask(scheduler_output, logits)

            # Sample the next token and get logprobs if needed.
            sampling_metadata = self.input_batch.sampling_metadata
            if spec_decode_metadata is None:
                if _enable_lmhead_tp():
                    logits = logits[:self.input_batch.num_reqs]
                sampler_output = self.sampler(
                    logits=logits,
                    sampling_metadata=sampling_metadata,
                )
            else:
                if _enable_lmhead_tp():
                    logits = logits[:len(spec_decode_metadata.logits_indices)]
                # When indexing with a tensor (bonus_logits_indices), PyTorch
                # creates a new tensor with separate storage from the original
                # logits tensor. This means any in-place operations on bonus_logits
                # won't affect the original logits tensor.
                bonus_logits = logits[
                    spec_decode_metadata.bonus_logits_indices]
                sampler_output = self.sampler(
                    logits=bonus_logits,
                    sampling_metadata=sampling_metadata,
                )
                bonus_token_ids = sampler_output.sampled_token_ids

                # Just like `bonus_logits`, `target_logits` is a new tensor with
                # separate storage from the original `logits` tensor. Therefore,
                # it is safe to update `target_logits` in place.
                target_logits = logits[
                    spec_decode_metadata.target_logits_indices]
                output_token_ids = self.rejection_sampler(
                    spec_decode_metadata,
                    None,  # draft_probs
                    target_logits,
                    bonus_token_ids,
                    sampling_metadata,
                )
                sampler_output.sampled_token_ids = output_token_ids

            # TODO(woosuk): The following loop can be slow since it iterates over
            # the requests one by one. Optimize.
            discard_sampled_tokens_req_indices = []
            for i, req_id in enumerate(self.input_batch.req_ids):
                req_state = self.requests[req_id]
                seq_len = (req_state.num_computed_tokens +
                           scheduler_output.num_scheduled_tokens[req_id])
                if seq_len < req_state.num_tokens:
                    # Ignore the sampled token.
                    # Rewind the generator state as if the token was not sampled.
                    generator = self.input_batch.generators.get(i)
                    if generator is not None:
                        generator.set_offset(generator.get_offset() - 4)
                    discard_sampled_tokens_req_indices.append(i)

            # NOTE: NPU -> CPU Sync happens here.
            # Move as many CPU operations as possible before this sync point.
            logprobs_tensors = sampler_output.logprobs_tensors
            logprobs_lists = logprobs_tensors.tolists() \
                if logprobs_tensors is not None else None

            # Get the valid generated tokens.
            sampled_token_ids = sampler_output.sampled_token_ids
            max_gen_len = sampled_token_ids.shape[-1]
            if max_gen_len == 1:
                # No spec decode tokens.
                valid_sampled_token_ids = sampled_token_ids.tolist()
            else:
                # Includes spec decode tokens.
                valid_sampled_token_ids = self.rejection_sampler.parse_output(
                    sampled_token_ids,
                    self.input_batch.vocab_size,
                )

            for i in discard_sampled_tokens_req_indices:
                valid_sampled_token_ids[i].clear()

            spec_token_ids = self._get_spec_token_ids(
                valid_sampled_token_ids,
                sampling_metadata,
                scheduler_output,
                spec_decode_metadata,
                positions,
                num_scheduled_tokens,
                hidden_states,
                attn_metadata,
            )
            if has_kv_transfer_group():
                get_kv_transfer_group().clear_connector_metadata()

            model_runner_output = ModelRunnerOutput(
                req_ids=self.input_batch.req_ids,
                req_id_to_index=self.input_batch.req_id_to_index,
                sampled_token_ids=valid_sampled_token_ids,
                spec_token_ids=spec_token_ids,
                logprobs=logprobs_lists,
                prompt_logprobs_dict={},
                finished_sending=finished_sending,
                finished_recving=finished_recving,
            )

        durations = ProfileExecuteDuration().pop_captured_sync()
        if durations:
            dr_str = [
                f"[{tag}]:{duration:.2f}ms"
                for tag, duration in durations.items()
            ]
            captured_name = "Decode" if self.attn_state == AscendAttentionState.DecodeOnly else "Prefill"
            logger.info("Profile execute duration [%s]:%s", captured_name,
                        " ".join(dr_str))

        if self.dynamic_eplb:
            self.eplb_updator.forward_end()

        return model_runner_output

    def kv_connector_no_forward(
            self, scheduler_output: "SchedulerOutput") -> ModelRunnerOutput:
        # TODO(zzzzwwjj): Check whether `set_ascend_forward_context` has influence with kv_connector or not.
        with set_ascend_forward_context(None, self.vllm_config):
            self.maybe_setup_kv_connector(scheduler_output)
            finsihed_sending, finished_recving = (
                self.get_finished_kv_transfer(scheduler_output))
            # For the case of no forward caused by receiving remote kv,
            # one round of dummy inference is necessary
            # to prevent hang over the collective calls.
        if not finsihed_sending and not finished_recving:
            return EMPTY_MODEL_RUNNER_OUTPUT

        output = copy.copy(EMPTY_MODEL_RUNNER_OUTPUT)
        output.finished_sending = finsihed_sending
        output.finished_recving = finished_recving
        return output

    @staticmethod
    def maybe_setup_kv_connector(scheduler_output: "SchedulerOutput"):
        # Update KVConnector with the KVConnector metadata forward().
        if has_kv_transfer_group():
            kv_connector = get_kv_transfer_group()
            assert isinstance(kv_connector, KVConnectorBase_V1)
            assert scheduler_output.kv_connector_metadata is not None
            kv_connector.bind_connector_metadata(
                scheduler_output.kv_connector_metadata)

            kv_connector.start_load_kv(get_forward_context())

    @staticmethod
    def maybe_wait_for_kv_save() -> None:
        if has_kv_transfer_group():
            get_kv_transfer_group().wait_for_save()

    @staticmethod
    def get_finished_kv_transfer(
        scheduler_output: "SchedulerOutput",
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        if has_kv_transfer_group():
            return get_kv_transfer_group().get_finished(
                scheduler_output.finished_req_ids)
        return None, None

    def _profile_multimodal(self) -> None:
        # TODO: handle encoder-decoder models once we support them.
        # NOTE: Currently model is profiled with a single non-text
        # modality with the max possible input tokens even when
        # it supports multiple.

        if (not self.is_multimodal_model
                or self.max_num_encoder_input_tokens <= 0
                or self.encoder_cache_size <= 0):
            return

        max_tokens_by_modality_dict = (
            MULTIMODAL_REGISTRY.get_max_tokens_per_item_by_nonzero_modality(
                self.model_config))
        dummy_data_modality, max_tokens_per_mm_item = max(
            max_tokens_by_modality_dict.items(), key=lambda item: item[1])

        # Check how many items of this modality can be supported by
        # the encoder budget.
        encoder_budget = min(self.max_num_encoder_input_tokens,
                             self.encoder_cache_size)

        max_num_mm_items_encoder_budget = cdiv(encoder_budget,
                                               max_tokens_per_mm_item)

        # Check how many items of this modality can be supported by
        # the decoder budget.
        max_mm_items_per_req = self.mm_registry.get_mm_limits_per_prompt(
            self.model_config)[dummy_data_modality]

        # NOTE: We do not consider max_num_batched_tokens on purpose
        # because the multimodal embeddings can be generated in advance
        # and chunked prefilled.
        max_num_mm_items_decoder_budget = self.max_num_reqs * \
            max_mm_items_per_req

        max_num_mm_items = min(max_num_mm_items_encoder_budget,
                               max_num_mm_items_decoder_budget)

        logger.info(
            "Encoder cache will be initialized with a budget of %s tokens,"
            " and profiled with %s %s items of the maximum feature size.",
            encoder_budget, max_num_mm_items, dummy_data_modality)

        # Create dummy batch of multimodal inputs.
        dummy_request_data = self.input_registry.dummy_data_for_profiling(
            model_config=self.model_config,
            seq_len=self.max_num_tokens,
            mm_registry=self.mm_registry,
        )
        dummy_mm_data = dummy_request_data.multi_modal_data

        if not isinstance(dummy_mm_data, MultiModalKwargs):
            # TODO: Delete this check once input mapper is fully removed.
            raise RuntimeError("Legacy input mapper is not supported in V1")

        # Dummy data definition in V0 may contain multiple multimodal items
        # (e.g, multiple images) for a single request, therefore here we
        # always replicate first item by max_num_mm_items times since in V1
        # they are scheduled to be processed separately.

        dummy_mm_item = dummy_mm_data.get_item(modality=dummy_data_modality,
                                               item_index=0)
        dummy_mm_kwargs = MultiModalKwargs.from_items([dummy_mm_item])

        batched_dummy_mm_inputs = MultiModalKwargs.batch([dummy_mm_kwargs] *
                                                         max_num_mm_items)
        batched_dummy_mm_inputs = MultiModalKwargs.as_kwargs(
            batched_dummy_mm_inputs, device=self.device)

        # Run multimodal encoder.
        dummy_encoder_outputs = self.model.get_multimodal_embeddings(
            **batched_dummy_mm_inputs)
        assert len(dummy_encoder_outputs) == max_num_mm_items, (
            "Expected dimension 0 of encoder outputs to match the number "
            f"of multimodal data items: {max_num_mm_items}, got "
            f"{len(dummy_encoder_outputs)=} instead. This is most likely "
            "due to the 'get_multimodal_embeddings' method of the model "
            "not implemented correctly.")

        # Cache the dummy encoder outputs.
        self.encoder_cache["tmp"] = dict(enumerate(dummy_encoder_outputs))

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        skip_attn: bool = True,
        with_prefill: bool = False,
        is_torchair_compile: bool = False,
        attn_state: AscendAttentionState = AscendAttentionState.DecodeOnly,
    ) -> torch.Tensor:
        maybe_padded_num_tokens = num_tokens
        if self.torchair_graph_enabled and not with_prefill:
            maybe_padded_num_tokens = self.select_torchair_padded_batch_size(
                num_tokens)

        # For kv producer, with prefill always true
        if self.is_kv_producer:
            with_prefill = True
        # Padding for DP
        (num_tokens, num_tokens_across_dp, with_prefill,
         enable_dbo) = self._get_forward_metadata_across_dp(
             maybe_padded_num_tokens, num_tokens, with_prefill, False)

        # Set num_scheduled_tokens based on num_tokens and max_num_seqs
        # for dummy run with LoRA so that the num_reqs collectively
        # has num_tokens in total.
        assert num_tokens <= self.scheduler_config.max_num_batched_tokens
        max_num_reqs = self.scheduler_config.max_num_seqs
        num_reqs = math.ceil(num_tokens / self.decode_token_per_req)
        if with_prefill:
            num_reqs = min(num_tokens, max_num_reqs)
        else:
            num_reqs = (num_tokens + self.decode_token_per_req -
                        1) // self.decode_token_per_req
        min_tokens_per_req = num_tokens // num_reqs
        num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
        num_scheduled_tokens_list[-1] += num_tokens % num_reqs
        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list,
                                        dtype=np.int32)

        # NOTE: If torchair graph mode and not with_prefill,
        # we can't skip_attn, it will cause graph recompile.
        if self.torchair_graph_enabled and not with_prefill:
            attn_metadata = self.attn_metadata_builder.build_torchair_graph_dummy(
                num_reqs=num_reqs, num_actual_tokens=1)
        elif skip_attn:
            attn_metadata = None
        else:
            attn_metadata = self.attn_metadata_builder.build_dummy_metadata(
                num_actual_tokens=num_tokens,
                num_reqs=num_reqs,
                num_scheduled_tokens=num_scheduled_tokens,
                attn_state=attn_state,
            )

        if not is_torchair_compile and not self.in_profile_run and self.dynamic_eplb:
            self.eplb_updator.forward_before()

        with self.maybe_dummy_run_with_lora(self.lora_config,
                                            num_scheduled_tokens):
            model = self.model
            if self.is_multimodal_model:
                input_ids = None
                inputs_embeds = self.inputs_embeds[:num_tokens]
            else:
                input_ids = self.input_ids[:num_tokens]
                inputs_embeds = None

            if self.uses_mrope:
                positions = self.mrope_positions[:, :num_tokens]
            else:
                positions = self.positions[:num_tokens]

            if get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                if self.intermediate_tensors is None:
                    self.intermediate_tensors = (
                        self.model.make_empty_intermediate_tensors(
                            batch_size=num_tokens,
                            dtype=self.dtype,
                            device=self.device))
                intermediate_tensors = IntermediateTensors({
                    k: v[:num_tokens]
                    for k, v in self.intermediate_tensors.items()
                })

            with set_ascend_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens,
                    num_tokens_across_dp=num_tokens_across_dp,
                    with_prefill=with_prefill,
                    in_profile_run=self.in_profile_run,
                    num_actual_tokens=0,
            ):
                model_kwargs = {}
                if self.torchair_graph_enabled and not with_prefill:
                    # Only mark static while compiling
                    if is_torchair_compile:
                        torch._dynamo.mark_static(input_ids)
                        torch._dynamo.mark_static(positions)
                        if not self.vllm_config.model_config.use_mla:
                            torch._dynamo.mark_static(
                                attn_metadata.block_tables)
                        else:
                            torch._dynamo.mark_static(
                                attn_metadata.decode.block_table)
                            torch._dynamo.mark_static(
                                attn_metadata.decode.input_positions)
                            torch._dynamo.mark_static(attn_metadata.decode.sin)
                            torch._dynamo.mark_static(attn_metadata.decode.cos)
                            torch._dynamo.mark_static(
                                get_forward_context().mc2_mask)
                        torch._dynamo.mark_static(attn_metadata.slot_mapping)
                        if self.speculative_config:
                            torch._dynamo.mark_static(
                                attn_metadata.decode.attn_mask)
                        for kv in self.kv_caches:
                            assert isinstance(
                                kv, tuple), "kv_cache must be a tuple"
                            if isinstance(kv, tuple):
                                torch._dynamo.mark_static(kv[0])
                                torch._dynamo.mark_static(kv[1])
                    compiled_model = self._get_torchair_lazy_compiled_model(
                        num_tokens)
                    model_kwargs["kv_caches"] = self.kv_caches
                    model_kwargs["attn_metadata"] = attn_metadata
                    if envs_ascend.VLLM_ASCEND_ENABLE_DBO:
                        model_kwargs["graph_enable"] = True  # type: ignore
                    hidden_states = compiled_model(
                        input_ids=input_ids,
                        positions=positions,
                        intermediate_tensors=intermediate_tensors,
                        inputs_embeds=None,
                        **model_kwargs,
                    )
                else:
                    if envs_ascend.VLLM_ASCEND_ENABLE_DBO:
                        model_kwargs["graph_enable"] = False  # type: ignore
                    hidden_states = model(
                        input_ids=input_ids,
                        positions=positions,
                        intermediate_tensors=intermediate_tensors,
                        inputs_embeds=inputs_embeds,
                        **model_kwargs)
                    
            if _enable_lmhead_tp() and not self.in_profile_run:
                if not with_prefill:
                    max_num_reqs_across_dp = num_reqs
                else:
                    max_num_reqs_across_dp = max_num_reqs
                dummy_indices = torch.zeros(max_num_reqs_across_dp,
                                            device=hidden_states.device,
                                            dtype=torch.int32)
                model.compute_logits(hidden_states[dummy_indices], None)

            if self.speculative_config and self.speculative_config.method == "deepseek_mtp":
                assert isinstance(self.drafter, MtpProposer)
                self.drafter.dummy_run(
                    num_tokens=num_tokens,
                    with_prefill=with_prefill,
                    skip_attn=skip_attn,
                    num_reqs=num_reqs,
                    num_tokens_across_dp=num_tokens_across_dp)
                if not self.in_profile_run and _enable_lmhead_tp():
                    if not with_prefill:
                        max_num_reqs_across_dp = num_reqs
                    else:
                        max_num_reqs_across_dp = max_num_reqs
                    dummy_indices = torch.zeros(max_num_reqs_across_dp,
                                                device=hidden_states.device,
                                                dtype=torch.int32)
                    model.compute_logits(hidden_states[dummy_indices], None)

            if self.in_profile_run and self.dynamic_eplb:
                self.model.clear_all_moe_loads()
            if not is_torchair_compile and not self.in_profile_run and self.dynamic_eplb:
                self.eplb_updator.take_update_info_from_eplb_process()
                self.eplb_updator.forward_end()

            return hidden_states

    @contextmanager
    def set_in_profile_run(self):
        self.in_profile_run = True
        try:
            yield
        finally:
            self.in_profile_run = False

    def profile_run(self) -> None:
        # FIXME Profile with multimodal encoder & encoder cache.
        # current _profile_multimodal() using PyTorch SDPA backend method not
        # support for window/full attn to reduce Memcpy operations, so will cause
        # Out Of Memory problem, so we currently don't use self._profile_multimodal()
        # self._profile_multimodal()

        # For profile, have maximum num_reqs and that collectively have
        # maximum num_tokens.
        num_reqs = self.scheduler_config.max_num_seqs
        num_tokens = self.max_num_tokens
        min_tokens_per_req = num_tokens // num_reqs

        num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
        num_scheduled_tokens_list[-1] += num_tokens % num_reqs
        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs

        num_scheduled_tokens = np.array(num_scheduled_tokens_list,
                                        dtype=np.int32)
        logit_indices = np.cumsum(num_scheduled_tokens) - 1

        # assert self.lora_manager is not None, "LoRA is not enabled"
        # TODO: call maybe_profile_with_lora()

        # Trigger compilation for general shape.
        with self.set_in_profile_run():
            hidden_states = self._dummy_run(self.max_num_tokens,
                                            with_prefill=True)

        if get_pp_group().is_last_rank:
            hidden_states = hidden_states[logit_indices]
            logits = self.model.compute_logits(hidden_states, None)
        else:
            logits = None

        NPUPlatform.synchronize()
        del hidden_states, logits
        self.encoder_cache.clear()
        gc.collect()

    def eplb_warmup(self):
        #EPLB
        if self.dynamic_eplb and not self.is_eplb_warmuped:
            self.is_eplb_warmuped = True
            self.eplb_adaptor = VllmEplbAdaptor(model=self.model)
            self.eplb_updator.set_adaptor(self.eplb_adaptor)
            self.eplb_updator.warm_up_eplb()

    def load_model(self) -> None:
        logger.info("Starting to load model %s...", self.model_config.model)

        with DeviceMemoryProfiler() as m:  # noqa: SIM117
            self.model = get_model(vllm_config=self.vllm_config)
            if hasattr(self, "drafter"):
                logger.info("Loading drafter model...")
                self.drafter.load_model()
            if self.lora_config:
                self.model = self.load_lora_model(self.model,
                                                  self.model_config,
                                                  self.scheduler_config,
                                                  self.lora_config,
                                                  self.device)
        logger.info("Loading model weights took %.4f GB",
                    m.consumed_memory / float(2**30))

    def _get_torchair_lazy_compiled_model(self, batch_size: int):
        if batch_size < 0 or batch_size > self.torchair_graph_batch_sizes[-1]:
            raise ValueError(
                f"Bad graph batch size:{batch_size}! max_graph_batch_sizes:{self.torchair_graph_batch_sizes[-1]}"
            )

        compiled_model = self.torchair_compiled_models.get(
            batch_size
        ) if self.use_cached_npu_graph else self.torchair_compiled_model

        if compiled_model:
            return compiled_model

        patch_for_hcom()
        config = torchair.CompilerConfig()
        config.experimental_config.frozen_parameter = True
        config.experimental_config.tiling_schedule_optimize = True
        config.experimental_config.enable_view_optimize = \
        get_ascend_config().torchair_graph_config.enable_view_optimize
        torch.npu.set_compile_mode(jit_compile=False)
        if not self.use_cached_npu_graph:
            npu_backend = torchair.get_npu_backend(compiler_config=config)
            self.torchair_compiled_model = torch.compile(
                self.model,
                dynamic=True,
                fullgraph=envs_vllm.VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE,
                backend=npu_backend)
            return self.torchair_compiled_model
        else:
            # Generate a new forward proxy code object to prevent the invalidation of
            # compilation cache caused by dynamo retracing
            forward_proxy_name = f"{self.model.__class__.__name__}_forward_with_batch_size_{batch_size}"
            forward_fn = self.model.forward
            code = forward_fn.__code__
            # Mark code object with a new proxy name
            modified_code = code.replace(co_name=forward_proxy_name, )

            modified_func = types.FunctionType(modified_code,
                                               forward_fn.__globals__,
                                               name=forward_proxy_name,
                                               argdefs=forward_fn.__defaults__)

            self.model.__dict__[forward_proxy_name] = modified_func.__get__(
                self.model, nn.Module)
            self.torchair_compiled_models[
                batch_size] = torchair.inference.cache_compile(
                    self.model.__dict__[forward_proxy_name],
                    dynamic=True,
                    fullgraph=envs_vllm.VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE,
                    config=config,
                    ge_cache=False)
            return self.torchair_compiled_models[batch_size]

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        kv_caches: Dict[str, torch.Tensor] = {}

        def align_memory(tensor: torch.Tensor, alignment: int) -> torch.Tensor:
            data_ptr = tensor.data_ptr()
            aligned_addr = (data_ptr + alignment - 1) // alignment * alignment
            offset = (aligned_addr - data_ptr) // tensor.element_size()
            return tensor[int(offset):]

        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.model_config.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=True,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=[self.cache_config.block_size],
        )

        kv_cache_sizes = {}
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            assert len(kv_cache_tensor.shared_by) == 1, (
                "KV cache tensor shared by multiple layers is not supported in "
                "NPU.")
            kv_cache_sizes[kv_cache_tensor.shared_by[0]] = kv_cache_tensor.size

        for kv_cache_group in kv_cache_config.kv_cache_groups:
            kv_cache_spec = kv_cache_group.kv_cache_spec
            for layer_name in kv_cache_group.layer_names:
                tensor_size = kv_cache_sizes[layer_name]
                assert tensor_size % kv_cache_spec.page_size_bytes == 0
                num_blocks = tensor_size // kv_cache_spec.page_size_bytes

                # `num_blocks` is the number of blocks the model runner can use.
                # `kv_cache_config.num_blocks` is the number of blocks that
                # KVCacheManager may allocate.
                # Since different GPUs may have different number of layers and
                # different memory capacities, `num_blocks` can be different on
                # different GPUs, and `kv_cache_config.num_blocks` is set to
                # the min of all `num_blocks`. Verify it here.
                assert num_blocks >= kv_cache_config.num_blocks
                alignment = 2 * 1024 * 1024
                # TODO: remove this after the OOM issue is located and fixed, otherwise, some model may
                # encounter OOM issue
                if isinstance(kv_cache_spec, FullAttentionSpec):
                    kv_cache_shape = self.attn_backend.get_kv_cache_shape(
                        num_blocks, kv_cache_spec.block_size,
                        kv_cache_spec.num_kv_heads, kv_cache_spec.head_size)
                    dtype = kv_cache_spec.dtype
                    if self.model_config.is_deepseek_mla:
                        # In order to transfer kv cache through the reigster_memory api from llmdatadist, the memory
                        # address should be aligned by 2M. In most case, torch_npu can allocate 2M aligned memory, but
                        # we found there are also some exceptions during test, so we manual align those memory here, this part
                        # of code may consume 2M * 2 * elem_size memory every layer.
                        num_blocks, block_size, num_kv_heads, head_size = kv_cache_shape
                        rope_dim = self.model_config.hf_text_config.qk_rope_head_dim
                        nope_dim = head_size - rope_dim
                        nope_allocate_shape = num_blocks * block_size * num_kv_heads * nope_dim
                        nope_allocate_shape_alignment = nope_allocate_shape + alignment
                        rope_allocate_shape = num_blocks * block_size * num_kv_heads * rope_dim
                        rope_allocate_shape_alignment = rope_allocate_shape + alignment
                        nope_cache_shape = (num_blocks, block_size,
                                            num_kv_heads, nope_dim)
                        rope_cache_shape = (num_blocks, block_size,
                                            num_kv_heads, rope_dim)
                        nope_cache = torch.zeros(nope_allocate_shape_alignment,
                                                 dtype=dtype,
                                                 device=self.device)
                        rope_cache = torch.zeros(rope_allocate_shape_alignment,
                                                 dtype=dtype,
                                                 device=self.device)
                        nope_cache = align_memory(
                            nope_cache, alignment)[:nope_allocate_shape].view(
                                nope_cache_shape)
                        rope_cache = align_memory(
                            rope_cache, alignment)[:rope_allocate_shape].view(
                                rope_cache_shape)
                        kv_caches[layer_name] = (nope_cache, rope_cache)
                    else:
                        # Round up a multiple of 64 to reduce repeated compilation caused by
                        # num_blocks fluctuation caused by memory differences.
                        num_blocks = (num_blocks + 63) // 64 * 64
                        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
                            num_blocks, kv_cache_spec.block_size,
                            kv_cache_spec.num_kv_heads,
                            kv_cache_spec.head_size)
                        k_v_cache_shape = kv_cache_shape[1:]
                        layer_k_cache = torch.zeros(k_v_cache_shape,
                                                    dtype=self.dtype,
                                                    pin_memory=True,
                                                    device=self.device)
                        layer_v_cache = torch.zeros(k_v_cache_shape,
                                                    dtype=self.dtype,
                                                    pin_memory=True,
                                                    device=self.device)
                        kv_caches[layer_name] = (layer_k_cache, layer_v_cache)
                else:
                    # TODO: add new branches when introducing more types of
                    # KV cache specs.
                    raise ValueError("Unknown KV cache spec type.")

        bind_kv_cache(
            kv_caches,
            self.vllm_config.compilation_config.static_forward_context,
            self.kv_caches)

        if has_kv_transfer_group():
            get_kv_transfer_group().register_kv_caches(kv_caches)

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """

        forward_ctx = self.vllm_config.compilation_config.static_forward_context
        block_size = self.vllm_config.cache_config.block_size
        use_mla = self.vllm_config.model_config.use_mla
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        for layer_name, attn_module in forward_ctx.items():
            if isinstance(attn_module, FusedMoE):
                continue

            # TODO: Support other attention modules, e.g., sliding window,
            # cross-attention
            assert isinstance(attn_module, Attention)
            if attn_module.attn_type == AttentionType.DECODER:
                kv_cache_spec[layer_name] = FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=attn_module.num_kv_heads,
                    head_size=attn_module.head_size,
                    dtype=attn_module.dtype,
                    use_mla=use_mla)
            elif attn_module.attn_type in (AttentionType.ENCODER,
                                           AttentionType.ENCODER_ONLY):
                # encoder-only attention does not need KV cache.
                continue
            elif attn_module.attn_type == AttentionType.ENCODER_DECODER:
                raise NotImplementedError
            else:
                raise ValueError(
                    f"Unknown attention type: {attn_module.attn_type}")

        return kv_cache_spec

    def capture_model(self) -> None:
        start_time = time.perf_counter()
        start_free_npu_memory = torch.npu.mem_get_info()[0]
        # TODO(NeverRaR): Calling graph_capture(device=self.device) in
        # torchair graph capture can cause some issues, so now we just
        # temporarily split the codepath for the two different graph patterns.
        if self.torchair_graph_enabled:
            torchair_graph_batch_sizes = self.torchair_graph_batch_sizes
            graph_num = len(torchair_graph_batch_sizes)
            logger.info(
                "Capturing torchair graph, this usually takes %.1f~%.1f mins.",
                0.5 * graph_num, 1.5 * graph_num)
            # Trigger torchair graph capture for specific shapes.
            # Capture the large shapes first so that the smaller shapes
            # can reuse the memory pool allocated for the large shapes.
            for idx, num_tokens in enumerate(
                    reversed(torchair_graph_batch_sizes)):
                for _ in range(self.vllm_config.compilation_config.
                               cudagraph_num_of_warmups):
                    # NOTE: when in torchair graph and not with_prefill,
                    # we don't need to set `skip_attn=False`
                    self._dummy_run(num_tokens, is_torchair_compile=True)
                self._dummy_run(num_tokens, is_torchair_compile=True)
                logger.info("Batchsize %d is compiled successfully: %d/%d.",
                            num_tokens, idx + 1, graph_num)
        elif self.use_aclgraph:
            # Trigger ACL graph capture for specific shapes.
            # Capture the large shapes first so that the smaller shapes
            # can reuse the memory pool allocated for the large shapes.
            # TODO(zzzzwwjj): Check dummy_run with ACL Graph and full graph mode
            with graph_capture(device=self.device):
                skip_attn = not self.vllm_config.compilation_config.full_cuda_graph
                # TODO: Make sure passing attn_state to _dummy_run in the future
                for num_tokens in reversed(self.aclgraph_batch_sizes):
                    for _ in range(self.vllm_config.compilation_config.
                                   cudagraph_num_of_warmups):
                        self._dummy_run(num_tokens, skip_attn=skip_attn)
                    self._dummy_run(num_tokens, skip_attn=skip_attn)
        else:
            logger.info("Skipping NPU graph capture for eager mode.")
            return
        end_time = time.perf_counter()
        end_free_npu_memory = torch.npu.mem_get_info()[0]
        elapsed_time = end_time - start_time
        npu_graph_size = start_free_npu_memory - end_free_npu_memory
        # This usually takes 5~20 seconds.
        logger.info("Graph capturing finished in %.0f secs, took %.2f GiB",
                    elapsed_time, npu_graph_size / (1 << 30))

    def _generate_draft_token_ids(
        self,
        sampled_token_ids: list[list[int]],
        sampling_metadata: SamplingMetadata,
    ) -> list[list[int]]:
        # TODO(woosuk): Optimize.
        draft_token_ids: list[list[int]] = []
        for i, sampled_ids in enumerate(sampled_token_ids):
            num_sampled_ids = len(sampled_ids)
            if not num_sampled_ids:
                # Skip speculative decoding.
                draft_token_ids.append([])
                continue

            # Skip requests that require top-p, top-k, etc.
            req_id = self.input_batch.req_ids[i]
            if not is_spec_decode_supported(req_id, self.input_batch):
                draft_token_ids.append([])
                continue

            # Add sampled_token_ids to token_ids_cpu.
            start_idx = self.input_batch.num_tokens_no_spec[i]
            end_idx = start_idx + num_sampled_ids
            self.input_batch.token_ids_cpu[i, start_idx:end_idx] = sampled_ids
            drafter_output = self.drafter.propose(
                self.input_batch.token_ids_cpu[i, :end_idx])
            if drafter_output is None or len(drafter_output) == 0:
                draft_token_ids.append([])
            else:
                draft_token_ids.append(drafter_output.tolist())
        return draft_token_ids

    def _generate_mtp_token_ids(
        self,
        valid_sampled_token_ids: list[list[int]],
        sampling_metadata: SamplingMetadata,
        scheduler_output: "SchedulerOutput",
        spec_decode_metadata: SpecDecodeMetadata,
        positions: torch.Tensor,
        num_scheduled_tokens: int,
        hidden_states: torch.Tensor,
        attn_metadata: SpecDecodeMetadata,
    ):
        next_token_ids: list[int] = []
        for i, token_ids in enumerate(valid_sampled_token_ids):
            if token_ids:
                # Common case.
                next_token_id = token_ids[-1]
            else:
                # Partial prefill (rare case).
                # Get the next token id from the request state.
                req_id = self.input_batch.req_ids[i]
                req_state = self.requests[req_id]
                seq_len = (req_state.num_computed_tokens +
                           scheduler_output.num_scheduled_tokens[req_id])
                next_token_id = req_state.get_token_id(seq_len)
            next_token_ids.append(next_token_id)
        next_token_ids = torch.tensor(next_token_ids,
                                      dtype=torch.int32,
                                      device=self.device)
        token_indices = None
        if spec_decode_metadata is None:
            # input_ids can be None for multimodal models.
            target_token_ids = self.input_ids[:num_scheduled_tokens]
            target_positions = positions[:num_scheduled_tokens]
            target_hidden_states = hidden_states[:num_scheduled_tokens]
            target_slot_mapping = attn_metadata.slot_mapping
            cu_num_tokens = attn_metadata.query_start_loc
        else:
            # TODO(woosuk): Refactor this.
            num_draft_tokens = spec_decode_metadata.num_draft_tokens
            num_rejected_tokens = [
                n + 1 - len(valid_sampled_token_ids[i]) if n > 0 else 0
                for i, n in enumerate(num_draft_tokens)
            ]
            num_rejected_tokens = torch.tensor(
                num_rejected_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            cu_num_tokens, token_indices = self.drafter.prepare_inputs(
                attn_metadata.query_start_loc,
                num_rejected_tokens,
                force_one_token=False,
                is_torchair_graph=self.torchair_graph_enabled)
            if self.torchair_graph_enabled:
                # the seq len of each bath is padded to 2, thus input is same as the main model
                target_token_ids = self.input_ids[:num_scheduled_tokens]
                target_positions = positions[:num_scheduled_tokens]
                target_hidden_states = hidden_states[:num_scheduled_tokens]
                target_slot_mapping = attn_metadata.slot_mapping[:
                                                                 num_scheduled_tokens]
            else:
                target_token_ids = self.input_ids[token_indices]
                target_positions = positions[token_indices]
                target_hidden_states = hidden_states[token_indices]
                target_slot_mapping = attn_metadata.slot_mapping[token_indices]

        draft_token_ids = self.drafter.propose(
            target_token_ids=target_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            target_slot_mapping=target_slot_mapping,
            next_token_ids=next_token_ids,
            cu_num_tokens=cu_num_tokens,
            block_table=attn_metadata.block_tables,
            sampling_metadata=sampling_metadata,
            token_indices=token_indices)
        spec_token_ids = draft_token_ids.tolist()
        return spec_token_ids

    def init_torchair_graph_batch_sizes(self):
        start_graph_batch_size = 4
        tp_size = get_tensor_model_parallel_world_size()

        # NOTE: When use all2all | mc2, We need to slice the `num_tokens` dimension into `tp_size` blocks
        start_graph_batch_size = max(start_graph_batch_size, tp_size)

        while (start_graph_batch_size <= self.scheduler_config.max_num_seqs):
            self.torchair_graph_batch_sizes.append(start_graph_batch_size)
            start_graph_batch_size *= 2

    def select_torchair_padded_batch_size(self, batch_size: int):
        for padded_batch_size in self.torchair_graph_batch_sizes:
            if batch_size <= padded_batch_size:
                # we treat batch_size as num of requests
                return padded_batch_size
        raise ValueError(
            f"cur batch_size is invalid, torchair_graph_batch_sizes is "
            f"{self.torchair_graph_batch_sizes}, but cur batch_size is {batch_size}."
        )

    def check_torchair_graph_batch_sizes(self):
        # return graph_batch_sizes according to the number of tokens
        # first pad according to the number of requests
        if len(self.torchair_graph_batch_sizes) == 0:
            self.torchair_graph_batch_sizes = [1, self.max_num_reqs]
        else:
            self.torchair_graph_batch_sizes = sorted(
                self.torchair_graph_batch_sizes)
            while self.torchair_graph_batch_sizes[-1] > self.max_num_reqs:
                self.torchair_graph_batch_sizes.pop()
                if len(self.torchair_graph_batch_sizes) == 0:
                    logger.warning(
                        "torch_graph_batch_sizes is invalid, reset it to [1, max_num_seqs]"
                    )
                    self.torchair_graph_batch_sizes = [1, self.max_num_reqs]
            if self.torchair_graph_batch_sizes[-1] < self.max_num_reqs:
                self.torchair_graph_batch_sizes.append(self.max_num_reqs)

        # we need to make sure that we can deal with max_num_req when `self.decode_token_per_req` is not 1
        self.torchair_graph_batch_sizes = [
            graph_batch_size * self.decode_token_per_req
            for graph_batch_size in self.torchair_graph_batch_sizes
        ]

        # NOTE: when enable_expert_parallel, we need to check if `graph_batch_size` is divisible by `tp_size`
        tp_size = self.parallel_config.tensor_parallel_size
        if self.parallel_config.enable_expert_parallel:
            new_graph_batch_sizes = []
            for graph_batch_size in self.torchair_graph_batch_sizes:
                cur_graph_batch_size = (graph_batch_size + tp_size -
                                        1) // tp_size * tp_size
                if cur_graph_batch_size not in new_graph_batch_sizes and \
                    cur_graph_batch_size <= self.scheduler_config.max_num_batched_tokens:
                    new_graph_batch_sizes.append(cur_graph_batch_size)
                elif cur_graph_batch_size > self.scheduler_config.max_num_batched_tokens \
                        and self.decode_token_per_req > 1:
                    logger.warning(
                        f"torchair_graph_batch_sizes {cur_graph_batch_size} is bigger than max_num_batched_tokens",
                        f"{self.scheduler_config.max_num_batched_tokens} will skip this batch size."
                    )
            self.torchair_graph_batch_sizes = new_graph_batch_sizes
