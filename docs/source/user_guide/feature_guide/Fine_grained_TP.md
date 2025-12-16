# Fine-Grained Tensor Parallelism (Finegrained TP)

## Overview

Fine-Grained Tensor Parallelism (Finegrained TP) extends standard tensor parallelism by enabling **independent tensor parallel sizes for different model components**. Instead of applying a single global `tensor_parallel_size` to all layers, Finegrained TP allows users to configure separate TP degrees for key modules—such as embedding, language model head (lm_head), attention output projection (oproj), and MLP blocks—via the `finegrained_tp_config` parameter.

This capability supports heterogeneous parallelism strategies within a single model, providing finer control over weight distribution, memory layout, and communication patterns across devices. The feature is compatible with standard dense transformer architectures and integrates seamlessly into vLLM’s serving pipeline.

## Benefits of Finegrained TP

Fine-Grained Tensor Parallelism delivers two primary performance advantages through targeted weight sharding:
- **Reduced Per-Device Memory Footprint**:  
    Finegrained TP shards large weight matrices (e.g., LM Head, o_proj) across devices, lowering peak memory usage and enabling larger batches or deployment on memory-limited hardware—without quantization.
- **Faster Memory Access in GEMMs**:  
    In decode-heavy workloads, GEMM performance is often memory-bound. Weight sharding reduces per-device weight fetch volume, cutting DRAM traffic and improving bandwidth efficiency—especially for latency-sensitive layers like LM Head and o_proj.
Together, these effects allow practitioners to better balance memory, communication, and compute—particularly in high-concurrency serving scenarios—while maintaining compatibility with standard dense transformer models.

## Supported Scenarios

### Models  
Finegrained TP is **model-agnostic** and supports all standard dense transformer architectures, including Llama, Qwen, DeepSeek (base/dense variants), and others.

### Component & Execution Mode Support  
- **`embedding`, `lm_head`, and `mlp`**: Can be configured with fine-grained TP in any execution context—prefill, decode, or mixed deployment.
- **`o_proj`**: Currently, fine-grained TP for the attention output projection is **only supported in graph-capture mode** (e.g., CUDA Graphs). It cannot be enabled in eager execution.

## How to Use Finegrained TP

### Configuration Format:

Finegrained TP is controlled via the `finegrained_tp_config` field inside `--additional-config`.

```bash
--additional-config '{
    "finegrained_tp_config": {
        "embedding_tensor_parallel_size": 8,
        "lmhead_tensor_parallel_size": 8,
        "oproj_tensor_parallel_size": 8,
        "mlp_tensor_parallel_size": 8
}
```

### Example Usage:

```bash
vllm serve deepseek-ai/DeepSeek-R1 \
    --data-parallel-size 16 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --additional-config '{
        "finegrained_tp_config": {
            "embedding_tensor_parallel_size": 8,
            "lmhead_tensor_parallel_size": 8,
            "oproj_tensor_parallel_size": 8,
            "mlp_tensor_parallel_size": 8
        }
    }' \
    --kv-transfer-config \
    '{
        "kv_connector": "MooncakeConnector",
        "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector",
        "kv_buffer_device": "npu",
        "kv_role": "kv_consumer",
        "kv_parallel_size": 2,
        "kv_port": "20002",
        "engine_id": "decode-'${NODE_RANK}'",
        "kv_rank": 1,
        "kv_connector_extra_config": {
            "use_ascend_direct": true,
            "prefill": {
                    "dp_size": '${PREFILL_DP_SIZE}',
                    "tp_size": '${PREFILL_TP_SIZE}',
                    "pp_size": '${PREFILL_PP_SIZE}'
             },
             "decode": {
                    "dp_size": '${DECODE_DP_SIZE}',
                    "tp_size": '${DECODE_TP_SIZE}'
             }
        }
    }'
```

## Deployment Recommendations  
- Finegrained TP is **most effective in Decode-dominant workloads**, where models are typically deployed in large-data-parallel (all-DP) configurations. In this setting, sharding weight-heavy components reduces redundant storage and memory pressure.
- The TP size for any component must:
  - Be **≤ the data-parallel (DP) degree**, and  
  - **Evenly divide the DP degree** (i.e., `dp_size % tp_size == 0`) to ensure valid device assignment and communication grouping.

> ⚠️ Violating these constraints will result in runtime errors or undefined behavior.
