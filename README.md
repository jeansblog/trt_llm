Install and use TensorRT-LLM for GPT-OSS 120B with the OpenAI-compatible API on the DGX Spark Sparks.

 ```
 Downloading tiktoken encoding files...
--2025-12-20 04:19:30--  https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken
Resolving openaipublic.blob.core.windows.net (openaipublic.blob.core.windows.net)... 20.60.241.33
Connecting to openaipublic.blob.core.windows.net (openaipublic.blob.core.windows.net)|20.60.241.33|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 3613922 (3.4M) [application/octet-stream]
Saving to: ‘/tmp/harmony-reqs/o200k_base.tiktoken’

     0K .......... .......... .......... .......... ..........  1%  162K 21s
    50K .......... .......... .......... .......... ..........  2%  325K 16s

...

2025-12-20 04:19:41 (380 KB/s) - ‘/tmp/harmony-reqs/o200k_base.tiktoken’ saved [3613922/3613922]

--2025-12-20 04:19:41--  https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken
Resolving openaipublic.blob.core.windows.net (openaipublic.blob.core.windows.net)... 20.60.241.33
Connecting to openaipublic.blob.core.windows.net (openaipublic.blob.core.windows.net)|20.60.241.33|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 1681126 (1.6M) [application/octet-stream]
Saving to: ‘/tmp/harmony-reqs/cl100k_base.tiktoken’

     0K .......... .......... .......... .......... ..........  3%  160K 10s
    50K .......... .......... .......... .......... ..........  6%  318K 7s

2025-12-20 04:19:51 (169 KB/s) - ‘/tmp/harmony-reqs/cl100k_base.tiktoken’ saved [1681126/1681126]

Downloading Hugging Face model: openai/gpt-oss-120b...
Fetching 37 files: 100%|██████████| 37/37 [00:00<00:00, 41875.13it/s]
/root/.cache/huggingface/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a
Creating /tmp/extra-llm-api-config.yml...
Starting trtllm-serve for model openai/gpt-oss-120b on port 8355...
[2025-12-20 04:19:59] INFO config.py:54: PyTorch version 2.8.0a0+5228986c39.nv25.6 available.
[2025-12-20 04:19:59] INFO config.py:66: Polars version 1.25.2 available.
/usr/local/lib/python3.12/dist-packages/modelopt/torch/__init__.py:36: UserWarning: transformers version 4.55.0 is incompatible with nvidia-modelopt and may cause issues. Please install recommended version with `pip install nvidia-modelopt[hf]` if working with HF models.
  _warnings.warn(
2025-12-20 04:20:03,864 - INFO - flashinfer.jit: Prebuilt kernels not found, using JIT backend
[TensorRT-LLM] TensorRT-LLM version: 1.1.0rc3
/usr/local/lib/python3.12/dist-packages/pydantic/_internal/_fields.py:198: UserWarning: Field name "schema" in "ResponseFormat" shadows an attribute in parent "OpenAIBaseModel"
  warnings.warn(
[12/20/2025-04:20:04] [TRT-LLM] [I] Using LLM with PyTorch backend
[12/20/2025-04:20:04] [TRT-LLM] [I] Set nccl_plugin to None.
[12/20/2025-04:20:04] [TRT-LLM] [I] neither checkpoint_format nor checkpoint_loader were provided, checkpoint_format will be set to HF.
rank 0 using MpiPoolSession to spawn MPI processes
/root/.cache/huggingface/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a
[12/20/2025-04:20:06] [TRT-LLM] [I] Generating a new HMAC key for server proxy_request_queue
[12/20/2025-04:20:06] [TRT-LLM] [I] Generating a new HMAC key for server worker_init_status_queue
[12/20/2025-04:20:06] [TRT-LLM] [I] Generating a new HMAC key for server proxy_result_queue
[12/20/2025-04:20:06] [TRT-LLM] [I] Generating a new HMAC key for server proxy_stats_queue
[12/20/2025-04:20:06] [TRT-LLM] [I] Generating a new HMAC key for server proxy_kv_cache_events_queue
[2025-12-20 04:20:13] INFO config.py:54: PyTorch version 2.8.0a0+5228986c39.nv25.6 available.
[2025-12-20 04:20:13] INFO config.py:66: Polars version 1.25.2 available.
Multiple distributions found for package optimum. Picked distribution: optimum
/usr/local/lib/python3.12/dist-packages/modelopt/torch/__init__.py:36: UserWarning: transformers version 4.55.0 is incompatible with nvidia-modelopt and may cause issues. Please install recommended version with `pip install nvidia-modelopt[hf]` if working with HF models.
  _warnings.warn(
2025-12-20 04:20:16,995 - INFO - flashinfer.jit: Prebuilt kernels not found, using JIT backend
[TensorRT-LLM] TensorRT-LLM version: 1.1.0rc3
/usr/local/lib/python3.12/dist-packages/pydantic/_internal/_fields.py:198: UserWarning: Field name "schema" in "ResponseFormat" shadows an attribute in parent "OpenAIBaseModel"
  warnings.warn(
[TensorRT-LLM][INFO] Refreshed the MPI local session
[12/20/2025-04:20:19] [TRT-LLM] [I] PyTorchConfig(extra_resource_managers={}, use_cuda_graph=True, cuda_graph_batch_sizes=[1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128], cuda_graph_max_batch_size=128, cuda_graph_padding_enabled=True, disable_overlap_scheduler=True, moe_max_num_tokens=None, moe_load_balancer=None, attention_dp_enable_balance=False, attention_dp_time_out_iters=50, attention_dp_batching_wait_iters=10, batch_wait_timeout_ms=0, attn_backend='TRTLLM', moe_backend='CUTLASS', moe_disable_finalize_fusion=False, enable_mixed_sampler=False, sampler_type=<SamplerType.auto: 'auto'>, kv_cache_dtype='auto', mamba_ssm_cache_dtype='auto', enable_iter_perf_stats=False, enable_iter_req_stats=False, print_iter_log=False, torch_compile_enabled=False, torch_compile_fullgraph=True, torch_compile_inductor_enabled=False, torch_compile_piecewise_cuda_graph=False, torch_compile_piecewise_cuda_graph_num_tokens=None, torch_compile_enable_userbuffers=True, torch_compile_max_num_streams=1, enable_autotuner=True, enable_layerwise_nvtx_marker=False, load_format=<LoadFormat.AUTO: 0>, enable_min_latency=False, allreduce_strategy='AUTO', stream_interval=1, force_dynamic_quantization=False, mm_encoder_only=False, _limit_torch_cuda_mem_fraction=True)
[12/20/2025-04:20:19] [TRT-LLM] [I] ATTENTION RUNTIME FEATURES:  AttentionRuntimeFeatures(chunked_prefill=False, cache_reuse=True, has_speculative_draft_tokens=False, chunk_size=8192)
/root/.cache/huggingface/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a
[12/20/2025-04:20:19] [TRT-LLM] [I] Validating KV Cache config against kv_cache_dtype="auto"
[12/20/2025-04:20:19] [TRT-LLM] [I] KV cache quantization set to "auto". Using checkpoint KV quantization.
[12/20/2025-04:20:21] [TRT-LLM] [I] Use 63.32 GB for model weights.
Loading safetensors weights in parallel: 100%|██████████| 15/15 [00:00<00:00, 81.92it/s]
Loading weights: 100%|██████████| 801/801 [00:46<00:00, 17.29it/s]
Model init total -- 49.14s
[12/20/2025-04:21:09] [TRT-LLM] [I] max_seq_len is not specified, using inferred value 131072
[12/20/2025-04:21:09] [TRT-LLM] [I] Using Sampler: TorchSampler
[12/20/2025-04:21:10] [TRT-LLM] [W] Both free_gpu_memory_fraction and max_tokens are set (to 0.8999999761581421 and 8224, respectively). The smaller value will be used.
[12/20/2025-04:21:10] [TRT-LLM] [W] Attention window size 131072 exceeds upper bound 8224 for available blocks. Reducing to 8224.
[12/20/2025-04:21:10] [TRT-LLM] [W] Adjusted max_attention_window_vec to [8224]
[12/20/2025-04:21:10] [TRT-LLM] [W] Adjusted window size 131072 to 8224 in blocks_per_window
[12/20/2025-04:21:10] [TRT-LLM] [W] Adjusted max_seq_len to 8224
[TensorRT-LLM][INFO] Number of tokens per block: 32.
[TensorRT-LLM][INFO] [MemUsageChange] Allocated 0.56 GiB for max tokens in paged KV cache (8224).
[12/20/2025-04:21:10] [TRT-LLM] [I] max_seq_len=8224, max_num_requests=64, max_num_tokens=8192, max_batch_size=64
[12/20/2025-04:21:10] [TRT-LLM] [I] cache_transceiver is disabled
[12/20/2025-04:21:10] [TRT-LLM] [I] [Autotuner] Autotuning process starts ...
2025-12-20 04:21:17,836 - INFO - flashinfer.jit: Loading JIT ops: norm
2025-12-20 04:21:30,232 - INFO - flashinfer.jit: Finished loading JIT ops: norm
[TensorRT-LLM][WARNING] Attention workspace size is not enough, increase the size from 0 bytes to 100738816 bytes
[12/20/2025-04:21:37] [TRT-LLM] [I] [Autotuner] Cache size after warmup is 28
[12/20/2025-04:21:37] [TRT-LLM] [I] [Autotuner] Autotuning process ends
[12/20/2025-04:21:37] [TRT-LLM] [I] Creating CUDA graph instances for 11 batch sizes.
[12/20/2025-04:21:37] [TRT-LLM] [I] Run generation only CUDA graph warmup for batch size=64, draft_len=0
[12/20/2025-04:21:47] [TRT-LLM] [I] Run generation only CUDA graph warmup for batch size=56, draft_len=0
[12/20/2025-04:21:47] [TRT-LLM] [I] Run generation only CUDA graph warmup for batch size=48, draft_len=0
[12/20/2025-04:21:47] [TRT-LLM] [I] Run generation only CUDA graph warmup for batch size=40, draft_len=0
[12/20/2025-04:21:47] [TRT-LLM] [I] Run generation only CUDA graph warmup for batch size=32, draft_len=0
[12/20/2025-04:21:48] [TRT-LLM] [I] Run generation only CUDA graph warmup for batch size=24, draft_len=0
[12/20/2025-04:21:48] [TRT-LLM] [I] Run generation only CUDA graph warmup for batch size=16, draft_len=0
[12/20/2025-04:21:48] [TRT-LLM] [I] Run generation only CUDA graph warmup for batch size=8, draft_len=0
[12/20/2025-04:21:49] [TRT-LLM] [I] Run generation only CUDA graph warmup for batch size=4, draft_len=0
[12/20/2025-04:21:49] [TRT-LLM] [I] Run generation only CUDA graph warmup for batch size=2, draft_len=0
[12/20/2025-04:21:49] [TRT-LLM] [I] Run generation only CUDA graph warmup for batch size=1, draft_len=0
[12/20/2025-04:21:50] [TRT-LLM] [I] Memory used after loading model weights (inside torch) in memory usage profiling: 63.52 GiB
[12/20/2025-04:21:50] [TRT-LLM] [I] Memory used after loading model weights (outside torch) in memory usage profiling: 49.78 GiB
[12/20/2025-04:21:51] [TRT-LLM] [I] Memory dynamically allocated during inference (inside torch) in memory usage profiling: 0.85 GiB
[12/20/2025-04:21:51] [TRT-LLM] [I] Memory used outside torch (e.g., NCCL and CUDA graphs) in memory usage profiling: 49.28 GiB
[12/20/2025-04:21:51] [TRT-LLM] [I] Peak memory during memory usage profiling (torch + non-torch): 113.65 GiB, available KV cache memory when calculating max tokens: 5.95 GiB, fraction is set 0.8999999761581421, kv size is 73728. device total memory 119.70 GiB, , tmp kv_mem 0.56 GiB
[12/20/2025-04:21:51] [TRT-LLM] [I] Estimated max memory in KV cache : 5.95 GiB
[12/20/2025-04:21:51] [TRT-LLM] [W] Both free_gpu_memory_fraction and max_tokens are set (to 0.8999999761581421 and 86711, respectively). The smaller value will be used.
[12/20/2025-04:21:51] [TRT-LLM] [W] Attention window size 131072 exceeds upper bound 86720 for available blocks. Reducing to 86720.
[12/20/2025-04:21:51] [TRT-LLM] [W] Adjusted max_attention_window_vec to [86720]
[12/20/2025-04:21:51] [TRT-LLM] [W] Adjusted window size 131072 to 86720 in blocks_per_window
[12/20/2025-04:21:51] [TRT-LLM] [W] Adjusted max_seq_len to 86720
[TensorRT-LLM][INFO] Number of tokens per block: 32.
[TensorRT-LLM][INFO] [MemUsageChange] Allocated 5.95 GiB for max tokens in paged KV cache (86720).
[12/20/2025-04:21:52] [TRT-LLM] [I] max_seq_len=86720, max_num_requests=64, max_num_tokens=8192, max_batch_size=64
[12/20/2025-04:21:52] [TRT-LLM] [I] cache_transceiver is disabled
[12/20/2025-04:21:52] [TRT-LLM] [I] [Autotuner] Autotuning process starts ...
[TensorRT-LLM][WARNING] Attention workspace size is not enough, increase the size from 0 bytes to 100738816 bytes
[12/20/2025-04:21:53] [TRT-LLM] [I] [Autotuner] Cache size after warmup is 28
[12/20/2025-04:21:53] [TRT-LLM] [I] [Autotuner] Autotuning process ends
[12/20/2025-04:21:53] [TRT-LLM] [I] Creating CUDA graph instances for 11 batch sizes.
[12/20/2025-04:21:53] [TRT-LLM] [I] Run generation only CUDA graph warmup for batch size=64, draft_len=0
[12/20/2025-04:21:53] [TRT-LLM] [I] Run generation only CUDA graph warmup for batch size=56, draft_len=0
[12/20/2025-04:21:54] [TRT-LLM] [I] Run generation only CUDA graph warmup for batch size=48, draft_len=0
[12/20/2025-04:21:54] [TRT-LLM] [I] Run generation only CUDA graph warmup for batch size=40, draft_len=0
[12/20/2025-04:21:54] [TRT-LLM] [I] Run generation only CUDA graph warmup for batch size=32, draft_len=0
[12/20/2025-04:21:55] [TRT-LLM] [I] Run generation only CUDA graph warmup for batch size=24, draft_len=0
[12/20/2025-04:21:55] [TRT-LLM] [I] Run generation only CUDA graph warmup for batch size=16, draft_len=0
[12/20/2025-04:21:55] [TRT-LLM] [I] Run generation only CUDA graph warmup for batch size=8, draft_len=0
[12/20/2025-04:21:56] [TRT-LLM] [I] Run generation only CUDA graph warmup for batch size=4, draft_len=0
[12/20/2025-04:21:56] [TRT-LLM] [I] Run generation only CUDA graph warmup for batch size=2, draft_len=0
[12/20/2025-04:21:56] [TRT-LLM] [I] Run generation only CUDA graph warmup for batch size=1, draft_len=0
[12/20/2025-04:21:57] [TRT-LLM] [I] Setting PyTorch memory fraction to 0.5983721838660735 (71.62367630004883 GiB)
/root/.cache/huggingface/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a
/root/.cache/huggingface/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a
INFO:     Started server process [41]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8355 (Press CTRL+C to quit)


 ```

![](nvtop.PNG)