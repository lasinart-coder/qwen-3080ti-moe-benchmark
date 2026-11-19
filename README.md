# Qwen3.6-35B-A3B Q4_K_M on an RTX 3080 Ti 12 GB: CPU-MoE vs two-machine RPC, plus a 128K-context startup test

**Experiment date:** 19 November 2026  
**Runtime:** llama.cpp `b12530` (RPC build reportedly at commit `b001d7b`)  
At 32K context, Qwen generated **58.8 tokens/s** on my RTX 3080 Ti with CPU expert offload. Moving 11 expert layers to a remote RTX 3050 over 2.5 GbE reduced generation to **34.0 tokens/s**, a **42.2% decrease**. In a separate test, the single-GPU server started with a 131,072-token window and handled short requests and tool calls. Full-window performance remains unmeasured.

## Hardware and software

| Item | Configuration |
|---|---|
| Primary machine | Windows 11 Pro; NVIDIA GeForce RTX 3080 Ti, 12 GiB VRAM; 32 GB host RAM |
| Secondary machine | Ubuntu Linux; NVIDIA GeForce RTX 3050, 6 GiB VRAM |
| Interconnect | Direct 2.5 Gb/s Ethernet, using localhost-bound RPC bridged through SSH |
| Inference | llama.cpp `b12530`, CUDA; experimental RPC only in the two-machine benchmark |
| Model | `Qwen3.6-35B-A3B`, Q4_K_M GGUF, approximately 19.13 GiB file |
| Initial server | Single inference slot, 32,768-token context, Flash Attention on, K/V cache `q8_0/q8_0`, 8 CPU threads |
| Original placement | `--n-gpu-layers 99 --n-cpu-moe 20` (selective CPU MoE placement) |

The two experiments below answer different questions. Keep their results separate: the RPC comparison is a same-context **32K vs 32K** benchmark; the 128K test used a different MoE split and KV precision.

## Experiment 1 — Does a remote RTX 3050 speed up inference?

The two-machine variant moved **11 MoE expert layers** (reported as approximately **4.64 GiB** of weights) to the remote RTX 3050. The attention/SSM and KV components remained on the RTX 3080 Ti. The reported allocation was about **10.82 GiB primary + 5.12 GiB remote**; combined GPU memory use was approximately **15.9 GiB**, compared with **11.7 GiB** on the original single-machine setup.

### 32K context results

| Metric | RTX 3080 Ti + CPU-MoE (baseline) | RTX 3080 Ti + RTX 3050 RPC | RPC change |
|---|---:|---:|---:|
| Generation | **58.8 tok/s** | 34.0 tok/s | **−42.2%** |
| Prompt processing, 2K tokens | **1,211 tok/s** | 641 tok/s | **−47.1%** |
| Prompt processing, 8K tokens | **1,343 tok/s** | 736 tok/s | **−45.2%** |
| Prompt processing, 28K tokens | **1,278 tok/s** | 685 tok/s | **−46.4%** |
| Total GPU memory in use | ~11.7 GiB | ~15.9 GiB across both | Higher |
| Strict tool-use tests | 26/27 | 26/27 | Same single failure |
| Mean measured tool-use latency | 1.65 s | 2.69 s | Slower |

Both configurations reportedly had zero invalid JSON responses and zero transport/inference errors in this tool-calling comparison. The shared failure was a minor argument-label formatting mismatch.

A separate 16K-context run reported generation of **50.4 vs 36.3 tok/s** (baseline vs RPC). Numbers for a nominal 28K-prompt test within that 16K group are not presented here because the effective test context/truncation method has not been verified.

**Interpretation (inference, not directly measured):** This MoE setup may benefit more from fast CPU expert offload than from slower remote layers executed sequentially, given the RTX 3050's memory bandwidth and RPC/SSH overhead. This result applies to the tested hardware, model placement and software build; it does not establish that all multi-GPU RPC configurations are slower.

**Decision:** Keep the original local RTX 3080 Ti + CPU-MoE setup. RPC and tunnel processes were shut down afterward, and the 3050 was freed for other workloads.

## Experiment 2 — Can the same 12 GiB GPU start a 128K-context server?

The inference server was subsequently restarted **without RPC**, with these settings:

| Setting | Reported value |
|---|---|
| Context | `131072` tokens (server `/props` confirmed) |
| MoE offload | `--n-cpu-moe 22` |
| KV cache | `q8_0` keys / `q4_0` values |
| Flash Attention | Enabled |
| Slots | 1 |
| Server start/model load | 12.6 s |

### Measured short-test results

| Metric | Reported observation |
|---|---:|
| Server GPU use | 11,817 MiB total GPU use; 11,783 MiB attributed to server |
| Free GPU memory under load | **270 MiB** |
| Available system RAM | **4.9 GB** |
| Windows committed memory | **54.52 / 55.79 GB** (about 1.27 GB headroom) |
| Generation | **50.47 tok/s**, 81 generated tokens |
| Very short prompt evaluation | **303.7 tok/s**, 284 tokens |
| HTTP health | OK |
| OpenAI-format tool call | Passed |
| Anthropic-format tool call | Passed |
| Remote access through pre-existing localhost tunnel | Passed |

### Follow-up interactive agent trial (not a throughput benchmark)

In a later Claude Code session, `/context` showed **1.2K / 131.1K tokens (0.9%)** at startup. I used `/compact` once during the multi-step tool-using session, which completed successfully. I did not record peak context usage before compaction or test full-window prefill and retrieval accuracy. The **50.47 tok/s** figure above comes from a separate 81-token output.

### FlashAttention warning

The CUDA build reported that it had no vector kernel compiled for the `q8_0`-K / `q4_0`-V pair and used an f16 conversion fallback instead. That fallback may impose extra costs, especially during long-context inference. Its effect has **not** been isolated by an A/B test in this report.

## Trade-offs and caveats

1. **RAM/commit margin is extremely tight** on the 128K setup despite reasonable physical available RAM. A different desktop workload or pagefile configuration may change stability.
2. **270 MiB VRAM headroom is small** and may be affected by display usage or other CUDA applications.
3. **128K is a verified allocation/startup result with an additional successful multi-tool interactive trial, not a full-context benchmark.** No 64K/128K prompt-processing benchmark, full-window retrieval test, or systematic long-session reliability study has been completed.
4. **The original 32K RPC comparison is more mature** than the later 128K feasibility test, but full raw commands, prompts, repetitions and uncertainty intervals were not included in the publicly shared measurements.
5. **Software version matters.** The RPC backend and CUDA FlashAttention quantization kernels can change between llama.cpp builds.
6. **Units:** Model weight size, Windows GPU memory reports and total split allocations come from different reporting contexts. Numbers are retained with their reported units; do not infer byte-perfect accounting from rounded GB/GiB figures.

## Reproducibility notes

The [benchmark CSV](benchmark_32k.csv) contains the 32K throughput and tool-use summary measurements. Raw per-run timings, repetition counts, full launch commands, and benchmark scripts are unavailable in this repository.

## Questions for the community

- On similar 12 GiB CUDA GPUs, what MoE expert placement gives the best throughput for this model family?
- Which llama.cpp build/configuration provides an efficient native FlashAttention path for `q8_0`-K / `q4_0`-V?
- Have you found a more stable KV configuration at 128K without substantially worsening retrieval accuracy or host memory pressure?
- Are there RPC placements/network configurations that beat CPU offload on modest heterogeneous consumer GPUs?
