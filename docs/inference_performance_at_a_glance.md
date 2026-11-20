# Which Mol-1 optimizations made inference faster?

Fast initialization and a four-thread CPU cap gave the largest **startup** savings. Atom-block caching cut model time by **3.5%** on an uncontended H200 and preserved all five predictions per input bitwise. First-sample ipSAE screening saved more time by generating fewer samples. Eager CUDA replay preserved the tested outputs, while its throughput benefit remains unproven.

## Startup and five-sample inference

| Change | Measured result | When the result applies |
| --- | --- | --- |
| Fast initialization | CPU model construction fell from **55–71 s to 2.4 s**. The strict checkpoint load still ran. | Once per invocation. A process handling 1,000 designs pays that construction cost once. [Startup benchmark](inference_optimization_summary.md#startup-and-model-reuse) |
| Four CPU threads per process | With fast initialization already enabled, construction fell from **50–62 s to 5.0–5.4 s** per process in the supplied seven-process test. | Concurrent startup on a host using 128 PyTorch threads per process by default. [Concurrent construction results](inference_optimization_summary.md#concurrent-startup-and-replay-evidence) |
| Atom-block cache | **63.36 → 61.13 s** for 9fm7 and 8cnh together. Both configurations produced ten structures and peaked at **44.46 GiB**. | Warmed, alternating cache-on/off runs. Coordinates, PAE, pLDDT, ipSAE and rankings matched bitwise. [Cache comparison](inference_optimization_summary.md#how-fast-it-is-now) |
| Grouping by actual token length | Four different-length examples took **134 s** in singleton groups, **167 s** in padded pairs and **238 s** in one batch of four. | The same batched scheduler handled all groups. Avoiding padding saved time on this queue. Changing trunk padding can also change a prediction. [Grouping comparison](inference_optimization_summary.md#how-fast-it-is-now) |
| Diffusion sample chunk size | On 9fm7, generating all five samples together took **43 s**, versus **62 s** one at a time. | Both runs produced five samples. Chunk size can change the order of random draws. First-sample screening requires diffusion and confidence chunks of one. [Chunk measurements](inference_memory.md#what-the-controls-are) |
| Fused triangle kernels and confidence chunking | On a synthetic 512/640-token pair, the combined optimized configuration took **62.33 s**, versus **128.30 s** on the original path. | Both used diffusion chunks of five. The **2.06× throughput** figure covers several changes together and depends on input size. [Kernel and chunk comparison](inference_batching.md#reproducible-throughput-measurements) |
| FP32 attention layout | **68.8 → 66.7 s** for 9fm7 and 8cnh together, **35.4 s** either way for 9fm7 alone. | About a 3% gain for the tested pair. [Attention measurements](inference_optimization_summary.md#earlier-gains-and-changes-that-did-not-pay-off) |

## Antibody-antigen ipSAE and binder ranking

The November 20, 2026 comparison used `mol-1-base.pt`, seed 101 and five predictions per antibody–antigen pair. Each test changed one setting. Pairs were ranked by their median ipSAE across the five predictions. [Run settings](../../Benchmark/runs/mol1_kernel_checks_20261119/manifest.json)

**LayerNorm.** Fused LayerNorm is the release default. Switching to PyTorch LayerNorm left the median ipSAE identical for 24 of the 35 tested pairs. Four pairs shifted by more than 0.01. The largest change was **0.7577 → 0.6633** (fused → PyTorch), crossing a 0.7 cutoff. One pair moved from **0.0000 to 0.0127** and another from **0.0115 to 0.0000**. These were the only pair medians that moved to or from zero. Across all 175 individual samples the largest absolute change was 0.0948, ruling out a 0.0 ↔ 0.7 flip in this subset. The LayerNorm comparison ended after these 35 pairs. [Paired LayerNorm scores](../../Benchmark/runs/mol1_kernel_checks_20261119/analysis/partial_layernorm_vs_saved_baseline.csv)

**Triangle multiplication.** With fused LayerNorm and cuEquivariance triangle attention fixed, FlashPairformer and cuEquivariance multiplication produced identical ipSAE scores for all **870 samples from 174 matched pairs**. Their rankings on those pairs were identical. These are interim results from the 01:13 UTC snapshot of the 663-pair run. [Paired kernel scores](../../Benchmark/runs/mol1_kernel_checks_20261119/analysis/interim_kernel_comparison_0113utc.csv) · [Full-run analysis directory](../../Benchmark/runs/mol1_kernel_checks_20261119/analysis/)

For context, the earlier 663-pair run changed several optimizations together. With the tested switches off, pooled AUROC was **0.5852** and the top 24 picks per target included **148 binders** across 13 targets. With the switches on, AUROC was **0.5609** and the same selection budget yielded **141 binders**. Attribution to an individual switch requires a matched single-setting comparison. [Full-panel ranking results](../../Benchmark/runs/mol1_runtime_ab_full_20261119/analysis/summary.json)

## First-sample screening and fixed-slot refill

Four separately seeded requests used the shipped 9fm7 example, batch size two, five requested samples, ten recycles and 200 diffusion steps. Times include trunk computation, diffusion and confidence on a shared H200. Startup, featurization and output writing are excluded.

| Screening policy | Samples produced per request | Batched sample rounds | Inference time | Peak GPU allocation |
| --- | --- | ---: | ---: | ---: |
| Generate all samples | 5, 5, 5, 5 | 10 | 472.84 s | 17.38 GiB |
| Ordinary screening at 0.05 | 1, 1, 1, 1 | 2 | 257.95 s | 21.56 GiB |
| Refill at 0.05 | 1, 1, 1, 1 | 2 | 253.96 s | 22.22 GiB |
| Mixed test, ordinary batching | 5, 5, 5, 5 | 10 | 469.25 s | 21.56 GiB |
| Mixed test, refill | 1, 5, 5, 1 | 6 | 359.28 s | 25.82 GiB |

**Screening accounted for the natural-cutoff gain.** All four first samples scored below 0.05, so each batch could stop together. Refill saved zero additional rounds. All 20 full-baseline scores were also below 0.05. The rate at which a later sample would cross the cutoff on other inputs remains unmeasured.

**Refill helped when some requests continued.** The mixed test forced two requests to continue and let two stop. Refilling their slots with equal-length jobs saved four rounds and **23.4% of observed inference time**, at a cost of **4.26 GiB** additional peak memory. All 12 retained predictions matched their full-baseline counterparts exactly. Production savings depend on the mix of early stops, continuations and compatible queued jobs. [Full benchmark and controls](inference_optimization_summary.md#experimental-fixed-slot-refill-2026-11-19)

## Changes with limited or unproven throughput gains

| Change | What the measurements showed |
| --- | --- |
| Eager CUDA graph replay | Outputs matched exactly with fused LayerNorm. Two baseline runs took **70.14 and 70.26 s**. Replay took **61.01 and 70.78 s** on the shared GPU and raised peak allocation from **9.29 to 10.13 GiB**. The timings leave the speed benefit uncertain. [Eager replay benchmark](inference_optimization_summary.md#eager-replay-default-2026-11-13) |
| Token-diffusion compilation | The external compiled-plus-replay run took **3,281 s** on its slowest shard, versus **3,088 s** for the eager control. It replayed **42.7%** of token-stack calls. Separately, compilation with fused LayerNorm changed a coordinate component by up to **2.407 Å** on 8cnh. [External timing](inference_optimization_summary.md#replay-and-layernorm-diagnosis-2026-11-13) · [Numerical comparison](inference_optimization_summary.md#fused-layernorm-and-repository-example-validation-2026-11-12) |
| RFD4-inspired runtime experiments | Reusing unchanged graph inputs cut counted copy volume by **46%** in the full run. Aggregation metadata reuse, sampler-branch precomputation and graph eviction also worked. Shared-GPU trials established output equivalence but left their individual wall-time benefits uncertain. [Five-candidate comparison](inference_optimization_summary.md#five-candidate-checkpoint-validation-2026-11-13) |
| CPU input prefetch and background writing | Serial preparation and writing took **147.74 s**. Enabling both overlap options took **159.71 s**. GPU contention changed during the test. [Pipeline measurements](inference_batching.md#cpu-pipeline-validation) |
| Moving triangle-attention copies outside the kernel | **4.226 → 4.225 ms** per call including the copies. Row chunking and separate per-input calls were slower. [Kernel measurements](inference_batching.md#triangle-attention-layout) |

**Two changes paid off elsewhere.** Stable atom-to-token aggregation made repeated seeded runs reproducible at fixed shape. Native ipSAE reduced CPU postprocessing time at 1,024 synthetic tokens from **1.16 s** for the PAE JSON round trip and scoring to **11.5 ms** for in-memory scoring. The latter measurement excludes model inference. [Reproducibility checks](inference_batching.md#stable-atom-aggregation-during-cuda-inference) · [Native scoring measurements](../output/ipsae_native_benchmark_rrsya920.json)

The benchmarks use different inputs, kernels, sample chunks and timing boundaries. Each percentage applies to its own comparison. Adding them would overstate what was measured for a single run.
