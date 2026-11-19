# Mol-1 inference optimization: final report

Controlled GPU measurements updated 2026-11-04, covering the implementation
through `9bd411b`. Startup validation added 2026-11-07; experimental
shared-GPU compilation and CUDA replay comparisons were added 2026-11-11–12.
The current defaults include eager-only CUDA replay, enabled on 2026-11-13
after separate validation. Compilation remains opt-in. Historical timing
results retain their original configurations.

## Experimental fixed-slot refill, 2026-11-19

`infer_setting.batch_refill` is opt-in and requires an explicit first-sample
ipSAE cutoff, input batch size greater than one, and sample/confidence chunks
of one. Each request owns its Torch and NumPy RNG streams, trunk tensors,
atom conditioning and sample counter. A stopped or completed request can be
replaced between full diffusion samples while survivors retain their slots.
Batch-wide pair projections are rebuilt on replacement; eager replay copies
updated buffers with the same kernels and precision.

Only consecutive inputs with exactly equal token counts share a session.
Incoming trunks run at the configured B, and placeholders retain B through
the partial tail. Thus controls must also retain B in their tails: ordinary
partial batches can use a smaller B and are a different numerical setup.
Preparation is bounded to one incoming group and does not overlap GPU work
with sampling. Additional cached conditioning and early confidence can raise
peak memory. Failed sessions do not silently retry at a smaller batch size.
Job names, seeds, completion markers and output ownership remain independent;
background writing now retains the top-level screening metadata.

The final CPU regression suite passed **204 tests and three subtests**, with
11 environment/CUDA checks skipped. Tests cover exact surviving and incoming
predictions with changed conditioning and atom counts, all cache modes,
placeholder tails, unavailable scores, incompatible settings, fixed budgets,
missing outputs, model/writer failures and per-job seed ownership.

The short real-checkpoint canary used five requests from the unchanged shipped
9fm7 example, B=2, one recycle, four steps and three samples. All native first
scores were below 0.05; ordinary screening and refill both retained one sample
per request, matching baseline raw predictions, native scores and CIFs exactly.
A test-only continuation override then exercised a mixed workload. Refill used
five batched sample rounds instead of seven, with retained outputs still exact.
Observed inference time was **41.90 s versus 58.57 s**, while peak allocation
was **24.99 GiB versus 21.56 GiB**. These shared-GPU timings include incoming
trunk preparation and confidence but exclude output comparison/writing. They
are not a production speedup or a measured natural acceptance rate. Six
comparable baseline CIFs also matched the pre-refactor November 18 artifacts.

When every request in a batch fails its first score, both screening policies
already need only one sample per request. Refill saves additional diffusion work
only when stopped and continuing requests mix; its benefit depends on that mix
and on trunk/conditioning overhead. The all-fail timings are not evidence of
an additional refill speedup.

The full-settings comparison used four separately featurized seeds of the
unchanged 9fm7 example, **B=2, five samples, ten recycles and 200 steps**.
Reference-feature hashes differ between seeds, so this checks replacement of
conditioning as well as replacement of RNG state. All retained raw coordinates,
confidence logits, contact probabilities, native ipSAE/ranking values and
corresponding generation-order CIFs matched their fixed-B controls exactly.

| Full-settings arm | Samples retained | Batched sample rounds | Inference time | Peak allocation |
| --- | --- | --- | --- | --- |
| No cutoff | 5, 5, 5, 5 | 10 | 472.84 s | 17.38 GiB |
| Ordinary 0.05 screening | 1, 1, 1, 1 | 2 | 257.95 s | 21.56 GiB |
| Refill, 0.05 screening | 1, 1, 1, 1 | 2 | 253.96 s | 22.22 GiB |
| Controlled mixed, ordinary batching | 5, 5, 5, 5 | 10 | 469.25 s | 21.56 GiB |
| Controlled mixed, refill | 1, 5, 5, 1 | 6 | 359.28 s | 25.82 GiB |

The controlled mixed arms use a **test-only override** to continue seeds 103
and 105 despite their low first scores, with a threshold of 1.0 for the other
requests. They exercise both a surviving request and an incoming request whose
neighbour subsequently changes. This is not a measured natural acceptance mix.
Refill executed 1,200 denoiser calls versus 2,000 for the mixed control, retaining
all twelve required samples exactly. Observed inference time decreased by
**23.4%**, with **4.26 GiB** additional peak allocation. Incoming trunk work,
conditioning refresh, sampling and confidence are included. Initialization,
input featurization and output comparison/writing are excluded from the table.
Timings are sequential on a shared H200 and do not establish production throughput.

Across the full run, eager replay used **one graph, 6,000 replays, zero fallbacks
and zero compilation calls**. Fused LayerNorm remained selected throughout.
All twenty full baseline ipSAE scores were below 0.05, so no discarded sample
crossed that threshold in this example; the false-negative rate on other inputs
remains unmeasured. Both validation phases completed; all **107 CIF hashes**,
**36 completion records**, and final source hashes were verified. The feature
remains disabled by default and existing job launchers were not edited.

Artifacts: `output/batch_refill_20261119/`, including reports, output trees,
completion records, source hashes/snapshots and `scripts/benchmark_batch_refill.py`.
The first-sample cutoff remains a heuristic that can miss better later poses;
output equivalence on retained samples does not measure that scientific loss.

## Optional ipSAE early stopping, 2026-11-18

`infer_setting.early_stop_ipsae_threshold` is disabled by default (`None`).
Setting it to 0.05 evaluates the first generated sample after diffusion and
confidence. Remaining samples are skipped only if **all inputs in the current
batch** have a defined native `ipsae_min_10` of zero or strictly below the
threshold. Mixed batches retain every sample and their exact shapes; there is
no active-input compaction or regrouping. Both sample chunks must be one.
The first confidence heads are reused and trunk recycling is not repeated.

The threshold uses the same PAE rounding and caller-selected interface as the
written score. Undefined scores continue normally. Checkpoint tests retain
fused LayerNorm and default eager graph replay. Ordinary global RNG noise draws
for skipped samples are advanced without neural-network calls so subsequent
jobs retain their stream. Summaries include explicit generation indices and
completion counts; an atomic per-seed `sampling_status.json` marks successful
intentional shortfalls. Existing output directories with leftover later samples
are rejected without deleting their files.

The CPU regression suite passed **186 tests and three subtests**, with 11
optional environment/CUDA checks skipped. Targeted cases cover the threshold
boundary, unavailable and explicitly selected multi-chain interfaces, mixed
batches, head reuse, per-input/global RNG streams, sampler call counts,
intentional-shortfall validation, output failure and stale output protection.

The real two-input check used two copies of the unmodified repository 9fm7
example, seeds 101/103, one recycle, four steps and three samples per input.
Both first scores were zero. Screening retained one sample per input, exactly
matching the baseline's first coordinates, confidence, native scores and CIFs.
A test-only override then forced continuation after the same low scores;
all three samples per input and RNG states matched the full baseline exactly.
Peak allocation increased from **16.57 to 21.57 GiB** because early confidence
runs while diffusion caches remain resident.

At **five samples, ten recycles and 200 steps**, the single-input baseline's
generation-order scores were approximately 0.02298, 0.02342, 0.02400, 0.02375
and 0.02556. The 0.05 policy retained sample zero and skipped four; that retained
prediction and the final global RNG states matched exactly. Observed forward
times were **99.53 s** for the five-sample baseline and **52.36 s** for the
screened run. These are sequential shared-GPU timings, excluding preparation
and writing, not a production throughput estimate. Every sample in this one
example was below the cutoff; this does not establish a false-negative rate
for other designs or validate the scientific usefulness of the threshold.
The full-settings forced-continuation check also reproduced all five baseline
samples, native scores, CIFs and RNG states exactly. Both validation phases
completed successfully; all 28 emitted CIF hashes, source hashes and per-seed
completion records were independently rechecked. The benchmark deliberately
disables output ranking so CIFs are compared by generation index; production
ranking behavior is unchanged.

The first confidence pass is included in the diffusion phase timer when this
option is enabled; use the complete forward time for cost comparisons. The
policy can miss better later samples and can increase memory usage. See
[usage and completion semantics](inference_batching.md#optional-first-sample-ipsae-screening).
Artifacts: [batch check](../output/ipsae_early_stop_20261118/batch/report.json),
[full-settings check](../output/ipsae_early_stop_20261118/full/report.json),
[CPU tests](../output/ipsae_early_stop_20261118/cpu_tests.log), and
[reproduction launcher](../output/ipsae_early_stop_20261118/run.sh).
The [combined result](../output/ipsae_early_stop_20261118/summary.json) records
settings, memory, timing, sample counts and replay counters for every arm.

## Current inference defaults

As of 2026-11-13, ordinary CUDA prediction uses the following settings:

| Setting | Default | Behavior |
| --- | --- | --- |
| `infer_setting.fast_init` | `true` | Skip redundant SciPy initialization before a strict, complete checkpoint load. |
| `infer_setting.cpu_thread_limit` | `4` | Cap PyTorch intra-op and not-yet-initialized inter-op CPU pools for CUDA inference; preserve lower limits. Zero leaves process settings alone. |
| `infer_setting.atom_block_cache` | `true` | Reuse fixed atom-block conditioning in input batching with full/shared diffusion caching. |
| `infer_setting.compile_diffusion` | `false` | Opt-in token diffusion compilation with Inductor; independent of replay. |
| `infer_setting.cuda_graphs` | `true` | Replay the eager token stack on CUDA; uses compiled kernels only if compilation is separately enabled. |
| `infer_setting.cuda_graph_max_cache_size` | `2` | Bound retained graph pools; additional signatures run without replay. |
| `infer_setting.compile_dynamic_shapes` | `true` | Compile for varying token lengths; captured graphs still use exact shapes without added padding. |
| `infer_setting.compile_threads` | `1` | Bound Inductor compilation threads. |

**Compilation can change predictions and remains experimental.** No production
speedup has been established. The fused-LayerNorm 8cnh experiment
below measured a 2.407 Å maximum coordinate-component difference between
compiled and eager execution. Compilation is not a promise of unchanged
accuracy or predictions. The selected LayerNorm implementation is retained;
the runner does not switch fused LayerNorm to the PyTorch implementation.

CPU, MPS, context-parallel inference and training retain their eager compute
paths. Fast initialization applies to strict inference loading on all devices;
partial loading requires `fast_init=false`. Sampling counts, sample chunk sizes,
precision, padding controls, MSA/template settings and kernel selection keep
their existing defaults. Prefetch and background writing remain optional.

The CUDA runner lazily installs one execution context after successful checkpoint
loading and reuses it across predictions and input JSONs. It reports compilation,
capture, replay, capacity-fallback and cache-entry counts. Checkpoint reloads,
model reconstruction, prediction failures and CLI exit close the context and
release graph buffers. Direct Python callers should call `runner.close()` when
finished or before moving/replacing the model or its weights. A later prediction
can reopen acceleration. The model requires one caller stream and exclusive
ownership while accelerated; an explicit caller-owned acceleration context takes
precedence over automatic runner setup.

Eager token diffusion with CUDA replay is the default. Use
`--infer_setting.cuda_graphs false` to disable replay and
`--infer_setting.compile_diffusion true` to opt into compilation independently.
Use `--infer_setting.atom_block_cache false` to disable batched atom caches.
The direct `compiled_diffusion_blocks` API retains static compilation by default;
pass `dynamic_shapes=True` to match the runner's varying-length policy.

### Eager replay default, 2026-11-13

At the user's request, ordinary CUDA prediction now enables replay independently
of compilation. `compile_diffusion=false, cuda_graphs=true` captures the original
token-diffusion stack through `eager_diffusion_replay`; no block is compiled and
no LayerNorm method is replaced. CPU, MPS, context-parallel and training paths
retain their existing behavior. The repository sampling defaults remain five
samples and 20 recycles; using ten recycles still requires an explicit setting.

This promotion enables eager replay only. The experimental aggregation-metadata
cache, sampler-branch change, stable-input copy skipping and LRU eviction remain
opt-in experiments. The default replay cache retains at most two exact signatures,
refreshes every input tensor, and clones returned outputs. Subsequent unseen
signatures use eager execution. Graph buffers increase GPU memory consumption;
`--infer_setting.cuda_graphs false` releases/disables them without changing the
compilation setting. No production speedup is established by the shared-GPU
measurements.

Runner regression suites passed **128 CPU tests and three subtests**. The
explicit CUDA suite passed **17 cases**, including ten repeated CPU cases,
six cache/correctness cases and a new two-input runner integration test. The
runner test checks exact output tensors across three lengths, capacity fallback,
checkpoint reload, failure cleanup and absence of `torch.compile` calls.

The released-checkpoint runner comparison used the unmodified 8cnh job from
`examples/example_without_msa.json`, seed 1, **five samples, ten recycles and
200 diffusion steps**, with sample/confidence chunks of one. Two baseline
predictions with replay disabled and two predictions using the new default
matched exactly in coordinates, confidence tensors, rankings, RNG state and
all 20 CIF/20 confidence JSON files. The model retained **764 fused LayerNorm
modules and zero PyTorch LayerNorm modules**. The replay session reported zero
compiled calls, one capture total and 2,000 replays across its two predictions.
This example has only one protein chain, so protein-protein ipSAE is null.

Baseline times were 70.14 and 70.26 seconds; default-replay times were 61.01 and
70.78 seconds. Peak allocated memory increased from approximately 9.29 to
10.13 GiB. These sequential shared-GPU measurements are not an isolated speed
comparison. Other inputs can require substantially larger graph pools.

The two-input check used the same unchanged example with seeds 1 and 2,
one recycle, four steps and two samples per input (sample chunk two). Both
default-replay passes matched both control passes exactly, including all
16 CIFs/16 confidence JSONs. Source hashes and the output hashes from both
validation phases were rechecked after completion. The short batch check is
not a full five-sample/ten-recycle throughput benchmark.

The change affects newly loaded configurations, including subsequent CLI blocks;
existing processes were not restarted or reconfigured. Validation artifacts:
[full-settings runner comparison](../output/eager_replay_default_20261113/full/report.json),
[two-input comparison](../output/eager_replay_default_20261113/batch/report.json),
[combined results](../output/eager_replay_default_20261113/summary.json),
[launcher](../output/eager_replay_default_20261113/run.sh) and
[CUDA test log](../output/eager_replay_default_20261113/tests.log).

### Replay and LayerNorm diagnosis, 2026-11-13

Compilation and compiled replay were initially returned to opt-in. Later the
same day, separately validated eager-only replay became the default at the
user's request; the current settings above supersede this historical rollback.
Passing correctness and repeat
checks did not establish a speed improvement or compatibility with existing
score pools, so the earlier promotion was premature. Fast initialization, the
four-thread CUDA host cap and atom-block caching remain enabled.

The external benchmark's original logs were inspected, without rerunning its
predictions. Both arms used PyTorch LayerNorm, seed 101, five samples, ten
recycles, 200 steps, identical batch-size sequences and an eight-signature graph
cache. The graph arm ran first and the control second. Both still used **128
intra-op and 128 inter-op threads per process**; this predates the four-thread
default. The following figures are different timing scopes:

| Scope | Compiled plus replay | Eager control | Increase |
| --- | ---: | ---: | ---: |
| Slowest shard wall time | 3,281 s | 3,088 s | 6.25% |
| Summed wall time across eight shards | 24,528 s | 24,250 s | 1.15% |
| Summed forward timings including output submission | 23,546.31 s | 23,364.57 s | 0.78% |

The largest forward-time regressions were on shards zero and six; six other
shards were slightly faster. These sequential runs do not isolate GPU
contention from implementation overhead. They show no net speedup, but do not
establish that replay itself makes every forward 6.25% slower.

Counters show **18,600 replays and 25,000 capacity fallbacks**, or **42.66% of
token-stack calls replayed** across 218 groups and 663 inputs. This is not the
fraction of total model time accelerated. Source inspection identifies costs
that limit the benefit:

- The cache keeps the first signatures without eviction. A signature includes
  shapes, strides, batch/sample dimensions, scalar arguments and execution
  settings; changing token length alone can require a new graph. Once full,
  later unseen signatures keep using compiled execution without replay.
- Every replay copies all tensor arguments, including request-constant pair
  conditioning/projections, into graph-owned storage and clones returned tensors.
  These operations preserve correctness across requests but consume bandwidth
  and launches outside the graph.
- Each new graph performs three warmup executions and a capture, and each
  retained signature owns a graph pool. Observed peak allocation rose from
  **26.57–26.89 GiB** for control to **40.93–45.12 GiB** for the graph arm.
- Only token diffusion is captured. Trunk recycling, atom processing,
  confidence prediction, input preparation and writing retain their existing
  execution paths. The external harness also used static compilation, so new
  shapes could require compilation independently of graph-cache capacity.

The logs contain no per-operation profile that apportions the overhead among
compilation, capture, copies, synchronization and GPU contention. Increasing
cache capacity alone is therefore not a demonstrated solution.

The statement that compilation requires `LAYERNORM_TYPE=torch` describes the
earlier harness, not the current fused-compatible implementation. The later
repository-example test used **764 fused LayerNorm modules and zero PyTorch
LayerNorm modules**. Its custom operator calls the original fused kernel;
correcting CUDA stream selection enabled capture. This removes the backend
switch, but compilation still changed eager coordinates in that test.

A new synthetic CUDA probe held tensors, weights, epsilon and seed fixed.
Each kernel repeated bitwise, while fused and PyTorch outputs differed. At
shape `[512, 768]`, **16 of 393,216 BF16 values differed**, with maximum absolute
difference **0.0078125**. FP32 cases differed by up to **1.90735e-6**. The two
paths use the same normalization formula and epsilon, with different reduction
and rounding implementations. Small deterministic perturbations can propagate
through recycling and diffusion; the exact amplification through the external
corpus was not traced here. Changing the global LayerNorm backend affects the
model beyond the compiled token stack and must not be treated as seed noise.
Comparing different samples or best-of subsets also does not replace a
controlled comparison that changes only the seed.

The default rollback passed **105 CPU regression tests and three subtests**,
with 12 optional tests skipped. No production process was restarted and no
external launchers were edited. Artifacts:
[log analysis](../output/replay_diagnosis_20261113/log_analysis.json),
[analysis script](../output/replay_diagnosis_20261113/analyze_logs.py),
[LayerNorm probe](../output/replay_diagnosis_20261113/layernorm_probe.py), and
[probe results](../output/replay_diagnosis_20261113/layernorm_probe.json).

### RFD4-Proteina runtime review, 2026-11-13

Reviewed the clean local RFD4-Proteina checkout at
`59e5abb1baa46bb10901812b875b738a15aa2d77`, including its installed
`bionemo-ir 0.2.0` runtime. This was a source review with small synthetic probes;
no production model code, defaults, launchers or jobs changed during this review.

RFD4 has two different execution paths. Its native path compiles individual
trunk blocks, pads to configured token buckets, sizes Dynamo caches for those
buckets, and can persist compiler artifacts. Its BioIR path uses explicit CUDA
graphs around a repeated trunk step, with stable-input handling and bounded
graph admission. The BioIR mechanisms are the more relevant lessons for
Mol-1's current replay implementation.

| Priority | Candidate for Mol-1 | Evidence and scope |
| --- | --- | --- |
| First | Cache atom-aggregation metadata per request | `_aggregate_atom_to_token_grouped` repeats stable sorting and `bincount` for an unchanged atom map on every encoder call. Reuse the order and segment lengths while retaining the existing `segment_reduce` and division. Applies to both ordinary and batched inference. |
| First, single-input path | Precompute sampler branch decisions | `sample_diffusion` evaluates a CUDA scalar in Python every step and sample chunk. The batched sampler already computes those decisions once. Extend that existing approach without changing the GPU schedule arithmetic or RNG draws. |
| Replay experiment | Avoid recopying unchanged conditioning | BioIR distinguishes stable kwargs from changing state and skips copies when tensor identity/version still match. Mol-1 copies every tensor every replay, including request-constant pair projections. |
| Replay experiment | Evict old graphs and admit by memory/size | BioIR uses LRU eviction, estimated memory budgets and token limits. Mol-1 permanently retains the first signatures, then falls back for every new shape. Request-aware admission could replace obsolete entries without padding or regrouping inputs. |
| Replay experiment | Capture eager kernels independently of compilation | BioIR's explicit graph runtime wraps ordinary calls. Mol-1's lower-level replay wrapper also works on a small eager token stack with fused LayerNorm, although the current runner exposes replay only with compilation. This would isolate replay overhead from compiler numerical changes. |
| Secondary | Test `inference_mode` and preserve allocator reuse | RFD4's Lightning validation uses inference mode; Mol-1 uses `no_grad`. Mol-1's batched path already keeps allocator caches warm, but the single-input CLI still calls `empty_cache()` after each success. Both need full-path validation before adoption. |
| Compiled runs only | Persist compiler artifacts and isolate local caches | RFD4 saves a per-user/per-Torch-build compiler artifact blob and uses a local rank-specific Inductor directory. This can reduce compilation/startup work across invocations; it does not accelerate eager production forwards. |

**Measured probes.** On a shared H200, a synthetic `[5, 4096, 128]`
atom-feature tensor with 512 token groups gave the following medians over four
alternating measurements of 100 calls each:

| Aggregation | Rebuild metadata each call | Reuse metadata | Outputs |
| --- | ---: | ---: | --- |
| FP32 | 0.2186 ms | 0.1171 ms | Bitwise identical |
| BF16 | 0.2204 ms | 0.1191 ms | Bitwise identical |

This saves about 0.10 ms per aggregation in this microbenchmark, not 46% of
model inference. The candidate retains the stable reduction that fixed the
original CUDA reproducibility bug; it does not return to unordered atomics.

The sampler probe used 200 steps, five samples and chunk size one, with a
synthetic denoiser and identical per-input RNG streams. Moving only branch
decisions out of the loop reduced `aten::item` and `aten::_local_scalar_dense`
counts from **1,000 to zero** and preserved output bits. Reading the schedule
once still transfers it to the host. This does not offer the same saving to
the production batched path, which already performs this optimization.

The eager-replay probe used a two-block token transformer with **ten fused
LayerNorm modules** and made `torch.compile` raise if called. Two exact shapes
produced **two captures and eight replays**, with bitwise matches to eager
outputs, including in-place conditioning edits. RNG state and retained outputs
were preserved, and closing the cache released its entries. This establishes
feasibility for a small stack, not full-checkpoint equivalence or speed.

For stable replay inputs, a direct copy of BioIR's policy would need adaptation:
its inference-tensor case accepts identity alone because those tensors have no
version counter. Mol-1 explicitly tests in-place edits to conditioning. A Mol-1
implementation should use a request-level immutable-data contract or continue
copying when mutation tracking is unavailable. The aggregation cache likewise
needs request ownership and invalidation when the atom map changes.

Do not transfer RFD4's padded compile buckets or its `medium` float32 matmul
setting into the existing Mol-1 score pool. Both can change numerical behavior.
The native Inductor tree-reset workaround also does not apply to Mol-1's
explicit `CUDAGraph` cache. RFD4's pinned input loader is worth profiling only
if transfer time is material; Mol-1's prior CPU-overlap experiment showed no
throughput benefit. The full BioIR model backend is a separate implementation,
not a checkpoint-compatible switch for Mol-1.

Recommended implementation order: request-scoped aggregation metadata, then
single-input sampler/allocator housekeeping; separately evaluate stable-input
copy avoidance and eager-only replay with cache eviction. Keep kernels, exact
tensor shapes, grouping, seeds and sample chunks fixed. Require actual Mol-1
example output comparisons before promotion and isolated, alternating timing
passes before claiming production speed gains.

Source anchors:
[RFD4 compile/cache helpers](/storage/frank/RFD4-Proteina/src/rfproteina/utils/torch_compile.py:78),
[RFD4 native shape reservation](/storage/frank/RFD4-Proteina/src/rfproteina/nn/local_latents_transformer.py:224),
[RFD4 BioIR graph policy](/storage/frank/RFD4-Proteina/src/rfproteina/nn/bioir_full.py:571),
[BioIR stable inputs](/storage/frank/RFD4-Proteina/.venv/lib/python3.12/site-packages/bionemo_ir/_torch/graph_optimization/cuda_graph/runtime.py:601),
[BioIR exact graph configuration](/storage/frank/RFD4-Proteina/.venv/lib/python3.12/site-packages/bionemo_ir/models/rfproteina/config.py:33),
[Mol-1 aggregation](../mol/model/utils.py), and
[Mol-1 replay cache](../mol/model/cuda_graph_replay.py).

Artifacts:
[source identity](../output/rfd4_runtime_review_20261113/source_snapshot.json),
[aggregation/sampler probe](../output/rfd4_runtime_review_20261113/probe_runtime_costs.py),
[probe results](../output/rfd4_runtime_review_20261113/probe_runtime_costs.json),
[eager replay probe](../output/rfd4_runtime_review_20261113/probe_eager_replay.py), and
[replay results](../output/rfd4_runtime_review_20261113/probe_eager_replay.json).

### Five-candidate checkpoint validation, 2026-11-13

The five candidates above are implemented as **explicit benchmark experiments**
in [runtime_experiments.py](../mol/model/runtime_experiments.py) and
[benchmark_runtime_candidates.py](../scripts/benchmark_runtime_candidates.py).
The ordinary runner does not import or install these wrappers. The sampler's
new `precompute_schedule_branches` argument defaults to false. No production
processes were restarted, external launchers edited, or inference defaults
changed for this experiment.

Seven arms isolate the baseline, aggregation metadata reuse, sampler branch
precomputation, eager graph capture, stable-input copy avoidance, graph eviction,
and their combination. The three replay variants share eager capture as their
control; none invokes `torch.compile` or switches the LayerNorm implementation.
The checkpoint reports 764 fused LayerNorm modules and zero PyTorch LayerNorm
modules. Precision, exact token lengths, seeds, sampling configuration and
kernel selection are held fixed within each comparison.

The short single-input sequence uses the unmodified `8cnh-assembly1` and
`8j2w-assembly1` jobs from `examples/example_without_msa.json`, with 509 and
1,000 tokens respectively. The sequence `0,1,0` runs twice per arm with one
recycle, three diffusion steps, two samples and sample chunk size one. One
retained graph forces shape eviction; the accounted graph-allocation budget is
12 GiB and the process allocator is capped at 40% of the shared H200. This is
not a hard bound on transient capture memory or total CUDA allocations.

All **42 short predictions** matched their eager controls exactly: coordinates,
PAE, PDE, pLDDT, resolved probabilities, native ipSAE, ranking, global Torch/CUDA
and NumPy RNG states, and emitted CIF/confidence JSON bytes. The two-protein
8j2w example supplies a non-null protein-interface comparison; 8cnh has only
one protein chain and cannot establish protein-protein ipSAE equivalence alone.

| Candidate | Observed mechanism in the short sequence |
| --- | --- |
| Aggregation metadata cache | Two metadata builds and five cache hits per prediction; unchanged sorted segment reduction. |
| Sampler branch precomputation | Exact output and RNG checks passed; only branch decisions move to the host once, preserving GPU schedule arithmetic. |
| Eager graph capture | First-signature cache: one capture, 24 replays and 12 capacity fallbacks across six predictions. |
| Stable-input copy avoidance | Same replay coverage; copies fall from 20.750 GB to 12.791 GB, a 38.36% reduction in counted replay input bytes. |
| Graph eviction | Five captures, four evictions, 36 replays and no capacity fallbacks. Recapture overhead remains a cost. |

The full-settings single-input comparison then ran `8cnh-assembly1` twice per
arm at **20 recycles, 200 steps, five samples and sample chunk size one**.
All **14 predictions** passed the same exact-output, RNG and file checks,
producing 70 CIFs and 70 confidence JSONs. Aggregation caching built metadata
twice and reused it 999 times per prediction. Each graph arm replayed all
1,000 token-stack calls per prediction, with one capture on its first pass and
none on its second. Stable-input handling reduced counted replay copies from
864.567 GB to 467.017 GB per prediction, a **45.98% reduction**. The full-settings
test has one shape; eviction coverage comes from the short shape-changing
sequence, not from this repeated single-shape run.

The following times include prediction, transfers and confidence, but exclude
preparation and writing. The first/second columns are sequential repetitions,
not randomized or isolated GPU trials. A competing process was present before
and after every pass, so these figures **do not establish speedup**:

| Arm | First prediction | Second prediction | Peak allocated memory |
| --- | ---: | ---: | ---: |
| Baseline | 95.57 s | 88.38 s | 9.29 GiB |
| Aggregation metadata | 87.05 s | 85.29 s | 9.29 GiB |
| Sampler branches | 83.98 s | 92.91 s | 9.29 GiB |
| Eager graph capture | 84.87 s | 89.41 s | 10.13 GiB |
| Graph plus stable inputs | 84.91 s | 86.06 s | 10.13 GiB |
| Graph plus eviction | 89.18 s | 83.73 s | 10.13 GiB |
| All five combined | 87.59 s | 80.71 s | 10.13 GiB |

The final batched check used two independent copies of the unchanged 8cnh
example with seeds 1 and 2, input batch size two, one recycle, four diffusion
steps, two samples per input and sample chunk size two. All **14 batch passes**
(28 input predictions, 56 CIFs and 56 confidence JSONs) matched their batched
controls exactly. No padding was added. The batched sampler already precomputes
its branch decisions, so the sampler-only arm is a control there; its new
single-input implementation is exercised by the preceding comparisons.

Across these three phases, **70 passes / 84 input predictions** produced
**210 CIFs and 210 confidence JSONs**, whose saved hashes were independently
rechecked after completion. Source hashes also match, including archived
versions of the harness before and after adding multi-input output checking.
These tests cover two example complexes and the stated seeds/settings; they
do not establish equivalence for an entire production corpus.

**Recycle-count clarification:** this checkout's CLI `--cycle` default,
`get_default_runner(n_cycle=...)` default and base `model.N_cycle` are all 20;
the sample default is five. The Stage 1 configuration discussed alongside this
work explicitly uses ten recycles. The full-settings experiment above used
20, not ten, and its timings must not be presented as a ten-recycle benchmark.
No sampling defaults changed during this validation.

Mutation tests cover tensor replacement, ordinary in-place edits and graph
output retention. Inference tensors have no version counter and are always
recopied; identity alone is not accepted as proof of unchanged contents.
Aggregation metadata is discarded after each prediction. A graph that exceeds
the accounted memory budget falls back to eager execution without repeated
capture attempts for that signature. These helpers require exclusive ownership
of the wrapped model and do not support mutations through `.data` or external
untracked CUDA writes.

CPU regression suites passed **114 tests**, with optional CUDA/environment
checks skipped. The explicit fused-LayerNorm CUDA suite passed **16 tests**
(ten CPU cases repeated plus six CUDA cases), including exact replay, shape
churn, memory-budget rejection and conservative inference-tensor refresh.

Timings use a GPU shared with an existing job and exclude featurization,
comparison and file writing. They do not establish a production speedup.
Artifacts: [short comparison](../output/runtime_candidates_20261113/smoke/report.json),
[full-settings comparison](../output/runtime_candidates_20261113/full/report.json),
[two-input comparison](../output/runtime_candidates_20261113/batch/report.json),
[combined results](../output/runtime_candidates_20261113/summary.json),
[test results](../output/runtime_candidates_20261113/test_results.json), and
[validation launcher](../output/runtime_candidates_20261113/run.sh).

### Default-promotion validation, 2026-11-12

This records the temporary promotion, reverted on 2026-11-13. The results
demonstrate execution and repeatability, not a production speedup or equivalence
to eager predictions. To reproduce the accelerated CLI paths with current code,
explicitly enable both compilation and replay; the archived script used the
then-current defaults. Source hashes refer to the archived validation snapshots.

The CPU regression suite passed **100 tests, with 12 skipped and three subtests**.
Two additional tests cover garbage collection and input-transfer failure cleanup;
the final targeted cleanup/checkpoint/CLI suite passed **31 tests, with one CUDA
test skipped**. The separate CUDA integration test passed across 12 token lengths,
including replay-cache capacity fallbacks, repeated shapes and checkpoint reload.

The real CLI used unchanged jobs from `examples/example_without_msa.json`, the
released checkpoint and default fused LayerNorm, with no acceleration flags or
CPU-thread environment overrides. Logs confirmed four intra-op/inter-op threads
and enabled fast initialization, compilation and CUDA replay.

| CLI check | Inputs and sampling | Output structures | Captures / replays / capacity fallbacks |
| --- | --- | ---: | ---: |
| Single-input path | 8cnh and 8j2w; two seeds, one recycle, three steps, two samples | 8 | 2 / 24 / 0 |
| Batched path | Same jobs and sampling, batch size two | 8 | 1 / 12 / 0 |
| Full defaults | 8cnh; seed one, 20 recycles, 200 steps, five samples | 5 | 1 / 1000 / 0 |

All 21 structures had finite coordinates and expected atom counts; all 21
confidence summaries had finite numeric values. Repeating the full-default run
after the final cleanup change produced byte-identical CIFs and confidence
summaries. This verifies repeated compiled execution on this example, not
equivalence to eager predictions. Source hashes matched the final validation
snapshot. These checks ran on a shared H200 and do not establish an isolated
throughput improvement.

Reproduction artifacts: [commands](../output/inference_defaults_20261112/run.sh),
[input provenance](../output/inference_defaults_20261112/inputs_manifest.json),
[validation report](../output/inference_defaults_20261112/validation.json),
[single-input log](../output/inference_defaults_20261112/single.log),
[batch log](../output/inference_defaults_20261112/batch.log), and
[full-default log](../output/inference_defaults_20261112/full_final.log).

## Earlier controlled measurements

**Earlier controlled speed on one H200:** the real 9fm7 + 8cnh batch takes
**61.13 seconds with atom caching**, producing ten structures. The four-example
mixed-length queue takes **134.02 seconds with singleton groups**, producing
twenty structures, with atom caching off. Both retain **five samples per input,
ten recycles and 200 diffusion steps**. These are separate model-time
measurements, excluding preparation and file writing; the combination of both
optimizations has not been timed.

Native `ipsae_min_10` is written directly into each confidence summary. The
fixed-shape reproducibility bug is fixed. A separate padding sensitivity was
diagnosed: changing tensor shape can substantially change predictions and ipSAE,
even when repeated runs at each fixed shape reproduce exactly.

## What is implemented

| Area | Current behavior and finding |
| --- | --- |
| Independent-input batching | Different-length complexes can share a forward without MSA/templates. Input batch size is separate from samples per input. Length sorting, padding-work and pair-memory limits, partial batches, and smaller-batch OOM retries are supported. More inputs per batch are not necessarily faster. |
| Multiple GPUs | One model process per GPU owns disjoint input jobs, including all seeds of each job. Each rank batches locally. This pass did not measure multi-GPU scaling. |
| Fused kernels | Explicit FlashPairformer/TMK3 multiplication works in the batched path, with shared packed weights and per-input calls. cuEquivariance provides batched triangle attention. The missing `opt_core` dependency was resolved in the `mol1` environment used for validation. |
| Diffusion and sample handling | Trunk padding buffers, diffusion conditioning and masks are reused. Diffusion and confidence can process five samples together using their chunk controls; the sample count stays five. Changing diffusion chunk size can change RNG ordering. |
| Native ipSAE | The writer scores in-memory PAE and writes `ipsae_min_10`, its reference-normalization variant, and chain-pair records without an external script or full-data JSON round trip. |
| FP32 attention layout (`5059413`) | Bounded 4D SDPA layouts select an efficient CUDA kernel; shared pair bias stays a view. Restricted to FP32, no-gradient, no-autocast CUDA execution. Training and the BF16 trunk retain their paths. |
| Stable atom aggregation (`dc12f1c`) | Sorted, segmented atom-to-token aggregation removes unordered BF16 CUDA atomic sums in the affected inference path. Enabled by default; historical ordinary-mode predictions can change. |
| Atom-block cache (`2ed8457`) | Computes fixed encoder/decoder pair biases, adaptive LayerNorm terms and gates once per request. Enabled by default, request-local, and available in the batched path with `full` or `shared` diffusion caching. |
| CPU overlap (`174c9cf`) | Optional bounded input prefetch and a background writer preserve seed ownership and drain writes before exit. Both remain off by default because no throughput gain was demonstrated. |
| Measurement and diagnosis (`c91c1c5`, `9bd411b`) | Added real-input queue benchmarks, attention-layout screening, padding controls and a masked-padding regression. The padding diagnosis changed no production inference code or defaults. |

See [batching and GPU ownership](inference_batching.md),
[kernel installation and selection](kernels.md), and
[native ipSAE definitions](output_format.md#native-ipsae).

## How fast it is now

The final targeted measurements used `conda activate mol1`, Torch
2.13.0+cu130, the cached `mol-1-base.pt` checkpoint, PyTorch LayerNorm, TMK3
multiplication and cuEquivariance attention. The trunk used BF16; diffusion and
confidence retained FP32. MSA, templates and TF32 were off. Diffusion and
confidence sample chunks were both five.

### Atom caching: real two-input batch

Real 9fm7 has 644 tokens and 8cnh has 509. Each implementation had a full
warmup, followed by alternating uncached/cached/cached/uncached timed passes.

| Configuration | Timed passes (seconds) | Median (seconds) | Peak allocated GPU memory |
| --- | --- | ---: | ---: |
| Cache off | 63.372, 63.342 | 63.357 | 44.46 GiB |
| Cache on | 61.112, 61.147 | 61.130 | 44.46 GiB |

This saves **2.23 seconds per two-input batch, or 3.52% of model time**.
Coordinates, PAE, atom pLDDT, native ipSAE and rankings matched bitwise,
including repeats. GPU process monitoring found no competing process. This
clean measurement supersedes the earlier inconclusive cache timings on shared
GPUs; it establishes a modest gain on this cohort, not a universal percentage.

### Grouping: real four-input queue

The complete repository examples, including their ligands, were 7r6r
(245 tokens, seed 101), 8cnh (509, seed 103), 9fm7 (644, seed 107), and 8j2w
(1,000, seed 109). **Atom caching was off in every row.**

| Grouping by token count | Model seconds for all four inputs | Measurement | Peak allocated GPU memory |
| --- | ---: | --- | ---: |
| `[245]`, `[509]`, `[644]`, `[1000]` | **134.021** | Median of two passes | 102.43 GiB |
| `[245, 509]`, `[644, 1000]` | 166.985 | Median of two passes | 104.05 GiB |
| `[245]`, `[509, 644]`, `[1000]` | 143.343 | One screening pass; padding ratio cap 1.4 | 102.43 GiB |
| `[245, 509, 644, 1000]` | 237.882 | One screening pass | 105.05 GiB |

Singleton groups used **19.74% less model time** than uncapped batches of two.
The best measured rate corresponds to about **107 inputs/hour per GPU**, or
537 generated structures/hour at five samples per input, if this exact queue
were repeated with no preparation or writing overhead. It is not measured
end-to-end throughput or an estimate for a different length distribution.
No competing GPU process was observed during the completed grouping sweep.

The singleton rows use `predict_batch([input])`, preserving the batched
per-input RNG stream. They do **not** use the legacy CLI `--batch_size 1` path.
For this distinct-length queue, `--batch_size 2` with
`--infer_setting.batch_max_padding_ratio 1.0` produces singleton groups through
the batched scheduler. Equal-length inputs can still share a batch.

**Grouping changes can change predictions.** The faster grouping above is a
throughput choice, not an output-equivalent replacement for every padded
batch. The padding diagnosis below explains the observed 9fm7 score change.
Do not add the 19.74% grouping reduction to the 3.52% cache reduction: they
come from different cohorts and comparisons.

### Preparation and writing: no demonstrated gain from overlap

With fixed `[245]`, `[509, 644]`, `[1000]` groups, the serial scheduler measured
**147.74 seconds** and prefetch plus background writing measured **159.71
seconds** (two-pass medians). These include preparation, prediction hashing and
all completed CIF/summary writes, with the model already loaded and parent CCD
caches warm. Each spawned worker starts cold; these are not cold CLI timings.

An unrelated GPU process appeared near the end of the first serial pass and
remained during later passes. Consequently these are shared-GPU observations,
not a controlled speedup or slowdown estimate. All raw predictions and all
40 output files per pass matched exactly. Leave overlap off unless a longer,
representative queue demonstrates a benefit.

### Earlier gains and changes that did not pay off

- The earlier synthetic 512/640-token benchmark improved from **128.30 to
  62.33 seconds per two-input batch**, or **2.06× throughput**, using the
  optimized path, TMK3/cuEquivariance and confidence chunking. Both configurations
  already used diffusion chunks of five. This is a combined configuration
  comparison on synthetic inputs, not a real-example before/after for the final
  cache. The smaller-input gain depended heavily on the pre-existing sample
  chunk control. See [the original measurements](inference_batching.md#reproducible-throughput-measurements).
- The 4D attention change measured **35.4 → 35.4 seconds** for real 9fm7 alone
  and **68.8 → 66.7 seconds** for 9fm7 + 8cnh in the earlier controlled
  deterministic comparison. No single-input gain was measurable; the batch
  gain was about 3%, below the initial estimate.
- Triangle attention was the largest measured trunk substage in the
  instrumented profile, but its cuEquivariance operation already makes Q/K/V
  contiguous internally. Moving the copies outside measured **4.226 → 4.225
  ms per call**; row chunking and separate input calls were slower. These were
  cost-inclusive kernel tests. The production layout was left unchanged.
- CUDA graphs/compilation, a new batched TMK3 kernel, and BF16 confidence are
  not validated speedups from this work. Confidence precision, sample count,
  recycle count and diffusion steps were not reduced.

## Correctness: what was fixed, and what remains shape-sensitive

**Repeated-run nondeterminism is fixed in the tested inference path.** Before
stable atom aggregation, unchanged ordinary runs with the same seed could
differ by tens of ångströms and produce different PAE/ipSAE. After the fix,
tested ordinary runs repeated bitwise and matched the explicit deterministic
reference. This default behavior change can alter historical predictions;
see the [compatibility changelog](../CHANGELOG.md#prediction-compatibility).

**Attention layout is not bitwise equivalent.** The mixed real-input comparison
had a maximum coordinate-component change of 0.422 Å on 8cnh, a maximum atom
displacement of 0.456 Å, and coordinate-component RMSE of 0.00261 Å. Native
9fm7 ipSAE and sample ordering were exact; other confidence values changed
slightly. 8cnh has one protein chain, so its protein-interface ipSAE is null.
These checks do not establish unchanged biological accuracy.

**Padding sensitivity was diagnosed, not eliminated.** Real 9fm7 at seed 107
gave five native ipSAE values around 0.02525–0.02665 at 644 tokens, and all zero
when its trunk was padded to 1,000 tokens. Forced padding without another input
reproduced the difference. Controls found:

- Changing only the internal attention chunk size gave bitwise-identical
  predictions. Padding only diffusion preserved ipSAE and changed coordinates
  by at most 0.00450 Å per component.
- Padding the trunk changed final coordinates substantially: component RMSE
  **23.814 Å**, maximum component change **86.421 Å**, and aligned all-atom
  RMSD **28.6–33.7 Å** across samples.
- The earliest detected change was just **15 of 185,802,752 valid BF16 pair
  values** in the first incoming triangle multiplication. Small shape-dependent
  floating-point differences amplified through ten recycles. Initial diffusion
  noise was exact.
- Randomizing masked padding left the padded prediction bitwise unchanged.
  Stock cuEquivariance matched padded TMK3 predictions bitwise, and a separate
  process repeated them exactly. No masking or scoring failure was reproduced.
- The padded runs had no interchain PAE below 10 Å; their minima were
  11.57–13.01 Å. The strict ipSAE cutoff therefore correctly returned zero.
  FP32 trunk controls reduced, but did not eliminate, padding divergence and
  did not restore the original BF16 scores.

For comparisons, hold actual grouping, tensor shapes, kernels, precision and
sampling settings fixed. A cap of 1.0 prevents unequal-length padding in the
tested queue but is not a universal bitwise-equivalence guarantee. No accuracy
benchmark establishes which of the differing structures is better. Full
controls and reproduction commands are in the
[padding diagnosis](inference_batching.md#padding-sensitivity-diagnosis-2026-11-04).

Native ipSAE matches the documented postprocessing convention, including PAE
rounding to two decimals before `PAE < 10`. The `min` field uses the smaller
directional score and count clip 27; `ipsae_min_10_d0ref` uses clip 26. This is
not a promise of equality with scripts using different normalization or
directional aggregation. Cache and CPU-overlap comparisons preserved the score
exactly at fixed grouping. Without an explicit interface selection, fewer or
more than two protein chains yield a null top-level value. Jobs can now declare
[`ipsae_interface`](infer_json_format.md#ipsae-interface-selection) to score two
groups of protein chains; pair records remain available in either case.

## Configuration to use as a starting point

For this varied-length workload on an H200, use the batched scheduler with
minimal unequal-length padding, five-sample chunks, explicit TMK3/cuEquivariance
and the atom cache (now enabled by default):

```bash
conda activate mol1
MOL_ROOT_DIR=/storage/frank/mol1_root LAYERNORM_TYPE=torch \
  mol pred -i inputs.json -o output/optimized \
  --use_msa false --use_template false --use_rna_msa false \
  --batch_size 2 --sample 5 --cycle 10 --step 200 \
  --trimul_kernel flashpairformer --triatt_kernel cuequivariance \
  --infer_setting.sample_diffusion_chunk_size 5 \
  --infer_setting.confidence_sample_chunk_size 5 \
  --infer_setting.atom_block_cache true \
  --infer_setting.batch_max_padding_ratio 1.0
```

`inputs.json` should contain the independent jobs in one list. Preserve your
seed selection; use `--use_seeds_in_json true` if seeds are stored in each job.
The command combines separately validated settings; **its combined throughput
has not been measured**. The 1,000-token example used about 102 GiB allocated
GPU memory with five-sample chunks, so this is not a low-memory preset.
The default `full` diffusion cache enables atom caching. TF32, prefetch,
background writing and full-data JSON remain off; native ipSAE still appears
in every summary. Default CLI batching and chunk sizes were not changed.

## Validation and retained evidence

See [startup and model reuse](#startup-and-model-reuse) for the separate
2026-11-07 validation. Counts below describe the earlier GPU optimization pass.

Latest CPU suite: **296 passed, 65 skipped, 27 subtests passed**, with three
pre-existing pandas failures in `tests/test_dataset_vectorization.py`:
`TestMolGroupType.test_all_nan`, `TestMolGroupType.test_basic`, and
`TestCalcWeightsSortedEntityId.test_with_nan`. The full suite is therefore not
entirely green. Focused CUDA checks: **51 passed and 86 subtests passed**.
These groups cover batching, kernels, native ipSAE and the masked-padding
regression. Earlier overlapping test groups must not be added to these counts.

The reports retain seeds, settings, source hashes, comparisons and GPU
monitoring. Local artifacts below are retained under the ignored `output/`
directory; a source checkout alone will not contain them.

| Evidence | Local report |
| --- | --- |
| Earlier synthetic optimization comparison | [batch_optimization_report.json](../output/batch_optimization_report.json) |
| Real attention comparison | [mol1_attention_9fm7_report.json](../output/mol1_attention_9fm7_report.json) |
| Aggregation reproducibility | [mol1_repro_report.json](../output/mol1_repro_report.json) |
| Clean cache, layout and queue pass | [mol1_targeted_pass/summary.json](../output/mol1_targeted_pass/summary.json) |
| Padding diagnosis and final tests | [mol1_padding_diagnosis/summary.json](../output/mol1_padding_diagnosis/summary.json) |

The targeted-pass report records the padding sensitivity as unresolved at that
point; the later padding report supplies the diagnosis. Its instrumented,
sometimes shared-GPU runs are not performance measurements.

Reproduction tools are [validate_attention_9fm7.py](../scripts/validate_attention_9fm7.py),
[benchmark_inference_queue.py](../scripts/benchmark_inference_queue.py),
[benchmark_triangle_layout.py](../scripts/benchmark_triangle_layout.py), and
[diagnose_padding_9fm7.py](../scripts/diagnose_padding_9fm7.py). The detailed
[batching guide](inference_batching.md) preserves commands and historical runs.
Remaining performance work needs a representative production queue and clean
GPU measurements; this report does not establish a further guaranteed gain.

## Startup and model reuse

`--infer_setting.fast_init true` skips the SciPy truncated-normal draws used
to initialize linear-layer weights that a complete checkpoint overwrites.
It defaults to **true** and requires `load_strict=true`; missing checkpoint
entries still fail. Other constructors, initializers, buffers, device transfers
and checkpoint loading retain their normal behavior. Training and direct
`Mol(configs)` construction always use normal initialization.

Loading now verifies checkpoint structure, key coverage and tensor shapes
before copying weights. Successful strict loads report the exact number of
model tensors loaded, with zero missing or unexpected keys. The runner exposes
`checkpoint_loaded` and `checkpoint_load_report` for API callers. Prediction
is blocked before loading, after any load failure, or if the model is rebuilt
or replaced. A subsequent successful load restores readiness. Explicit
`load_strict=false` with normal initialization remains supported and reports
partial coverage with a warning; it cannot be combined with `fast_init`.

The switch changes startup, not inference kernels, precision, batching,
recycles or sampling. The CLI seeds preparation/inference after model loading.
Direct API users should likewise seed their prediction: skipping the draws
changes the NumPy RNG state left by construction.

On 2026-11-07, two trials per mode using the released 576,788,351-parameter
checkpoint on CPU with two PyTorch threads measured:

| Mode | Model construction (s) | SciPy draws (s) | Checkpoint load (s) |
| --- | --- | --- | --- |
| Normal | 71.04, 54.51 | 66.64, 50.70 | 2.10, 1.45 |
| Fast | 2.37, 2.39 | 0, 0 | 1.26, 1.19 |

Normal construction made 1,519 SciPy draw calls per model; they accounted for
93–94% of construction time. Both modes produced identical hashes of the
complete loaded state and all returned prediction data in every trial. These
are CPU FP32 checks on a small synthetic input, not a production BF16/CUDA
equivalence or throughput benchmark. GPUs were occupied and were not used.
Host timing varied between trials, and checkpoint caching affects load times.
Raw results: [startup report](../output/model_startup_20261107/report.json).

The initial startup, CLI, checkpoint, batching and ipSAE regressions passed
**100 tests and 16 subtests**, with one optional CUDA test skipped. They cover
both checkpoint key formats, missing-weight failures, strict-load enforcement,
normal training initialization, and exception/thread isolation of the shortcut.

After adding checkpoint coverage and readiness guards, the expanded focused
suite passed **127 tests and 16 subtests**, with one optional CUDA test skipped.
The released checkpoint loaded all **4,048/4,048** model tensors with no missing
or unexpected keys; complete model-state and seeded CPU prediction hashes still
matched the earlier normal-initialization baseline. The failure tests cover
missing/unexpected keys, shape mismatches, malformed checkpoints, copy errors,
failed reloads, model replacement and recovery. See the
[checkpoint validation report](../output/model_startup_20261107/checkpoint_validation.json).

Each runner now logs model construction/device-transfer time separately from
checkpoint loading. These timings exclude Python imports, featurization and
predictions. The existing batch log includes validation and output submission
but excludes per-input feature preparation; subtracting its sum from process
wall time does not isolate fixed startup costs.

The CLI already reuses a model across input JSON files:

```bash
mol pred -i /path/to/shard_job_jsons -o /path/to/output \
  --infer_setting.fast_init true
```

Supply the same inference options as the validated run. Keep each block in its
own JSON to preserve batching boundaries; merging block contents can change
padding and predictions. The input directory is searched recursively for JSON
files, so it should contain only job inputs with distinct job names. This does
not add resume support: retain the launcher's existing filtering of completed
designs. A directory run keeps the model loaded while each file gets its own
dataset and scheduler.

Reproduce construction timing, SciPy draw profiling, complete state hashes and
seeded synthetic prediction comparisons with the cached released checkpoint:

```bash
PYTHONPATH=. LAYERNORM_TYPE=torch MOL_ROOT_DIR=/path/to/mol_root \
  OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
  python scripts/benchmark_model_startup.py \
    --checkpoint /path/to/mol_root/checkpoint/mol-1-base.pt \
    --output output/model_startup/report.json --device cpu --threads 2 --repeats 2
```

The benchmark alternates normal/fast/fast/normal construction in one process,
so imports and cold filesystem caches are not controlled startup measurements.
It uses five synthetic tokens, one recycle, two diffusion steps and one sample
with FP32/PyTorch kernels to check equality, not prediction accuracy or
production GPU throughput.

### Concurrent startup and replay evidence

Claude's supplied construction results separately measured seven simultaneous
CPU constructions. With SciPy initialization skipped, the default 128 PyTorch
threads per process took **49.7–62.1 s**; four threads took **5.0–5.4 s**. These
numbers describe model construction, not complete CLI startup or GPU inference.
They establish a second source of overhead beyond the redundant initializer.
The raw timings were supplied in `mol1_startup_findings.json`; this concurrent
experiment was not rerun during the checkpoint-loader change.

Set `OMP_NUM_THREADS=4` in the shard launcher's environment before starting
Python. The runner now logs the effective PyTorch intra-op/inter-op thread
counts so overrides can be detected. It does not change process-wide thread
settings from inside the library. Thread caps and block size belong to the
external launcher.

The four supplied replay trees (`out_default`, `out_default2`, `out_omp4`,
`out_fastinit`) were compared independently using SHA-256. All **20 CIF and
summary JSON files per tree matched byte-for-byte**, with matching paths:
two designs, batch size two, five samples each. The logs record BF16,
10 recycles, 200 diffusion steps, TMK3 and cuEquivariance. This is evidence of
output equivalence on those inputs with the changed settings; it is not an
argument that the CLI flags were identical or a general accuracy benchmark.
See [replay comparison](../output/model_startup_20261107/replay_comparison.json)
for source paths and hashes.

For the active run discussed here, the agreed integration point is the next
planned restart: combine `fast_init=true`, the four-thread cap, and the external
`BLOCK=4000` change then. The active jobs and launcher were not changed, and
the flag defaulted off at that point. The current defaults are listed at the
top of this report; no external launcher or active process was restarted.

<a id="experimental-token-diffusion-compilation"></a>

## Token-diffusion compilation and replay

`mol.model.compiled_diffusion.compiled_diffusion_blocks` compiles the existing
token diffusion blocks with `torch.compile(fullgraph=True)`.
The prediction runner offers opt-in compilation and graph replay on CUDA,
with dynamic token lengths. Both are disabled by default. Direct context-manager use retains static shapes
and no replay unless `dynamic_shapes=True` and `cuda_graphs=True` are supplied.
Sampling, random-number generation,
input shapes, padding, precision and conditioning-cache preparation retain
their existing paths.

```python
from mol.model.compiled_diffusion import compiled_diffusion_blocks

# runner has already loaded its checkpoint and is in evaluation mode.
with compiled_diffusion_blocks(runner.model.diffusion_module.diffusion_transformer):
    predictions = runner.predict_batch(prepared_inputs, seeds)
```

The context compiles bound forward methods without replacing registered model
parameters or changing state-dictionary keys. It restores the original methods
on exit, including failures. Training and gradient-enabled calls bypass
compilation. One caller must own the model instance while the context is active;
do not reload weights, move the model or serve concurrent requests through it.
Compilation errors propagate instead of silently falling back to eager execution.

The original compilation experiment was validated with `LAYERNORM_TYPE=torch`
only. Its success did **not** establish support for the default fused LayerNorm:
the extension's direct Python/CUDA binding was not traceable with
`torch.compile(fullgraph=True)`. Switching a production baseline to PyTorch
LayerNorm changes a numerical implementation and requires a separate comparison.

The compilation context now temporarily exposes fused LayerNorm through a
registered inference custom operator. It executes the original CUDA kernel,
including its existing casts and epsilon, with a fake implementation describing
the output shape, dtype and contiguous layout to the compiler. Eager and
gradient-enabled calls keep their original forward; instance methods are
restored when the context exits. This follows PyTorch's
[custom operator approach](https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html)
for code that the compiler cannot trace. No LayerNorm backend is substituted.

Fused LayerNorm forward launches now use PyTorch's current CUDA stream. The
previous launches implicitly used the default stream, so side-stream graph
capture could omit the normalization and replay incorrect outputs. Only launch
stream selection changed; kernel arithmetic is unchanged. This source change
invalidates the extension build cache once for new processes. Already-running
processes keep their loaded extension. The benchmark can additionally load a
pre-fix binary with `--reference-fused-extension` for an original eager baseline,
separating the stream correction from compilation and replay numerics.

For CUDA graph replay, pass `cuda_graphs=True, max_graphs=2` to the context.
After compiling and warming up on a side stream, it captures the entire token
transformer stack in a `torch.cuda.CUDAGraph`. Per-block Inductor CUDA graphs
stay disabled; replay is explicit and does not use private Inductor tree APIs.
Each invocation copies current tensor inputs into owned buffers with matching
strides, including masks and cached pair projections. Returned outputs are
cloned outside capture so a later replay cannot overwrite retained results.
Random sampling and atom encoding/decoding remain outside the captured region.
This follows PyTorch's [CUDA graph input/output lifetime requirements](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs).

Graphs are cached by exact tensor signatures and execution settings, without
adding padding. `max_graphs` bounds retained graph pools; additional signatures
use compiled execution and increment a fallback counter. The session reports
capture, replay and fallback counts. Capture failures propagate. Graph objects
and buffers are released when the context exits. Use one caller stream with
exclusive ownership of the model instance while the context is active. During
initial capture, do not submit other GPU work concurrently from the same
process; GPU sharing with separate processes is permitted by the benchmark's
explicit sharing option.

`scripts/benchmark_diffusion_compile.py` defaults to a small synthetic CPU
fixture. To compare a cached checkpoint on CUDA, provide `--checkpoint`,
`--input-json` and `--device cuda`, along with explicit sampling and kernel
settings. The input file must contain one equal-length batch with one seed per
job. Alternatively, `--input-index` selects one job directly from an existing
example JSON and `--seed` selects a single seed without rewriting that file.
The report records the original file hash, job index/name, seed, selected
LayerNorm setting and actual fused/PyTorch module counts. The script compares
the same features and seeds, separates first calls
from alternating warmed passes, and records coordinates, PAE, ipSAE, rankings,
effective configuration, source/checkpoint hashes and peak GPU allocations.
The timed region includes prediction and input transfers, but excludes
featurization, output comparison and writing. Compilation is not assumed to
preserve exact floating-point results.

Add `--cuda-graphs` for a three-mode eager/compiled/graph comparison. It checks
that graph passes actually replay, reports new captures and replays per pass,
and compares graph outputs against both eager and compiled references. The
first pass per mode is separate; warmed passes run in forward and reverse mode
order. New captures should cease for repeated inputs of the same signature.

Set `CUDA_VISIBLE_DEVICES` to one full GPU UUID. A busy GPU is rejected unless
the caller explicitly passes `--allow-busy-gpu`; competing process IDs are
recorded before and after every pass. `--gpu-memory-fraction` defaults to 0.4
and caps this process's PyTorch CUDA allocator, not all driver/library memory.
Sharing an occupied GPU can slow other jobs and makes timings provisional.
Compiler and extension caches should use separate directories when running
alongside production. The benchmark never changes a launcher or restarts jobs.

### Shared-H200 validation, 2026-11-11

The released checkpoint was tested on two copies of real 9fm7 (644 tokens,
including both ligands), at seeds 107 and 109. This is one complex at two seeds,
not two different biological examples. Both modes used batch size two,
10 recycles, 200 diffusion steps, five samples, diffusion/confidence chunks of
five, full diffusion caching and atom caching. The environment was Torch
2.13.0+cu130 with PyTorch LayerNorm, TMK3/cuEquivariance, BF16 trunk and the
existing FP32 diffusion/confidence policy; MSA, templates and TF32 were off.

A separate Mol-1 production process occupied the same GPU throughout. After
one first pass per mode, warmed passes alternated eager/compiled/compiled/eager:

| Mode | Warmed seconds per two-input batch | Median seconds |
| --- | --- | ---: |
| Eager | 128.900, 121.873 | 125.387 |
| Compiled token diffusion | 125.569, 123.856 | 124.712 |

The observed median reduction is **0.54%**, smaller than the within-mode
variation. The last eager pass was faster than either compiled pass.
**This shared-GPU experiment does not establish a useful speedup.** A shorter
one-recycle/two-step smoke test showed an apparent 18.7% reduction, which did
not persist at full settings and should not be used as a production estimate.
The benchmark's input preparation, output comparisons and hashing are outside
these timings; these are not end-to-end CLI times.

Both modes repeated bitwise within their own execution path. Between eager and
compiled execution, the maximum coordinate-component difference was
**0.008817 Å**, with maximum per-input component RMSE **0.000184 Å**. The maximum
expected-PAE difference was **0.012766 Å**. All ten native ipSAE values matched
exactly; ranking-score changes were at most **3.28e-7**, with unchanged sample
ordering. These comparisons do not establish unchanged accuracy or exact
outputs on other inputs. Peak allocated GPU memory was approximately
**45.82 GiB** in both modes.

The focused CPU regression suite in the production Python environment passed
**38 tests**, with two optional CUDA cases skipped. The separate GPU smoke and
full-checkpoint comparisons completed successfully. Compilation remains an
explicit API experiment; production CLI defaults and launchers were unchanged.
The benchmark processes exited and released their GPU allocations afterward.

Raw artifacts: [full report](../output/diffusion_compile_20261111/gpu_full/report.json),
[summary](../output/diffusion_compile_20261111/summary.json),
[input batch](../output/diffusion_compile_20261111/inputs.json), and
[reproduction script](../output/diffusion_compile_20261111/run_full.sh).
The report includes checkpoint/source hashes, resolved settings, numerical
differences and competing GPU process IDs for every pass. The reproduction
script explicitly enables GPU sharing, bounds the allocator to 40% of device
memory and uses a 30-minute process timeout; select an appropriate GPU UUID and
a fresh output path before rerunning it.

### Shared-H200 CUDA replay validation, 2026-11-12

The three-mode comparison used the released checkpoint and the same 644-token
9fm7 complex at seeds 107 and 109: batch size two, 10 recycles, 200 diffusion
steps, five samples, chunks of five, full diffusion and atom caching, BF16
trunk and FP32 diffusion/confidence. Torch 2.13.0+cu130, PyTorch LayerNorm,
TMK3/cuEquivariance and disabled MSA/templates/TF32 matched the compilation
experiment. Checkpoint loading was strict, with all 4,048 tensors covered.

After one first pass per mode, warmed passes ran eager/compiled/graph, then
graph/compiled/eager. The first full-run passes took 207.163, 201.175 and
199.781 seconds respectively; compiler caches had already been populated by
the smoke run, so these are not cold compilation measurements.

| Mode | First warmed pass, seconds | Second warmed pass, seconds |
| --- | ---: | ---: |
| Eager | 199.169 | 135.622 |
| Compiled token diffusion | 195.079 | 132.561 |
| Compiled token diffusion with graph replay | 203.766 | 132.467 |

The GPU initially hosted both Mol-1 and OpenDDE alongside this benchmark.
OpenDDE left before the second graph pass; the same Mol-1 process remained
throughout. This change coincides with the large timing decrease between
columns. The second graph and compiled passes differed by only 0.095 seconds.
**This run validates replay correctness but does not establish an additional
speedup from CUDA graphs.** Medians in the raw report mix two occupancy states
and should not be treated as production throughput estimates. Prediction
timings exclude model loading, featurization, comparisons and output writing.

The full run captured **one graph**, performed **600 replays** over three
200-step predictions, and recorded **zero fallbacks**. Neither warmed graph
pass captured again. Graph predictions matched compiled predictions bitwise
for coordinates, raw confidence outputs, expected PAE, native ipSAE and
ranking scores. Every mode repeated bitwise within its own path. Compared
with eager execution, the maximum coordinate-component difference was
**0.008817 Å**, expected PAE differed by at most **0.012766 Å**, and ranking
scores by at most **3.28e-7**. All ten native ipSAE values and sample ordering
matched eager exactly. This remains one complex at two seeds, not evidence
of unchanged accuracy across other inputs.

Peak allocated memory increased from **45.82 GiB** before graph capture to
**48.47 GiB**, approximately **2.65 GiB** extra. Later eager/compiled passes
also retained the graph pool because all modes shared one active context;
their warmed memory figures are therefore not independent baselines.

The CPU regression suite passed **40 tests**, with three optional CUDA cases
skipped. A separately enabled CUDA test passed, covering changed masks,
conditioning and cached projections, retained outputs across later replays,
unchanged CUDA RNG state, bounded-cache fallback and method restoration.
The released-checkpoint smoke comparison also matched compiled outputs
exactly, with one capture and six replays. The CPU benchmark without graph
replay completed successfully, preserving the existing benchmark path.

All **118 recorded source hashes** matched the local files after the full
run. The benchmark exited successfully and released its GPU allocations;
the Mol-1 production process remained running. At the time of that experiment,
the API/benchmark path was opt-in and production defaults were unchanged.
The later default promotion is described at the top of this report.

Raw artifacts: [full report](../output/diffusion_graph_20261112/full/report.json),
[summary](../output/diffusion_graph_20261112/summary.json),
[smoke report](../output/diffusion_graph_20261112/smoke/report.json),
[CUDA test log](../output/diffusion_graph_20261112/checks.log), and
[reproduction script](../output/diffusion_graph_20261112/run_benchmark.sh).
The reproduction script accepts `smoke` or `full`, explicitly allows GPU
sharing, uses separate compiler/extension caches and caps the PyTorch allocator
at 40% of device memory. The full run has a 40-minute timeout. Select an
appropriate GPU UUID and fresh output paths before rerunning it.

### Fused LayerNorm and repository-example validation, 2026-11-12

The default fused LayerNorm path was tested using **8cnh-assembly1**, index zero
of the unchanged `examples/example_without_msa.json`, with its existing seed
**1**. Its protein and ligand produce **509 tokens**. This was a single-input
batch with the released checkpoint, 10 recycles, 200 diffusion steps, five
samples, chunks of five, BF16 trunk and FP32 diffusion/confidence. MSA,
templates and TF32 were off; TMK3/cuEquivariance and full diffusion/atom
caching were enabled. The full report records `fast_layernorm`, **764 fused
LayerNorm modules and zero PyTorch LayerNorm modules**. The compilation context
temporarily adapted 120 fused modules in the token transformer.

The comparison included an eager pass using the original, pre-stream-fix CUDA
binary, then corrected fused eager, compiled and compiled-plus-replay paths.
This separates three numerical comparisons:

| Comparison | Full-run result |
| --- | --- |
| Stream-corrected fused eager vs original fused eager | Bitwise identical coordinates, confidence outputs, expected PAE and ranking scores |
| Graph replay vs compiled execution | Bitwise identical for the same outputs |
| Compiled execution vs original fused eager | Different: maximum coordinate-component difference **2.406797 Å**, component RMSE **0.052428 Å** |

The maximum expected-PAE difference from compilation was **1.069635 Å** and
the maximum ranking-score difference was **0.00031352**; sample ordering stayed
the same. This single-protein/ligand example has no protein-protein ipSAE, so
it provides no ipSAE-equivalence evidence. Each execution mode repeated bitwise
within its own path. The run used **one capture and 600 replays**, with zero
capacity fallbacks and no additional warmed captures.

**Retaining fused LayerNorm removes the backend-switch confound; it does not
make compilation preserve eager predictions.** The 2.407 Å local coordinate
change is material even though aggregate RMSE is smaller. This experiment does
not establish unchanged accuracy. Compilation/replay were briefly promoted to
defaults on 2026-11-12, then returned to opt-in on 2026-11-13 because a production
speedup and compatibility with existing score pools were not established.

| Mode | Warmed seconds per input | Median seconds |
| --- | --- | ---: |
| Corrected fused eager | 57.918, 59.727 | 58.822 |
| Compiled token diffusion, fused LayerNorm | 61.167, 62.267 | 61.717 |
| Compiled token diffusion with replay, fused LayerNorm | 53.306, 57.005 | 55.156 |

Two other compute processes shared the H200 throughout. These are provisional
prediction timings, excluding loading, featurization, comparison and output
writing; they do not establish an isolated production speedup. Peak allocated
memory before capture was **28.47 GiB**, rising to **29.33 GiB** with the graph
pool retained, approximately **0.85 GiB** extra.

Validation passed **41 CPU tests** (11 optional CUDA cases skipped) and
**nine explicitly enabled CUDA tests**. The CUDA tests cover all four affine
configurations in FP32/BF16, noncontiguous inputs, custom-operator metadata,
raw fused-kernel side-stream capture, refreshed inputs and conditioning,
retained outputs, RNG preservation, cache capacity and method restoration.
The first replay test exposed the default-stream launch issue; it passed after
the stream correction without relaxing the exact-equality checks.

The short checkpoint run also succeeded. Its environment-label field reflects
a test-helper import that set `LAYERNORM_TYPE=torch` after the module factory
had already selected fused LayerNorm; its actual module counts were 764 fused
and zero PyTorch modules. Checkpoint benchmarks now avoid that test-helper
import. The full run's environment label and actual module counts agree, and
all **123 Python/CUDA/C++/header source hashes** matched afterward. The benchmark
exited and released its GPU memory; the same production Mol-1 process remained
running. Existing launchers were not edited.

Artifacts: [full report](../output/diffusion_fused_compile_20261112/full/report.json),
[summary](../output/diffusion_fused_compile_20261112/summary.json),
[CUDA test log](../output/diffusion_fused_compile_20261112/tests.log), and
[reproduction script](../output/diffusion_fused_compile_20261112/run.sh).
The original fused extension binary is preserved under the artifact directory's
`reference_extension/`; its SHA-256 matches the baseline recorded in the report.
Use a fresh process to load the corrected extension, select a suitable GPU UUID
and fresh output paths, then run the reproduction script with `tests`, `smoke`
or `full`. The script leaves `LAYERNORM_TYPE` unset, bounds the CUDA allocator
to 40% for checkpoint runs and uses separate build/compiler caches.
