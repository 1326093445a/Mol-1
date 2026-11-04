# Mol-1 inference optimization: final report

Updated 2026-11-04. Covers the local implementation through `9bd411b` on
`fix/mol1-padding-diagnosis`; the changes have not been pushed or merged.

**Current measured speed on one H200:** the real 9fm7 + 8cnh batch takes
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
| Atom-block cache (`2ed8457`) | Computes fixed encoder/decoder pair biases, adaptive LayerNorm terms and gates once per request. Opt-in, request-local, and available in the batched path with `full` or `shared` diffusion caching. |
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
exactly at fixed grouping. Fewer or more than two protein chains yield a null
top-level value; pair records remain available.

## Configuration to use as a starting point

For this varied-length workload on an H200, use the batched scheduler with
minimal unequal-length padding, five-sample chunks, explicit TMK3/cuEquivariance
and the opt-in atom cache:

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
