# Independent complexes in one batch

`mol pred --batch_size B --use_msa false` predicts up to B independent
complexes together on each GPU. Their token counts, atom counts and chain
counts may differ. Put the complexes in the **same JSON job list**, with a
unique `name` for each. A directory's JSON files are processed separately.

```bash
mol pred -i examples/no_msa_batch/inputs.json -o ./output/batched \
  --use_msa false --batch_size 2 --sample 1
```

Here the first forward contains two complexes and the final forward contains
one. Each input gets one predicted structure. `--sample 5` instead generates
five structures **per input**, retaining all five and ranking them within
that input's seed. `--batch_size` and `--sample` are separate axes.

Outputs retain the usual layout:
`output/<job_name>/seed_<seed>/predictions/`. Each CIF, confidence summary and
optional `--need_atom_confidence true` full-data JSON belongs to its own input.
The default `--batch_size 1` keeps the existing singleton inference path.

## Execution and supported settings

The PairMixer/Pairformer trunk and diffusion token transformer run batched.
Their token axes are padded and masked. Atom encoding/decoding, reference
geometry, chain mappings and confidence calculations run per complex at the
original dimensions. No-MSA mode retains the existing query-only features.

Batching requires:

- `--use_msa false`, `--use_template false`, and `--use_rna_msa false`.
- Triangle multiplication: `auto`, `torch`, `cuequivariance`, or
  `flashpairformer`; triangle attention: `auto`, `torch`, or `cuequivariance`.
  On CUDA, `auto` uses cuEquivariance when installed. TMK3 requires an explicit
  `--trimul_kernel flashpairformer`; its per-input kernels reuse packed weights
  inside the batch. `triattention` is not supported for independent batches.
- CPU or CUDA, `num_workers=0`, no Monte Carlo dropout, and
  `model.N_model_seed=1`. The CLI's defaults meet these conditions except MSA.

All three diffusion cache modes (`full`, `shared`, `off`), efficient fusion,
and the existing chunk controls remain available. The sample chunk size
controls how many diffusion samples **per input** are denoised together.

For five samples and ten recycles, a configuration to benchmark on an H200 is:

```bash
conda activate mol1
mol pred -i inputs.json -o output/batched_fast \
  --use_msa false --batch_size 2 --sample 5 --cycle 10 --step 200 \
  --trimul_kernel cuequivariance --triatt_kernel cuequivariance \
  --infer_setting.sample_diffusion_chunk_size 5 \
  --infer_setting.confidence_sample_chunk_size 5 \
  --infer_setting.batch_max_padding_ratio 1.4
```

This still produces five structures per input. Sample chunks change the number
processed concurrently, not the number of samples, recycles or diffusion steps.
Diffusion chunk size changes RNG draw ordering, so changing it does not preserve
the exact coordinates for a seed. For comparisons of implementation numerics,
hold the chunk size and kernels fixed. Both chunk defaults remain one to limit
memory use. Confidence chunks automatically stay at one above 2,000 tokens,
preserving the existing CPU offload policy for large confidence matrices.

The batched implementation reuses trunk padding buffers and masks between
recycles. In `full` and `shared` cache modes, it also reuses padded diffusion
pairs, single conditioning and fully conditioned atom features between steps.
`full` folds the padding mask into the cached per-block attention biases once.
CUDA allocator blocks are retained after successful batches; OOM retries still
release them. These changes do not alter the precision policy or TF32 setting.

Jobs are sorted by approximate sequence length, then grouped without mixing
the token-count thresholds that select different precision policies. A final
partial batch is processed normally. CUDA forward OOMs retry smaller batches;
a singleton failure or output-writing failure is reported and causes a
nonzero exit after the other inputs finish.

Two optional limits use **actual featurized token counts** to split batches:

- `--infer_setting.batch_max_padding_ratio R` bounds
  `B * max_tokens**3 / sum(tokens_i**3)`, an estimate of wasted triangle work.
- `--infer_setting.batch_max_pair_tokens Q` bounds `B * max_tokens**2`, a proxy
  for pair-state memory. It is not a byte or VRAM guarantee.

Both default to zero (disabled). A singleton always runs, even if it exceeds
the pair budget. Neither limit changes an input's samples or seed ownership.

## Seeds and multiple GPUs

With `--use_seeds_in_json true`, each job's `modelSeeds` is respected. Otherwise
`--seeds` applies to every job. Each input has its own diffusion random stream;
changing batch membership or retrying an OOM does not consume another input's
random numbers. The batched path also seeds featurization per job.

Legacy singleton inference uses its historical global RNG stream. Identical
CLI seeds therefore do not imply identical coordinates between the legacy and
batched paths. Shape-dependent floating-point differences also occur, especially
in BF16. Numerical comparisons should use FP32 and controlled per-input seeds.

For multiple GPUs, launch one model process per GPU:

```bash
torchrun --standalone --nproc_per_node 2 -m mol.runner.inference \
  --input_json_path examples/no_msa_batch/inputs.json \
  --dump_dir ./output/batched_2gpu --use_msa false --batch_size 2 \
  --triangle_multiplicative torch --triangle_attention torch
```

Ranks own disjoint jobs, including all seeds of each job. No jobs are duplicated
to pad a rank's workload. Each rank batches its inputs locally.

To have multiple GPUs cooperate on a single complex, use
[context parallel inference](context_parallel_inference.md) instead. It requires
`--batch_size 1` and a matching `--context_parallel_size` and torchrun worker count.

## Memory and validation

Pair-state memory grows roughly with `B * max_tokens**2`; diffusion also scales
with the sample chunk size. Group similarly sized inputs and start with B=2.
Measure completed inputs/second and peak allocated GPU memory while holding
cycles, diffusion steps, sample count, dtype and kernels fixed. Batching may
increase or decrease throughput depending on padding and the workload.

The CPU tests exercise mixed-width PairMixer/Pairformer blocks, mixed lengths,
nonzero attention projections, sample ownership, confidence and atom mappings,
cache/fusion modes, independent RNG streams, partial batches, rank ownership,
and OOM/error handling:

```bash
LAYERNORM_TYPE=torch pytest tests/test_input_batching.py tests/test_batched_prediction.py
```

These tests use small randomly initialized models. The example inputs are
synthetic execution fixtures, not a prediction-accuracy benchmark.

With a cached released checkpoint, the opt-in CUDA test also compares the real
model against explicitly seeded singleton forwards and checks the saved CIFs,
confidence dimensions and rankings. Select an available GPU and run:

```bash
MOL_RUN_BATCH_SMOKE=1 LAYERNORM_TYPE=torch pytest \
  tests/test_input_batching_checkpoint.py -q -s
```

This test uses two recycles and eight diffusion steps to check execution and
numerical consistency; production prediction quality needs the normal sampling
settings and representative inputs.

Local validation on 2026-10-28 used the released `mol-1-base` checkpoint on an
H200 with the 32-, 64- and 96-residue fixtures, batch size two and two samples
per input. The CLI wrote six CIFs with the correct atom counts, confidence
dimensions and per-input rankings. The deterministic FP32 regression repeated
a singleton prediction bitwise identically; batched versus singleton coordinates
differed by at most 0.00121 angstrom, with coordinate RMSE below 0.000204 angstrom.
The BF16 batch also produced finite outputs with the expected dimensions.
The non-network CPU suite passed 213 tests (19 skipped), and the opt-in checkpoint
test passed separately. These short runs validate execution, not prediction
accuracy or a throughput improvement.

## Reproducible throughput measurements

`scripts/benchmark_input_batch.py` uses synthetic protein dimers of different
lengths with the released checkpoint, five samples, ten recycles, 200 diffusion
steps, and the CLI's precision policy. It writes a JSON report including warmup,
repeat timings, peak allocated memory, environment and source hashes. Timings
include input transfer, inference and confidence summaries; they exclude model
loading, featurization, output validation and disk writing.

```bash
conda activate mol1
MOL_ROOT_DIR=/storage/frank/mol1_root LAYERNORM_TYPE=torch \
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
  python scripts/benchmark_input_batch.py \
  --checkpoint /storage/frank/mol1_root/checkpoint/mol-1-base.pt \
  --output output/batch_benchmark.json --lengths 512 640 \
  --kernel cuequivariance --attention cuequivariance \
  --samples-per-call 5 --confidence-chunk 5
```

The script requires cached assets and never fetches a checkpoint. Synthetic
timing fixtures do not establish prediction accuracy or production throughput
for a different length/atom-count distribution.

Measured on 2026-11-02 in `conda activate mol1` (Torch 2.13.0+cu130), on H200s,
using PyTorch LayerNorm and TF32 disabled. Each row is one batch of two complexes
producing ten structures total, with five samples, ten recycles and 200 diffusion
steps. The trunk used BF16; diffusion and confidence used FP32. Times are medians
of two repetitions after one warmup:

| Token lengths | Implementation | Diffusion / confidence chunk | Seconds/batch | Peak allocated GiB |
| --- | --- | --- | ---: | ---: |
| 128, 160 | Original, torch kernels | 1 / 1 | 43.37 | 3.71 |
| 128, 160 | Optimized, torch kernels | 1 / 1 | 41.02 | 3.71 |
| 128, 160 | Original, torch kernels | 5 / 1 | 13.00 | 3.71 |
| 128, 160 | Optimized, cuEquivariance | 5 / 5 | 11.82 | 5.33 |
| 512, 640 | Original, torch kernels | 5 / 1 | 128.30 | 36.16 |
| 512, 640 | Optimized, cuEquivariance | 5 / 5 | 68.76 | 45.46 |
| 512, 640 | Optimized, TMK3 + cuEquivariance attention | 5 / 5 | 62.33 | 45.71 |

The larger case is **2.06x faster** than the original code with diffusion sample
chunking already enabled. TMK3 adds about 10% throughput over optimized
cuEquivariance there. At 128/160 tokens the two fused backends are effectively
tied. The much larger gain from 43.37 to 11.82 seconds on small inputs mostly
comes from the pre-existing diffusion sample chunk control; it should not all
be attributed to the new code. The faster configuration uses more memory.

The local aggregate report is `output/batch_optimization_report.json`, with
individual JSON reports and source hashes beside it. CPU featurization and disk
output remain synchronous and are excluded from these timings.

Validation included 244 CPU tests (three pre-existing pandas 3.1 vectorization
failures deselected) and 30 GPU tests with 80 TMK3 subtests. A controlled FP32
comparison against the saved original batching code produced bitwise-identical
trunk, coordinate, PAE and pLDDT outputs using the same preprocessed inputs.
A second controlled comparison with a BF16 trunk, FP32 diffusion/confidence,
five samples, ten recycles and 200 steps was also bitwise identical. These
comparisons held torch kernels and diffusion chunk size fixed and enabled
deterministic algorithms; changing kernels or sample chunks has separate
floating-point/RNG effects.
The unchanged code in this environment already differs from singleton inference
by 0.00356 angstrom RMS / 0.0171 angstrom maximum on the shortest checkpoint
fixture; the checkpoint regression accounts for that measured baseline and also
checks exact repeated-batch reproducibility.

## FP32 attention layout validation on real inputs

FP32 CUDA inference now exposes four-dimensional batch/head axes to PyTorch
scaled-dot-product attention. This lets the mixed-input diffusion transformer
use efficient attention instead of the math backend. Biases shared across the
five samples retain their original storage: the helper processes each input
separately when flattening would require materializing repeated pair biases.
CPU, gradient-enabled execution and BF16 autocast retain the existing path.
PyTorch still chooses the supported attention backend for each call.

Validation on 2026-11-03 used the `mol1` environment (Torch 2.13.0+cu130), H200s,
the released checkpoint, TMK3 multiplication, cuEquivariance triangle attention,
PyTorch LayerNorm and TF32 disabled. Both implementations used **five samples,
ten recycles and 200 diffusion steps**, with BF16 trunk and FP32 diffusion and
confidence. The real 9fm7 fixture retained both protein chains and both ligands
(644 tokens); the mixed-length batch added the real 8cnh-assembly1 fixture
(509 tokens). MSA and templates were disabled.

Each comparison used the same loaded model and prepared inputs, warmed both
implementations, then measured baseline/optimized/optimized/baseline. These are
medians of two timed runs per implementation, including input copying, transfer,
inference and confidence summaries. Featurization, native ipSAE calculation,
output validation and writing are outside the timed region.

| Real input | Deterministic algorithms | Baseline seconds | 4D attention seconds | Time reduction |
| --- | --- | ---: | ---: | ---: |
| 9fm7 alone | Enabled | 35.44 | 35.37 | 0.2% (no meaningful gain) |
| 9fm7 + 8cnh | Enabled | 68.76 | 66.72 | 3.0% |
| 9fm7 + 8cnh | Disabled | 65.17 | 63.03 | 3.3% |

Peak allocated GPU memory was unchanged between timed implementations:
43.55 GiB for the singleton and 44.46 GiB for the batch. A singleton's token
attention already has four dimensions, so the main improvement is specific to
mixed-input batching. This experiment does not establish a speedup for other
lengths, devices or a complete production queue.

In the controlled runs, each implementation repeated bitwise identically.
Comparing old versus new attention, all five 9fm7 `ipsae_min_10` values and sample
rankings matched exactly in both singleton and batched execution. Batched 9fm7
coordinate RMSE was 0.000063 angstrom, maximum component difference 0.00173
angstrom, and maximum CA-aligned RMSD across samples 0.000099 angstrom. PAE
maximum difference was 0.00787 angstrom. Coordinates and PAE are therefore **not
bitwise identical**. The companion 8cnh case had coordinate RMSE 0.00261 angstrom,
maximum component difference 0.422 angstrom and maximum CA-aligned RMSD 0.00515
angstrom; rankings matched. Its ipSAE is null because it has one protein chain.
Before/after CIFs, summary JSONs and full-precision comparison tensors are saved.

The ordinary-mode timing run is unsuitable for claiming numerical equivalence:
repeating the unchanged baseline already changed 9fm7 ipSAE by up to 0.02581 and
changed sample rankings. The optimized implementation also varied across
repeats. The cause of that existing nondeterminism was not isolated in this
change. Exact ipSAE agreement above is limited to the controlled real-example
test, not a guarantee for arbitrary predictions or ordinary-mode repeated runs.

Reproduce the comparison using a saved baseline helper from revision `8c3c5e5`:

```bash
git show 8c3c5e5:mol/model/modules/primitives.py > /tmp/mol1_attention_baseline.py
conda activate mol1
MOL_ROOT_DIR=/storage/frank/mol1_root LAYERNORM_TYPE=torch \
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python scripts/validate_attention_9fm7.py \
  --case batch --deterministic \
  --baseline-source /tmp/mol1_attention_baseline.py \
  --checkpoint /storage/frank/mol1_root/checkpoint/mol-1-base.pt \
  --output output/attention_9fm7
```

Use `--case single` for 9fm7 alone; omit `--deterministic` for ordinary settings.
The validation script fixes samples/cycles/steps at 5/10/200 and records active
model precision settings. No MSA or template search is performed.

The aggregate local report is `output/mol1_attention_9fm7_report.json`, with
individual reports, original logs, prediction artifacts and corrections in
`output/mol1_attention_9fm7_validation/`. The original measurement driver's final
console print failed to serialize tensors after saving all completed results.
The maintained script fixes that print, uses boolean CA masks for alignment and
reports the model's active precision configuration. Corrected alignment metrics
were recomputed from the saved tensors; timings and native ipSAE were unaffected.

Validation preserved all previously passing tests. The full non-network CPU
suite retains three existing pandas vectorization failures (`test_all_nan`,
`test_basic`, `test_with_nan`), reproduced on the unmodified revision. GPU
regressions passed 93 tests (19 opt-in tests skipped); separately enabling the
batch/TMK3 kernel tests passed 30 tests and 80 subtests. Run the suite explicitly
under `tests/` so local archived experiments in `output/` are not collected:

```bash
LAYERNORM_TYPE=torch PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest tests -m 'not network' -q
MOL_RUN_BATCH_KERNEL_TESTS=1 TEST_FLASHPAIRFORMER_CUDA=1 LAYERNORM_TYPE=torch \
  python -m pytest tests/test_batch_optimizations.py tests/test_flashpairformer.py -q
```

## Stable atom aggregation during CUDA inference

The ordinary-mode variability reported above was traced to atom-to-token
averaging. On real 9fm7 features the input atom embeddings were bitwise identical
across calls, but CUDA `scatter_add_` changed the resulting BF16 averages by up
to 0.005859375. Fixed RNG seeds do not fix floating-point atomic addition order;
these differences can grow through ten recycles of the BF16 trunk.

CUDA inference now groups atoms by token using a stable sort and performs a
segmented sum followed by division in the existing dtype. This gives each token
a fixed atom order without changing global PyTorch determinism settings. The
change covers ordinary no-gradient sum/mean aggregation with a shared 1D atom
map. CPU execution, gradients, generic batched maps and explicitly enabled
deterministic algorithms retain their existing paths. Missing tokens remain
zero. No model precision or sampling parameters were changed.

Both standalone 9fm7 and the mixed 9fm7/8cnh batch, using five samples, ten
recycles and 200 steps, repeated coordinates, PAE, atom pLDDT, native ipSAE and
rankings bitwise exactly with global deterministic algorithms disabled. Those
saved tensors also matched the earlier deterministic 4D-attention references
exactly (8cnh has no protein-protein ipSAE). The unchanged ordinary
baseline still varied across repeats. This fix can therefore change historical
ordinary-mode predictions; it does not promise to reproduce a particular
nondeterministic run or establish prediction accuracy.

Reproduce this comparison by substituting the aggregation helper:

```bash
git show 5059413:mol/model/utils.py > /tmp/mol1_aggregation_baseline.py
conda activate mol1
MOL_ROOT_DIR=/storage/frank/mol1_root LAYERNORM_TYPE=torch \
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python scripts/validate_attention_9fm7.py \
  --case batch --comparison aggregation \
  --baseline-source /tmp/mol1_aggregation_baseline.py \
  --checkpoint /storage/frank/mol1_root/checkpoint/mol-1-base.pt \
  --output output/aggregation_validation
```

Local artifacts are in `output/mol1_repro_validation/`. GPU sharing began after
the devices were checked and before the model runs finished. The timings are
retained as shared-GPU observations, not evidence of a speedup. Correctness
comparisons use the saved full-precision tensors and native summary values.

## Reusing fixed atom-block conditioning

An opt-in cache prepares the encoder and decoder atom blocks' fixed
pair-bias projections, adaptive LayerNorm affine terms and output gates once
per input during mixed-input diffusion. Each denoising step still computes attention on the changing atom
representations. The cache is local to one prediction request, shared across
sample chunks, and never registered on the model or checkpoint. It is enabled
with `--infer_setting.atom_block_cache true` and the existing `full` or `shared`
diffusion-cache modes; its default is **false**, and `off` retains the
uncached path. Gradient-enabled execution and training ignore the cache.

This change currently applies to `predict_batch`, including a partial batch of
one input. The ordinary singleton prediction path is unchanged. Fixed local
attention masks are still rebuilt; this optimization caches projections and
conditioning, not those masks.

Compare cached and uncached execution while keeping the same attention helper,
stable atom aggregation, checkpoint and precision settings:

```bash
conda activate mol1
MOL_ROOT_DIR=/storage/frank/mol1_root LAYERNORM_TYPE=torch \
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python scripts/validate_attention_9fm7.py \
  --case batch --comparison atom-cache \
  --checkpoint /storage/frank/mol1_root/checkpoint/mol-1-base.pt \
  --output output/atom_cache_validation
```

Add `--deterministic` for a second comparison using the existing explicitly
deterministic aggregation path. The baseline disables only atom-block cache
construction; the same loaded model performs both implementations.

The validation command enables the cache for the optimized passes automatically.
Local results are in `output/mol1_atomcache_validation/`. CPU/CUDA unit checks
cover exact outputs, immutable cached tensors, different sample chunks,
request-local construction and gradients. The combined non-network CPU suite
passed 282 tests with the same three pandas failures (63 skipped, 27 subtests).
CUDA regression checks passed 64 tests and 80 subtests (one skipped); a separate
attention/aggregation/cache group passed 73 tests (13 skipped). These groups
overlap and should not be summed.
Final opt-in/default, aggregation and cache checks passed 44 CUDA/CLI tests and
three subtests after the config switch was added.

An interleaved component microbenchmark with five samples and 4372 atoms measured
10.76 ms uncached versus 5.70 ms cached for the three-block atom transformer.
It used synthetic tensors and excluded cache construction. All GPUs were shared
with unrelated jobs, including during the real-model measurements, so these
results do not establish a production throughput gain.

In the real 9fm7/8cnh batch, all cached-versus-uncached coordinates, PAE, atom
pLDDT, native ipSAE values and sample rankings were bitwise identical, with
both ordinary and deterministic algorithms. All 40 before/after CIF and summary
file pairs matched byte for byte. Peak allocated memory was 44.46 GiB in both
implementations. Shared-GPU full-model timings were:

| Settings | Uncached seconds | Cached seconds |
| --- | ---: | ---: |
| Ordinary | 95.63 | 96.99 |
| Deterministic | 100.38 | 102.33 |

These measurements showed no full-model gain. Caching therefore remains opt-in
pending a measurement without competing GPU work; the component improvement
alone is insufficient to enable it by default.

The instrumented real-input trunk pass identified triangle attention as the
largest measured substage: 20.65 s, versus 13.50 s for triangle multiplication,
7.37 s for pair transitions and 4.97 s for the query-only MSA module. Total
instrumented trunk time was 51.47 s. These shared-GPU measurements are for
prioritization, not an isolated performance claim. MSA/template computation
semantics, confidence precision, five samples and ten recycles are preserved.
Input prefetch and background output writing remain future work.
