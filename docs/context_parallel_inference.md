# One prediction across multiple GPUs

`--context_parallel_size P` makes P workers cooperate on **each individual
complex**. All workers visit the same jobs and seeds. Rank zero featurizes each
job, broadcasts its features, and writes its structures and confidence files.

With the environment activated and the checkpoint and common assets already
cached under `MOL_ROOT_DIR`, select two available GPUs and run:

```bash
CUDA_VISIBLE_DEVICES=0,1 LAYERNORM_TYPE=torch \
torchrun --standalone --nproc_per_node 2 -m mol.runner.batch_inference \
  -i examples/no_msa_batch/inputs.json -o output/context_parallel \
  --use_msa false --context_parallel_size 2 --sample 1
```

The Python module directly exposes the prediction command, so there is no
`pred` subcommand in this invocation. `--sample` still means structures per
input. Increase both `--nproc_per_node` and `--context_parallel_size` together
to use more workers. The output layout is unchanged:
`output/<job_name>/seed_<seed>/predictions/`.

## What is distributed

Within the PairMixer/Pairformer and confidence stacks, each GPU owns a slab
of pair rows and the corresponding single-token features. Triangle
multiplication exchanges projected slabs; triangle attention exchanges its
biases and redistributes rows for ending-node attention. Width adapters preserve
Mol-1's 448-to-320-to-448 channel changes.

Within the diffusion token transformer, workers own different query tokens and
exchange the keys and values' input representations. Every worker contributes
to the same denoised structure, including when only one sample is requested.
Token counts need not divide evenly by the worker count.

This is **stack-level context parallelism**, with full tensors gathered at
stack boundaries. Weights, initial/recycling pair states, query-only MSA work,
atom encoding/decoding, diffusion caches, and final confidence outputs remain
replicated. It does **not** provide OpenDDE's end-to-end Fold-CP memory scaling
or combine GPU memory into a single pool. Those replicated allocations and
gather buffers can still determine whether a large complex fits. Communication
and synchronized error checks can make small predictions slower; measure your
actual workload before assuming a speed or memory improvement.

## Supported settings

- `--use_msa false`, `--use_template false`, `--use_rna_msa false`.
- `--batch_size 1`; independent-input batching and context parallelism are
  separate modes and cannot currently be combined.
- Torch triangle kernels. The CLI's `auto` selects torch in this mode; explicit
  alternative kernels are rejected.
- FP32 or BF16, inference only, no Monte Carlo dropout, `model.N_model_seed=1`,
  and `num_workers=0`.
- CUDA workers use NCCL. Use `--device cuda` or `auto` with `CUDA_VISIBLE_DEVICES`,
  rather than giving every worker the same explicit `cuda:N` device.
- All diffusion cache modes and efficient fusion are supported. Existing
  triangle and transition chunk controls remain available.
- The checkpoint must already exist before launching workers, avoiding
  concurrent downloads to the same file. CPU/Gloo is also available for tests.

The raw config runner accepts the same setting. Its triangle kernel defaults
must be overridden explicitly:

```bash
torchrun --standalone --nproc_per_node 2 -m mol.runner.inference \
  --input_json_path examples/no_msa_batch/inputs.json \
  --dump_dir output/context_parallel --use_msa false \
  --context_parallel_size 2 \
  --triangle_multiplicative torch --triangle_attention torch
```

Each input uses an explicit diffusion RNG seed on every rank. Legacy singleton
CLI inference consumes its historical global random stream, so matching CLI
seed numbers alone do not establish matching coordinates. For numerical
comparisons, use controlled per-input features and `inference_seed`, FP32,
deterministic execution, and identical sampling settings. Partitioning also
changes floating-point reduction order.

Rank-local compute errors are propagated before the next exchange. Input and
output errors on rank zero are broadcast. A distributed failure stops the run;
workers do not independently skip inputs or retry different execution paths.

If NCCL reports `Failed to bind NVLink SHARP (NVLS) Multicast memory` at startup,
retry with `NCCL_NVLS_ENABLE=0` in the launch environment. This machine required
that workaround; it is not forced globally by Mol-1. Communicator startup is
checked before model weights are loaded.

## Validation

The CPU tests use real two- and three-worker collectives, mixed trunk widths,
uneven token partitions, empty partitions, activated residual projections,
full-model sampling and confidence, cache/fusion modes, per-job seeds,
rank-zero-only writing, and injected failures:

```bash
LAYERNORM_TYPE=torch OMP_NUM_THREADS=1 pytest tests/test_context_parallel.py -q
```

The opt-in CUDA test loads the cached released checkpoint. It compares a shared
prediction to explicitly seeded singleton execution, checks FP32 and BF16,
verifies identical coordinates across ranks, and runs the shared-job scheduler
to validate CIF atom counts, confidence dimensions and rankings:

```bash
MOL_RUN_CP_SMOKE=1 LAYERNORM_TYPE=torch \
torchrun --standalone --nproc_per_node 2 -m pytest \
  tests/test_context_parallel_checkpoint.py -q -s
```

It uses two recycles and eight diffusion steps. This checks execution and
numerical consistency, not production prediction quality or scaling performance.

Local validation on 2026-10-28 passed the non-network CPU suite (221 tests,
16 skipped) and the released-checkpoint regression on two H200 GPUs. The
64-residue FP32 comparison had maximum coordinate error 0.00341 angstrom and
coordinate RMSE 0.000349 angstrom against controlled singleton execution.
Both workers returned bitwise identical coordinates. BF16 produced finite
outputs with the expected dimensions. The shared scheduler wrote six CIFs
for the 32-, 64- and 96-residue fixtures, with correct atom counts, confidence
dimensions, and rankings.

A separate public CLI run with `--sample 1 --cycle 2 --step 8 --dtype bf16`
also passed on two H200s. It produced three CIFs, one per fixture, while each
worker processed 16, 32 and 48 query tokens respectively for the 32-, 64- and
96-residue inputs. This confirms that one sample's computation is shared across
GPUs. Peak allocated memory was approximately 2.65–2.80 GiB per worker for these
small fixtures; these measurements do not establish savings for large inputs.
