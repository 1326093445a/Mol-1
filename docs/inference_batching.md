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
- PyTorch triangle kernels: `auto` selects `torch` for input batches; explicit
  `cuequivariance`, `triattention` and `flashpairformer` are rejected.
- CPU or CUDA, `num_workers=0`, no Monte Carlo dropout, and
  `model.N_model_seed=1`. The CLI's defaults meet these conditions except MSA.

All three diffusion cache modes (`full`, `shared`, `off`), efficient fusion,
and the existing chunk controls remain available. The sample chunk size
controls how many diffusion samples **per input** are denoised together.

Jobs are sorted by approximate sequence length, then grouped without mixing
the token-count thresholds that select different precision policies. A final
partial batch is processed normally. CUDA forward OOMs retry smaller batches;
a singleton failure or output-writing failure is reported and causes a
nonzero exit after the other inputs finish.

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
