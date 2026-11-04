# Changelog

## Unreleased

### Prediction compatibility

- **Default CUDA inference behavior changes.** Atom-to-token sum/mean aggregation
  now uses a fixed atom order for ordinary no-gradient CUDA inference with a
  shared 1D atom map. This removes the unordered atomic additions that caused
  repeated seeded predictions to vary in the tested cases. With the same
  checkpoint, inputs and seeds, coordinates, confidence scores, ipSAE and sample
  rankings can differ from historical ordinary-mode predictions. CPU execution,
  gradient-enabled execution and explicitly enabled deterministic algorithms
  retain their existing aggregation paths (`dc12f1c`).
- **The 4D FP32 attention layout can change individual atom coordinates.** In the
  controlled 9fm7/8cnh batch, 8cnh had a maximum coordinate-component difference
  of 0.422 Å and a maximum per-atom displacement of 0.456 Å, despite a coordinate
  RMSE of 0.00261 Å. Native 9fm7 ipSAE values and sample ordering matched exactly;
  other confidence values changed slightly, and 8cnh has no protein-protein
  ipSAE. These results do not establish negligible changes at every atom or
  unchanged prediction accuracy (`5059413`).

The reproducibility fix matched the earlier deterministic references bitwise
for the tested 9fm7 singleton and 9fm7/8cnh batch. This is not a guarantee across
all inputs, hardware or software versions. For comparisons with old predictions,
retain the code revision, environment, kernels, precision, batching and sampling
settings; re-run both sides under the same configuration when possible. See
[inference validation and compatibility](docs/inference_batching.md#stable-atom-aggregation-during-cuda-inference)
for the measured results and reproduction commands.

### Added

- Request-local atom-block conditioning caches for input batching, available
  through `--infer_setting.atom_block_cache true`. Caching remains **off by
  default**. A subsequent uncontended H200 comparison on real 9fm7/8cnh measured
  63.36 s uncached versus 61.13 s cached (3.5% less model time), with bitwise
  identical coordinates, confidence and native ipSAE. This small-cohort result
  supersedes the earlier inconclusive shared-GPU timing (`2ed8457`).
- Opt-in bounded CPU preparation and output writing for independent input
  batches: `--infer_setting.prefetch_inputs 2` and
  `--infer_setting.background_write true`. Preparation uses one spawned CPU
  worker per rank and preserves each job's post-featurization RNG state.
  Background writing owns CPU copies, drains on exit and propagates failures.
  Both controls default off and require `--batch_size` greater than one.
  Four real examples reproduced raw predictions and all CIF/summary files
  exactly at fixed grouping. The short-queue measurement showed no gain;
  later passes shared the GPU with another process. See the
  [targeted measurements](docs/inference_batching.md#targeted-h200-pass-2026-11-04).
