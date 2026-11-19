# Changelog

## Unreleased

### Prediction compatibility

- **Experimental fixed-slot batch refill (off by default).**
  `--infer_setting.batch_refill true`, together with an explicit first-sample
  ipSAE cutoff, replaces stopped or completed jobs between diffusion samples.
  Survivors retain their slots and RNG streams. Trunks and diffusion retain the
  configured batch size; only identical token counts share a session, and tail
  placeholders emit no output. Memory use can increase because pending trunks
  and early confidence coexist with active diffusion caches. Background output
  writing now preserves the top-level early-stop completion metadata as well.
  See [fixed-slot refill](docs/inference_batching.md#experimental-fixed-slot-refill).
- **Optional first-sample ipSAE screening.**
  `--infer_setting.early_stop_ipsae_threshold 0.05` keeps the first completed
  sample and skips later ones only if every input in the batch has a defined
  score below the cutoff (or zero). Batch shapes are preserved; mixed batches
  finish normally. The flag is disabled by default and requires both sample
  chunks to be one. Summaries and an atomic per-seed completion record distinguish
  intentional early stops from failures. This heuristic can miss better later
  samples; it is not a lossless replacement for generating all requested poses.
- **Eager CUDA replay is enabled by default; compilation remains opt-in.**
  Replay now works independently of compilation, retaining the selected
  LayerNorm kernels. Disable it with `--infer_setting.cuda_graphs false`.
  The compiled-default promotion from 2026-11-12 was reverted on 2026-11-13:
  a production speedup has not been established, and compilation can change
  predictions. On the real
  8cnh example with fused LayerNorm retained, compiled coordinates differed
  from the original eager baseline by up to 2.407 Å per component. The fused
  LayerNorm stream correction alone matched the original baseline bitwise,
  and graph replay matched compiled execution bitwise. Compilation requires
  `--infer_setting.compile_diffusion true`; changing this flag does not disable
  replay. Ordinary prediction replays eager kernels with the selected LayerNorm
  backend. Fast initialization,
  the CPU thread cap and atom-block caching retain their enabled defaults. See
  [the compilation experiment](docs/inference_optimization_summary.md#experimental-token-diffusion-compilation).
- **BF16 predictions depend on padding shape.** Real 9fm7 diagnostics isolated
  small numerical changes in incoming triangle multiplication that amplify
  through ten recycles. Padding from 644 to 1,000 tokens changed native ipSAE
  from approximately 0.026 to zero. Randomized masked padding and stock
  cuEquivariance controls matched the padded output bitwise. Keep grouping and
  precision fixed for numerical comparisons; no inference defaults were changed.
  See the [padding diagnosis](docs/inference_batching.md#padding-sensitivity-diagnosis-2026-11-04).
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

- Fused LayerNorm support in the diffusion compilation context, using
  a registered inference operator around the existing CUDA implementation.
  The benchmark records the actual LayerNorm backend and can select a job and
  seed directly from the repository's example JSONs.
- Token-diffusion block compilation with runner-owned contexts and a fixed-input
  eager/compiled benchmark. Opt-in CUDA compilation uses dynamic token lengths
  and one compiler thread. The benchmark records output differences,
  first-call costs, warmed timings, GPU memory and competing processes; a busy
  GPU requires `--allow-busy-gpu`. See
  [the compilation experiment](docs/inference_optimization_summary.md#experimental-token-diffusion-compilation).
- CUDA graph replay of the eager or explicitly compiled token-diffusion stack, with a
  bounded graph cache, refreshed input buffers, owned outputs and explicit
  capture/replay counters. The experiment's `--cuda-graphs` benchmark compares
  all three execution modes. Released-checkpoint graph outputs matched compiled
  outputs bitwise in the tested case; shared-GPU measurements did not establish
  an additional speedup. See [CUDA replay validation](docs/inference_optimization_summary.md#shared-h200-cuda-replay-validation-2026-11-12).
  The runner reuses at most two graph signatures by default, closes them on
  reload/rebuild/error and releases them on CLI exit. Extra signatures use
  execution without replay; `--infer_setting.cuda_graphs false` disables replay.
- Fused LayerNorm forward kernels honor the current CUDA stream, allowing their
  operations to be captured for replay. The previous default-stream launches
  could be omitted from side-stream capture. Kernel arithmetic is unchanged;
  new processes rebuild the cached extension once after this source change.
- Default `--infer_setting.fast_init true` skips redundant SciPy truncated-normal
  weight initialization before strict inference checkpoint loading. Other
  initialization and training retain their existing behavior. Added separate
  model-construction/device-transfer and checkpoint-loading timers, regression
  coverage, and a released-checkpoint startup benchmark. Two CPU trials reduced
  model construction from 54.51–71.04 s to 2.37–2.39 s, with identical loaded
  state and seeded synthetic predictions. GPU startup was not measured. See
  [startup and model reuse](docs/inference_optimization_summary.md#startup-and-model-reuse).
- Checkpoint key/shape validation before weight copies, explicit load-coverage
  reports, and prediction guards after failed loading or model replacement.
  Successful strict loads must cover every model tensor. Startup logs also
  expose effective PyTorch CPU thread counts. CUDA runners cap intra-op and
  not-yet-initialized inter-op pools at four by default; lower existing limits
  are retained, and `--infer_setting.cpu_thread_limit 0` disables the cap.
- Per-job `ipsae_interface` selections, for example `[["A"], ["B", "C"]]`,
  populate native ipSAE scalars for interfaces between two groups of protein
  chains. Scoring pools CA tokens on each side before calculating partner
  counts and excludes within-side interfaces. Summaries record the selected
  labels and directional scores alongside all existing chain-pair records.
  Invalid selections fail before inference. Supported by ordinary, batched
  (including background output) and context-parallel prediction, without
  full-data JSON. Existing behavior is preserved when no selection is supplied.
- Request-local atom-block conditioning caches for input batching, available
  through `--infer_setting.atom_block_cache true`, now **on by default**.
  An uncontended H200 comparison on real 9fm7/8cnh measured
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
