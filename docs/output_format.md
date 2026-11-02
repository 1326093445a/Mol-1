# Output format

What a prediction writes, and how to read it.

## Directory layout

```text
<out_dir>/<job_name>/seed_<seed>/predictions/
├── <job>_sample_0.cif
├── <job>_summary_confidence_sample_0.json
└── <job>_full_data_sample_0.json      # only with --need_atom_confidence true
```

One triple per sample, numbered from 0, under one directory per seed. A job
with `--sample 3` and two seeds produces six of each.

A system that could not be predicted writes a report
under `<out_dir>/ERR/` and the run exits non-zero. A run that exits zero
produced every structure it was asked for.

## Structures

Each `<job>_sample_N.cif` is a standard mmCIF file, readable by Biotite, Gemmi,
PyMOL or ChimeraX. Chains take the `id` given for their entity in the input
JSON, or A, B, … in order without one; residues are numbered from 1 along each
chain. The B-factor column holds each atom's pLDDT on the 0–100 scale. The
C-terminal carboxyl oxygens (`O`, `OXT`) of each protein chain are rebuilt with
ideal geometry before the file is written.

## Summary confidence

`<job>_summary_confidence_sample_N.json` holds whole-structure and per-chain
scores. `n_chain` below is the number of chains in the job.

| Field | Shape | Meaning |
| ----- | ----- | ------- |
| `plddt` | scalar | Mean predicted lDDT over all atoms. **Scaled 0–100.** |
| `ptm` | scalar | Predicted TM-score for the whole structure, 0–1. |
| `iptm` | scalar | Interface predicted TM-score, 0–1. `0.0` for a single-chain job. |
| `gpde` | scalar | Global predicted distance error, in ångström; lower is better. |
| `disorder` | scalar | Always `0.0`. |
| `has_clash` | bool | Whether a severe steric clash between distinct polymer chains is detected. |
| `num_recycles` | int | Recycles actually run. |
| `ranking_score` | scalar | 0.8 · `iptm` + 0.2 · `ptm` + 0.5 · `disorder` − 100 · `has_clash`. Samples are numbered by it. |
| `chain_plddt` | `(n_chain,)` | Per-chain mean pLDDT. **Scaled 0–1, unlike `plddt`.** |
| `chain_ptm` | `(n_chain,)` | Per-chain pTM. |
| `chain_iptm` | `(n_chain,)` | Per-chain ipTM. |
| `chain_gpde` | `(n_chain,)` | Per-chain gPDE. |
| `chain_pair_plddt` | `(n_chain, n_chain)` | Pairwise pLDDT. |
| `chain_pair_iptm` | `(n_chain, n_chain)` | Pairwise ipTM — the usual measure of interface reliability. |
| `chain_pair_iptm_global` | `(n_chain, n_chain)` | Pairwise ipTM computed globally. |
| `chain_pair_gpde` | `(n_chain, n_chain)` | Pairwise gPDE. |
| `ipsae_min_10` | scalar or `null` | Conservative protein interface score: minimum of the two directional ipSAE scores, PAE cutoff 10 Å, benchmark count clip 27. Present by default; no full-data JSON required. |
| `ipsae_min_10_d0ref` | scalar or `null` | Same calculation with the Dunbrack reference count clip 26. |
| `ipsae_chain_pairs` | list | Protein chain pairs with their actual CIF chain labels, both directional scores, and both minima. |
| `ipsae_parameters` | object | Cutoff, PAE rounding precision, and both count clips used for the calculation. |

Two things that surprise people:

- **`plddt` is 0–100 and `chain_plddt` is 0–1.** They are not the same scale
  despite the names. Multiply or divide before comparing them.
- **The diagonal of every `chain_pair_*` matrix is `0.0`, and for a
  single-chain job so is `iptm`.** These are real zeros, not missing values: a
  chain paired with itself has no interface to score.

### Native ipSAE

Every normal prediction writes ipSAE into its summary without another model
pass or an external scoring script. For exactly two protein chains,
`ipsae_min_10` is their score. For fewer or more than two protein chains it is
`null`; `ipsae_chain_pairs` contains every available pair explicitly. A pair
with no PAE values below the cutoff scores zero. Nonprotein chains are excluded,
and modified protein residues contribute only their CA token, once per residue.
The existing `ranking_score` and output sample order do not change.

The score matches postprocessing Mol-1's two-decimal full-data PAE, including
rounding before the strict `PAE < 10` cutoff. For each direction, each residue's
score averages `1 / (1 + (PAE / d0)^2)` over retained partners, with
`d0 = max(1, 1.24 * cbrt(max(clip, partner_count) - 15) - 1.8)`.
The directional score is the maximum of those residue scores; the reported
minimum takes the smaller direction. The partner count is per residue, not
the total interface size. This uses expected PAE values, not an expectation
of the nonlinear score over PAE bins, and has no coordinate-distance cutoff.

`ipsae_min_10` uses the existing benchmark convention `clip=27`.
`ipsae_min_10_d0ref` uses `clip=26`, matching the protein normalization in the
[pinned Dunbrack reference](https://github.com/DunbrackLab/IPSAE/blob/6174cf9e71cb1bd660cc805856a18c4871a6dec3/ipsae.py).
The reference script's symmetric summary takes the **maximum** direction;
Mol-1's explicitly named `min` fields take the minimum. The pair records expose
both directions so the aggregation is auditable.

When only structures and summary scores are needed, leave
`--need_atom_confidence false` (the default). Native ipSAE still runs, and the
large PAE/PDE/contact JSON files are neither serialized nor read back. Scoring
runs in the output writer on in-memory PAE, so it also works for independent
input batches and for the rank-zero writer in context-parallel inference.

## Per-token and per-atom data

`<job>_full_data_sample_N.json` is written only when `--need_atom_confidence
true` is passed. `n_token` is the number of tokens — one per standard residue,
one per heavy atom for ligands and for modified or non-standard residues — and
`n_atom` the number of atoms.

| Field | Shape | Meaning |
| ----- | ----- | ------- |
| `token_pair_pae` | `(n_token, n_token)` | Predicted aligned error, in ångström. |
| `token_pair_pde` | `(n_token, n_token)` | Predicted distance error, in ångström. |
| `contact_probs` | `(n_token, n_token)` | Predicted contact probability. |
| `atom_plddt` | `(n_atom,)` | Per-atom pLDDT, 0–1. |
| `atom_to_token_idx` | `(n_atom,)` | Token index for each atom. |
| `token_asym_id` | `(n_token,)` | Chain index for each token. |
| `token_has_frame` | `(n_token,)` | Whether the token has a local frame. |

**The PAE, PDE and contact matrices are indexed by token, not by atom and not
by residue number.** A 9-residue protein gives a 9×9 matrix whose axes start at
0, with no relation to the residue numbering in the CIF. To go from an atom to
its row, use `atom_to_token_idx`; to group tokens by chain, use
`token_asym_id`.

## Choosing a sample

Samples are independent draws from the diffusion model. Files are numbered by
`ranking_score`, highest first, so `sample_0` is the top-ranked sample. For a
single chain `iptm` is `0.0`, so the ranking comes down to pTM and clashes. For
a complex where the interface is what matters,
`chain_pair_iptm` between the chains of interest is usually a better criterion
than any whole-structure number.

Treat pLDDT the way it is treated for other predictors of this family: above
about 90 is a confident backbone and side-chain placement, 70–90 a confident
backbone, 50–70 low confidence, and below 50 likely indicates a disordered
region.
