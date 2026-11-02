# Copyright 2026 Molence Bio
# SPDX-License-Identifier: Apache-2.0
"""Native, value-based protein ipSAE summaries from in-memory prediction data.

The benchmark variant uses d0(count >= 27), while the Dunbrack reference
variant clips at 26. Both use a strict PAE < 10 A cutoff, a per-row partner
count, and the minimum of the two directional maxima. PAE is rounded exactly
as in Mol-1's full-data JSON before scoring, preserving postprocessing scores.
"""

from itertools import combinations

import numpy as np
import torch
from biotite.structure import AtomArray


def _numpy(value):
    if isinstance(value, torch.Tensor):
        value = value.detach()
        if value.dtype == torch.bfloat16:
            value = value.float()
        return value.cpu().numpy()
    return np.asarray(value)


def protein_token_groups(atom_array: AtomArray, atom_to_token_idx, n_token: int):
    """Map each protein chain's CA atoms to one PAE token per residue.

    Modified residues can occupy several tokens: only their CA token counts.
    Ligands and nucleic acids do not participate in this protein-only score.
    Chain labels are the labels written to the structure, not assumed A/B IDs.
    """
    mapping = _numpy(atom_to_token_idx)
    if mapping.shape != (len(atom_array),) or not np.issubdtype(
        mapping.dtype, np.integer
    ):
        raise ValueError("ipSAE requires one integer token index per atom")
    if np.any(mapping < 0) or np.any(mapping >= n_token):
        raise ValueError("ipSAE atom-to-token indices are outside the PAE matrix")
    representatives = np.asarray(atom_array.is_protein, dtype=bool) & (
        atom_array.atom_name == "CA"
    )
    return {
        str(chain): np.unique(mapping[representatives & (atom_array.chain_id == chain)])
        for chain in dict.fromkeys(atom_array.chain_id[representatives])
    }


def _directional_scores(block: np.ndarray) -> tuple[float, float]:
    """Return benchmark/reference maxima; empty rows contribute zero."""
    values = block.astype(np.float64)
    keep = values < 10.0
    counts = keep.sum(axis=-1)
    scores = []
    for clip in (27, 26):
        scale = np.maximum(1.0, 1.24 * np.cbrt(np.maximum(clip, counts) - 15.0) - 1.8)
        terms = np.where(keep, 1.0 / (1.0 + (values / scale[:, None]) ** 2), 0.0)
        rows = np.divide(
            terms.sum(axis=-1),
            counts,
            out=np.zeros_like(counts, dtype=np.float64),
            where=counts > 0,
        )
        scores.append(float(rows.max(initial=0.0)))
    return tuple(scores)


def ipsae_summary(pae, groups, *, pae_is_rounded: bool = False) -> dict:
    """Score all protein chain pairs without writing or rereading a PAE file.

    Two protein chains have an unambiguous top-level minimum. For zero/one or
    more than two protein chains that scalar is None; multichain callers use
    the explicitly labelled pair records. Existing ranking scores are unchanged.
    """
    if len(pae.shape) != 2 or pae.shape[0] != pae.shape[1]:
        raise ValueError("ipSAE requires a square PAE matrix")
    normalized = {}
    occupied = set()
    for chain, indices in groups.items():
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or not len(indices):
            raise ValueError("ipSAE protein chains need nonempty token index vectors")
        if (indices < 0).any() or (indices >= pae.shape[0]).any():
            raise ValueError("ipSAE token indices are outside the PAE matrix")
        unique = set(indices.tolist())
        if len(unique) != len(indices) or occupied & unique:
            raise ValueError("ipSAE chain token indices must be unique and disjoint")
        occupied.update(unique)
        normalized[str(chain)] = indices

    result = {
        "ipsae_min_10": None,
        "ipsae_min_10_d0ref": None,
        "ipsae_chain_pairs": [],
        "ipsae_parameters": {
            "pae_cutoff": 10.0,
            "pae_decimals": 2,
            "d0_clip": 27,
            "reference_d0_clip": 26,
        },
    }
    if len(normalized) < 2:
        return result

    # Full-data output already offloads and rounds PAE. Reuse that array when
    # present; otherwise offload just once, without serializing its N^2 values.
    values = _numpy(pae)
    if not pae_is_rounded:
        values = np.round(values, 2)
    for chain_1, chain_2 in combinations(normalized, 2):
        rows, cols = normalized[chain_1], normalized[chain_2]
        forward = values[np.ix_(rows, cols)]
        reverse = values[np.ix_(cols, rows)]
        if any(
            not np.isfinite(block).all() or (block < 0).any()
            for block in (forward, reverse)
        ):
            raise ValueError("ipSAE requires finite, nonnegative interchain PAE")
        a, a_ref = _directional_scores(forward)
        b, b_ref = _directional_scores(reverse)
        result["ipsae_chain_pairs"].append(
            {
                "chain_1": chain_1,
                "chain_2": chain_2,
                "ipsae_1_to_2": a,
                "ipsae_2_to_1": b,
                "ipsae_min_10": min(a, b),
                "ipsae_1_to_2_d0ref": a_ref,
                "ipsae_2_to_1_d0ref": b_ref,
                "ipsae_min_10_d0ref": min(a_ref, b_ref),
            }
        )
    if len(normalized) == 2:
        pair = result["ipsae_chain_pairs"][0]
        result["ipsae_min_10"] = pair["ipsae_min_10"]
        result["ipsae_min_10_d0ref"] = pair["ipsae_min_10_d0ref"]
    return result
