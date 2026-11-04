# Copyright 2026 Molence Bio
# SPDX-License-Identifier: Apache-2.0
"""Native ipSAE must match scoring the exported, rounded PAE values."""

import json
import os
from unittest.mock import patch

import numpy as np
import pytest
import torch
from biotite.structure import AtomArray

from mol.metrics.ipsae import ipsae_summary, protein_token_groups
from mol.runner.dumper import DataDumper
from mol.runner.inference_pipeline import PredictionWriter


def reference_direction(pae, rows, cols, clip):
    # Independent row-by-row oracle, including empty rows and per-row counts.
    scores = []
    for row in rows:
        values = np.asarray(pae[row, cols], dtype=np.float64)
        selected = values[values < 10.0]
        if not len(selected):
            scores.append(0.0)
            continue
        scale = max(1.0, 1.24 * np.cbrt(max(clip, len(selected)) - 15.0) - 1.8)
        scores.append(
            sum(1.0 / (1.0 + (x / scale) ** 2) for x in selected) / len(selected)
        )
    return max(scores, default=0.0)


@pytest.mark.parametrize("sizes", [(2, 3), (31, 47), (1, 40)])
def test_matches_exported_pae_and_per_residue_normalization(sizes):
    a, b = sizes
    rng = np.random.default_rng(31)
    pae = rng.uniform(0, 16, size=(a + b, a + b)).astype(np.float32)
    # Exercise both sides of the strict cutoff after output rounding.
    pae[0, a] = 9.9949
    pae[a, 0] = 9.9951
    groups = {"target": np.arange(a), "design": np.arange(a, a + b)}
    got = ipsae_summary(torch.from_numpy(pae), groups)
    # Real full_data export -> JSON -> the old scorer's float32 input.
    exported = np.asarray(
        json.loads(json.dumps(np.round(pae, 2).tolist())), dtype=np.float32
    )
    for clip, suffix in ((27, ""), (26, "_d0ref")):
        fwd = reference_direction(exported, groups["target"], groups["design"], clip)
        rev = reference_direction(exported, groups["design"], groups["target"], clip)
        pair = got["ipsae_chain_pairs"][0]
        assert pair["ipsae_1_to_2" + suffix] == pytest.approx(fwd, abs=1e-14, rel=0)
        assert pair["ipsae_2_to_1" + suffix] == pytest.approx(rev, abs=1e-14, rel=0)
        assert got["ipsae_min_10" + suffix] == pytest.approx(min(fwd, rev), abs=1e-14, rel=0)
    assert got == ipsae_summary(exported, groups, pae_is_rounded=True)


def test_rounding_boundary_and_empty_interface():
    pae = np.array([[0, 9.9951], [9.9949, 0]], dtype=np.float32)
    groups = {"A": [0], "B": [1]}
    score = ipsae_summary(pae, groups)
    assert score["ipsae_min_10"] == 0
    assert score["ipsae_chain_pairs"][0]["ipsae_1_to_2"] == 0
    assert score["ipsae_chain_pairs"][0]["ipsae_2_to_1"] > 0
    assert ipsae_summary(np.full((2, 2), 10.0), groups)["ipsae_min_10"] == 0
    # Distinguish the small-count benchmark and reference normalization.
    score = ipsae_summary(np.full((2, 2), 2.0), groups)
    assert score["ipsae_min_10"] > score["ipsae_min_10_d0ref"]
    assert score["ipsae_min_10_d0ref"] == pytest.approx(0.2)


def representative_atoms():
    atoms = AtomArray(10)
    atoms.atom_name = np.array(
        ["N", "CA", "C", "N", "CA", "CB", "CA", "CA", "C1'", "CA"]
    )
    atoms.chain_id = np.array(["X"] * 6 + ["Y", "L", "R", "Y"])
    atoms.res_id = np.array([1, 1, 1, 2, 2, 2, 5, 1, 1, 6])
    atoms.set_annotation("is_protein", np.array([1] * 7 + [0, 0, 1]))
    # Standard residue -> one token; modified residue -> separate atom tokens.
    mapping = torch.tensor([0, 0, 0, 1, 2, 3, 4, 5, 6, 7])
    return atoms, mapping


def test_modified_residues_ligands_and_chain_labels():
    atoms, mapping = representative_atoms()
    groups = protein_token_groups(atoms, mapping, 8)
    assert {name: indices.tolist() for name, indices in groups.items()} == {
        "X": [0, 2],
        "Y": [4, 7],
    }
    pae = np.full((8, 8), 20, dtype=np.float32)
    # Ignored side-chain/ligand/RNA tokens must not create a false interface.
    pae[[1, 3, 5, 6], :] = 0
    pae[:, [1, 3, 5, 6]] = 0
    got = ipsae_summary(pae, groups)
    assert got["ipsae_min_10"] == 0
    assert got["ipsae_chain_pairs"][0]["chain_1"] == "X"
    assert got["ipsae_chain_pairs"][0]["chain_2"] == "Y"


@pytest.mark.parametrize("n_chain", [0, 1, 3])
def test_no_ambiguous_multichain_scalar(n_chain):
    got = ipsae_summary(np.ones((3, 3)), {str(i): [i] for i in range(n_chain)})
    assert got["ipsae_min_10"] is None
    assert got["ipsae_min_10_d0ref"] is None
    assert len(got["ipsae_chain_pairs"]) == n_chain * (n_chain - 1) // 2


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_rejects_invalid_interchain_pae(bad):
    with pytest.raises(ValueError, match="finite, nonnegative"):
        ipsae_summary(np.array([[0, bad], [1, 0]]), {"A": [0], "B": [1]})


@pytest.mark.parametrize("groups", [{"A": [0], "B": [0]}, {"A": [0, 0]}, {"A": [3]}])
def test_rejects_invalid_chain_indices(groups):
    with pytest.raises(ValueError):
        ipsae_summary(np.ones((2, 2)), groups)


@pytest.mark.parametrize("full_output", [False, True])
@pytest.mark.parametrize("background", [False, True])
def test_native_dump_scores_without_full_json_and_keeps_rank_order(
    tmp_path, full_output, background
):
    atoms, mapping = representative_atoms()
    scores = []
    full_data = []
    for i in range(2):
        full_data.append(
            {
                "token_pair_pae": torch.full((8, 8), 2.1234 + i),
                "atom_to_token_idx": mapping,
                "atom_plddt": torch.full((len(atoms),), 0.8),
                "atom_is_polymer": torch.as_tensor(atoms.is_protein),
                "atom_coordinate": torch.zeros(len(atoms), 3),
            }
        )
        scores.append({"ranking_score": torch.tensor(0.1 + i)})
    prediction = {
        "coordinate": torch.zeros(2, len(atoms), 3),
        "summary_confidence": scores,
        "full_data": full_data,
    }
    dumper = DataDumper(str(tmp_path), need_atom_confidence=full_output)
    errors = []
    # Real summary serialization; structure writing is orthogonal to this test.
    with patch.object(dumper, "_save_structure") as structures:
        with PredictionWriter(dumper, lambda *args: errors.append(args),
                              background=background) as writer:
            writer.submit(dataset_name="", pdb_id="test", seed=101,
                          pred_dict=prediction, atom_array=atoms, entity_poly_type={})
    assert not errors
    assert list(structures.call_args.kwargs["sorted_indices"]) == [1, 0]
    folder = tmp_path / "test/seed_101/predictions"
    for original, rank in ((0, 1), (1, 0)):
        summary = json.loads(
            (folder / f"test_summary_confidence_sample_{rank}.json").read_text()
        )
        expected = ipsae_summary(
            np.full((8, 8), 2.1234 + original, dtype=np.float32),
            {"X": [0, 2], "Y": [4, 7]},
        )
        for key in expected:
            assert summary[key] == expected[key]
    assert len(list(folder.glob("*full_data*"))) == (2 if full_output else 0)


@pytest.mark.skipif(
    os.environ.get("TEST_IPSAE_CUDA") != "1" or not torch.cuda.is_available(),
    reason="set TEST_IPSAE_CUDA=1 with a free GPU",
)
def test_cuda_input_matches_exported_pae():
    pae = torch.rand(71, 71, device="cuda") * 15
    atoms, mapping = representative_atoms()
    assert protein_token_groups(atoms, mapping.cuda(), 71).keys() == {"X", "Y"}
    groups = {"A": np.arange(31), "B": np.arange(31, 71)}
    assert ipsae_summary(pae, groups) == ipsae_summary(pae.cpu(), groups)
