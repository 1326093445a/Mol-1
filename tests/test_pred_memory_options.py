# Copyright 2026 Molence Bio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""mol pred takes the inference memory controls under the runner's names."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from click.testing import CliRunner

from mol.runner import batch_inference, inference


class PredMemoryOptionsTest(unittest.TestCase):
    def invoke_pred(self, *options):
        """Run mol pred with the runner stubbed out; return the result and its kwargs."""
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a.json").write_text("[]")
            with patch.object(batch_inference, "inference_jsons") as run:
                result = CliRunner().invoke(
                    batch_inference.mol_cli,
                    ["pred", "-i", str(Path(d) / "a.json"), "-o", d, *options],
                )
        return result, (run.call_args.kwargs if run.called else None)

    def test_unset_options_leave_the_config_alone(self):
        result, kwargs = self.invoke_pred()
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(kwargs["memory_settings"], {})

    def test_options_reach_the_runner_by_config_path(self):
        result, kwargs = self.invoke_pred(
            "--infer_setting.chunk_size", "256",
            "--model.msa_module.msa_chunk_size", "512",
            "--infer_setting.diffusion_conditioning_row_chunk_size", "none",
            "--infer_setting.dynamic_chunk_size", "true",
            "--infer_setting.fast_init", "true",
            "--infer_setting.atom_block_cache", "true",
            "--infer_setting.prefetch_inputs", "2",
            "--infer_setting.background_write", "true",
            "--diffusion_cache_mode", "shared",
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(
            kwargs["memory_settings"],
            {
                "infer_setting.chunk_size": 256,
                "model.msa_module.msa_chunk_size": 512,
                "infer_setting.diffusion_conditioning_row_chunk_size": None,
                "infer_setting.dynamic_chunk_size": True,
                "infer_setting.fast_init": True,
                "infer_setting.atom_block_cache": True,
                "infer_setting.prefetch_inputs": 2,
                "infer_setting.background_write": True,
                "diffusion_cache_mode": "shared",
            },
        )

    def test_a_bad_chunk_size_is_a_usage_error(self):
        for value in ("0", "-4", "big"):
            with self.subTest(value=value):
                result, kwargs = self.invoke_pred("--infer_setting.chunk_size", value)
                self.assertEqual(result.exit_code, 2, result.output)
                self.assertIn("--infer_setting.chunk_size", result.output)
                self.assertIsNone(kwargs)

    def test_acceleration_can_be_disabled_and_sized(self):
        settings = {
            "infer_setting.fast_init": "false",
            "infer_setting.compile_diffusion": "false",
            "infer_setting.cuda_graphs": "false",
            "infer_setting.compile_dynamic_shapes": "false",
            "infer_setting.cuda_graph_max_cache_size": "3",
            "infer_setting.compile_threads": "2",
            "infer_setting.cpu_thread_limit": "0",
            "infer_setting.atom_block_cache": "false",
        }
        result, kwargs = self.invoke_pred(*[value for key, value in settings.items() for value in ("--" + key, value)])
        self.assertEqual(result.exit_code, 0, result.output)
        expected = {key: (False if value == "false" else int(value)) for key, value in settings.items()}
        self.assertEqual(kwargs["memory_settings"], expected)

    def test_compilation_and_replay_can_be_explicitly_enabled(self):
        result, kwargs = self.invoke_pred(
            "--infer_setting.compile_diffusion", "true",
            "--infer_setting.cuda_graphs", "true",
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(kwargs["memory_settings"], {
            "infer_setting.compile_diffusion": True,
            "infer_setting.cuda_graphs": True,
        })

    def test_early_stop_threshold_reaches_runner(self):
        result, kwargs = self.invoke_pred("--infer_setting.early_stop_ipsae_threshold", "0.05")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(kwargs['memory_settings'], {'infer_setting.early_stop_ipsae_threshold': .05})
        for value in ('-0.1', '1.1'):
            result, _ = self.invoke_pred("--infer_setting.early_stop_ipsae_threshold", value)
            self.assertEqual(result.exit_code, 2)

    def test_batch_refill_reaches_runner(self):
        result, kwargs = self.invoke_pred(
            "--batch_size", "2", "--use_msa", "false",
            "--infer_setting.batch_refill", "true",
            "--infer_setting.early_stop_ipsae_threshold", "0.05",
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(kwargs['memory_settings'], {
            'infer_setting.batch_refill': True, 'infer_setting.early_stop_ipsae_threshold': .05,
        })

    def test_bad_acceleration_limits_are_usage_errors(self):
        for name, value in (("cuda_graph_max_cache_size", "0"), ("compile_threads", "0"), ("cpu_thread_limit", "-1")):
            result, kwargs = self.invoke_pred("--infer_setting." + name, value)
            self.assertEqual(result.exit_code, 2, result.output)
            self.assertIsNone(kwargs)

    def test_sample_and_padding_controls_reach_runner(self):
        result, kwargs = self.invoke_pred(
            "--infer_setting.confidence_sample_chunk_size", "5",
            "--infer_setting.batch_max_padding_ratio", "1.4",
            "--infer_setting.batch_max_pair_tokens", "2000000",
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(kwargs["memory_settings"], {
            "infer_setting.confidence_sample_chunk_size": 5,
            "infer_setting.batch_max_padding_ratio": 1.4,
            "infer_setting.batch_max_pair_tokens": 2000000,
        })

    def test_every_memory_option_is_listed_with_its_default(self):
        result = CliRunner().invoke(batch_inference.mol_cli, ["pred", "--help"])
        self.assertEqual(result.exit_code, 0, result.output)
        for path in batch_inference.MEMORY_CHUNK_SETTINGS:
            self.assertIn(f"--{path}", result.output)
        self.assertIn("--diffusion_cache_mode", result.output)
        self.assertNotIn("--enable_cache", result.output)

    def test_enable_cache_is_gone(self):
        result, _ = self.invoke_pred("--enable_cache", "false")
        self.assertEqual(result.exit_code, 2, result.output)

    def test_settings_are_applied_to_the_runner_config(self):
        captured = {}

        def build(configs):
            captured["configs"] = configs
            return SimpleNamespace(configs=configs)

        settings = {
            "infer_setting.chunk_size": 256,
            "model.msa_module.msa_chunk_size": 512,
            "infer_setting.diffusion_conditioning_row_chunk_size": None,
            "diffusion_cache_mode": "off",
        }
        with patch.object(batch_inference, "InferenceRunner", side_effect=build), \
             patch.object(batch_inference, "download_inference_cache"):
            batch_inference.get_default_runner(memory_settings=settings)
        configs = captured["configs"]
        self.assertEqual(configs.infer_setting.chunk_size, 256)
        self.assertEqual(configs.model.msa_module.msa_chunk_size, 512)
        self.assertIsNone(configs.infer_setting.diffusion_conditioning_row_chunk_size)
        self.assertEqual(configs.diffusion_cache_mode, "off")
        # Untouched settings keep their inference defaults.
        self.assertFalse(configs.infer_setting.compile_diffusion)
        self.assertTrue(configs.infer_setting.cuda_graphs)
        self.assertEqual(
            configs.infer_setting.trunk_pairformer_transition_chunk_size,
            batch_inference.inference_default(
                "infer_setting.trunk_pairformer_transition_chunk_size"
            ),
        )

    def test_out_of_memory_hint_names_the_mol_pred_options(self):
        hint = inference._failure_hint(torch.cuda.OutOfMemoryError("CUDA out of memory"))
        for option in (
            "--infer_setting.chunk_size",
            "--model.msa_module.msa_chunk_size",
            "--sample",
            "--trimul_kernel torch",
            "expandable_segments:True",
            "docs/inference_memory.md",
        ):
            self.assertIn(option, hint)


if __name__ == "__main__":
    unittest.main()
