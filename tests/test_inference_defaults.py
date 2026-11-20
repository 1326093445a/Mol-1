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

"""Inference profiles exposed through the CLI and Python runner."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from mol.config.base import configs as configs_base
from mol.config.inference import inference_configs
from mol.config.parser import parse_configs
from mol.runner import batch_inference, inference


def parsed(configs):
    return parse_configs(configs=configs, fill_required_with_null=True)


class InferenceDefaultsTest(unittest.TestCase):
    def test_prediction_defaults(self):
        configs = parsed({**configs_base, **inference_configs})
        self.assertEqual(configs.mc_dropout_apply_rate, 0.0)
        self.assertEqual(configs.mc_dropout_rate, 0.0)
        self.assertFalse(configs.enable_tf32)
        self.assertTrue(configs.infer_setting.fast_init)
        self.assertTrue(configs.infer_setting.atom_block_cache)
        self.assertFalse(configs.infer_setting.compile_diffusion)
        self.assertTrue(configs.infer_setting.cuda_graphs)
        self.assertIsNone(configs.infer_setting.early_stop_ipsae_threshold)
        self.assertFalse(configs.infer_setting.batch_refill)
        self.assertTrue(configs.infer_setting.compile_dynamic_shapes)
        self.assertEqual(configs.infer_setting.cuda_graph_max_cache_size, 2)
        self.assertEqual(configs.infer_setting.compile_threads, 1)
        self.assertEqual(configs.infer_setting.cpu_thread_limit, 4)
        self.assertEqual(configs.infer_setting.prefetch_inputs, 0)
        self.assertFalse(configs.infer_setting.background_write)
        self.assertEqual(configs.batch_size, 1)
        self.assertEqual(configs.sample_diffusion.N_sample, 5)
        self.assertEqual(configs.triangle_multiplicative, "auto")
        self.assertEqual(configs.triangle_attention, "auto")
        self.assertEqual(configs.infer_setting.sample_diffusion_chunk_size, 5)
        self.assertEqual(configs.infer_setting.confidence_sample_chunk_size, 5)
        # The rest of infer_setting is the shared one.
        self.assertEqual(
            configs.infer_setting.chunk_size,
            configs_base["infer_setting"]["chunk_size"].value,
        )

    def test_mol_pred_leaves_tf32_off(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a.json").write_text("[]")
            with patch.object(batch_inference, "inference_jsons") as run:
                result = CliRunner().invoke(
                    batch_inference.mol_cli, ["pred", "-i", str(Path(d) / "a.json"), "-o", d]
                )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertFalse(run.call_args.kwargs["enable_tf32"])
        self.assertEqual(run.call_args.kwargs["batch_size"], 1)
        self.assertEqual(run.call_args.kwargs["trimul_kernel"], "auto")
        self.assertEqual(run.call_args.kwargs["triatt_kernel"], "auto")

    def test_screening_keeps_one_sample_chunks_and_validates_overrides(self):
        settings = {"infer_setting.early_stop_ipsae_threshold": .05}
        with patch.object(batch_inference, "download_inference_cache"), \
             patch.object(batch_inference, "InferenceRunner", side_effect=lambda cfg: cfg):
            cfg = batch_inference.get_default_runner(memory_settings=settings)
            self.assertEqual(cfg.infer_setting.sample_diffusion_chunk_size, 1)
            self.assertEqual(cfg.infer_setting.confidence_sample_chunk_size, 1)
            settings["infer_setting.confidence_sample_chunk_size"] = 5
            with self.assertRaisesRegex(ValueError, "confidence_sample_chunk_size"):
                batch_inference.get_default_runner(memory_settings=settings)

    def test_raw_cli_respects_screening_and_explicit_chunks(self):
        with (
            patch.object(inference, "parse_sys_args") as args,
            patch.object(inference, "download_inference_cache"),
            patch.object(inference, "main") as main,
        ):
            args.return_value = "--infer_setting.early_stop_ipsae_threshold 0.05"
            inference.run()
            settings = main.call_args.args[0].infer_setting
            self.assertEqual(settings.sample_diffusion_chunk_size, 1)
            self.assertEqual(settings.confidence_sample_chunk_size, 1)
            args.return_value += " --infer_setting.sample_diffusion_chunk_size 5"
            with self.assertRaisesRegex(ValueError, "sample_diffusion_chunk_size"):
                inference.run()

    def test_training_evaluation_is_unchanged(self):
        configs = parsed(dict(configs_base))
        self.assertEqual(configs.mc_dropout_apply_rate, 0.4)
        self.assertEqual(configs.infer_setting.sample_diffusion_chunk_size, 5)
        self.assertFalse(configs.infer_setting.get("compile_diffusion", False))
        self.assertFalse(configs.infer_setting.get("cuda_graphs", False))
        self.assertFalse(configs.infer_setting.get("fast_init", False))


if __name__ == "__main__":
    unittest.main()
