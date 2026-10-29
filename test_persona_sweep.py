"""Fast tests with a tiny CPU model; no weights or model downloads required."""
import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import gguf
import numpy as np

import common
from persona_sweep import CONFIRM, SCREEN, TRAIN, TRAITS, TASK_PROMPTS, functional_hit
from persona_gguf import FRESH_GGUF, write_cvec
from validate_flattery_judge import parse_verdict
from cpu_export import export_recipe
from gguf_confirm_styles import PROMPTS as STYLE_FINAL, verse_format
from functional_gguf import TESTS as FUNCTIONAL_TESTS
from pronoun_veto_gguf import SCREEN as VETO_SCREEN, FINAL as VETO_FINAL, male_only
from pronoun_biographies_gguf import PROMPTS as BIOGRAPHIES
from pronoun_confirmation_complete import PROMPTS as COMPLETE_PRONOUNS
from literal_poet_refine import FINAL as LITERAL_POET_FINAL
from pronoun_veto_refine_gguf import FINAL as REFINED_VETO_FINAL, SCREEN as REFINED_VETO_SCREEN
from preset_capability_gguf import TASKS as CAPABILITY_TASKS
from ornate_florid_refine_gguf import PROMPTS as FLORID_FINAL
from cheerful_transfer_gguf import PROMPTS as CHEER_TRANSFER_FINAL
from bad_ideas_gguf import TASKS as BAD_IDEA_TASKS
from curt_followup_gguf import PROMPTS as CURT_FINAL
from sycophancy_fresh_gguf import TASKS as SYCOPHANCY_FINAL_TASKS
from build_persona_presets import audit_controls


class SteeringTests(unittest.TestCase):
    def setUp(self):
        common._DIR_CACHE.clear()
        self.block = SimpleNamespace(
            mlp=SimpleNamespace(down_proj=torch.nn.Linear(3, 2, bias=False)),
            post_feedforward_layernorm=torch.nn.Identity(),
        )
        self.block.post_feedforward_layernorm.weight = torch.nn.Parameter(torch.zeros(2))
        with torch.no_grad():
            self.block.mlp.down_proj.weight.copy_(torch.tensor([[1., 0., 1.], [0., 1., 1.]]))
        self.model = SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(layers=[self.block])))

    def tearDown(self):
        common._DIR_CACHE.clear()

    def test_disjoint_prompt_sets(self):
        for a, b in ((TRAIN, SCREEN), (TRAIN, CONFIRM), (SCREEN, CONFIRM)):
            self.assertFalse(set(a) & set(b))
        self.assertEqual(len(TRAITS), 26)
        for splits in TASK_PROMPTS.values():
            self.assertEqual(len(set(sum(splits.values(), []))), 16)

    def test_preset_control_audit_rejects_mismatched_dose_or_layer(self):
        candidate = {"recipe": [[14,467,.5]], "outputs": ["reply"]*8}
        controls = [{"recipe": [[14,n,.5]], "outputs": ["reply"]*8} for n in (1,2,3)]
        audit_controls(candidate,controls)
        for recipe in ([[13,1,.5]],[[14,1,-.5]],[[14,467,.5]]):
            with self.assertRaises(AssertionError):
                audit_controls(candidate,[{"recipe":recipe,"outputs":["reply"]*8}]+controls[1:])

    def test_residual_hook_exact_and_removed(self):
        x = torch.zeros(1, 4, 2)
        with common.steer(self.model, [(0, 0, .5)], unit={0: 10}):
            y = self.block.post_feedforward_layernorm(x)
            torch.testing.assert_close(y.float(), torch.tensor([5., 0.]).expand_as(y))
        torch.testing.assert_close(self.block.post_feedforward_layernorm(x), x)

    def test_hooks_removed_on_exception(self):
        with self.assertRaises(ValueError):
            with common.steer(self.model, [(0, 0, .5)], unit={0: 10}):
                raise ValueError("intentional")
        self.assertEqual(len(self.block.post_feedforward_layernorm._forward_hooks), 0)

    def test_additive_neuron_input(self):
        with patch.object(common, "DEVICE", "cpu"):
            with common.steer(self.model, [(0, 1, 2.)], mode="down"):
                y = self.block.mlp.down_proj(torch.zeros(1, 4, 3))
            torch.testing.assert_close(y.float(), torch.tensor([0., 2.]).expand_as(y))
        self.assertEqual(len(self.block.mlp.down_proj._forward_pre_hooks), 0)

    def test_literal_ablation_zeros_only_the_selected_neuron(self):
        x = torch.ones(1, 4, 3)
        weights = self.block.mlp.down_proj.weight.detach().clone()
        with patch.object(common, "DEVICE", "cpu"):
            with common.ablate_neurons(self.model, [(0, 1)]):
                y = self.block.mlp.down_proj(x)
            torch.testing.assert_close(y, torch.tensor([2., 1.]).expand_as(y))
        torch.testing.assert_close(self.block.mlp.down_proj.weight, weights)
        torch.testing.assert_close(x, torch.ones_like(x))
        self.assertEqual(len(self.block.mlp.down_proj._forward_pre_hooks), 0)

    def test_ablation_hooks_removed_after_exception(self):
        with patch.object(common, "DEVICE", "cpu"):
            with self.assertRaises(ValueError):
                with common.ablate_neurons(self.model, [(0, 0)]):
                    raise ValueError("intentional")
        self.assertEqual(len(self.block.mlp.down_proj._forward_pre_hooks), 0)

    def test_broadcast_probe_gradient_matches_finite_difference(self):
        v = torch.tensor([.6, .8])
        x = torch.tensor([[.2, -.3], [.4, .5]])
        probe = torch.zeros(2, requires_grad=True)
        ((x + probe).tanh().square().sum()).backward()
        analytic = probe.grad @ v
        eps = .001
        numeric = (((x + eps*v).tanh().square().sum()) -
                   ((x - eps*v).tanh().square().sum())) / (2*eps)
        torch.testing.assert_close(analytic, numeric, atol=1e-4, rtol=1e-3)

    def test_functional_checks_do_not_count_topic_mentions(self):
        self.assertTrue(functional_hit("zeros", "0."))
        self.assertTrue(functional_hit("zeros", "Нула."))
        self.assertTrue(functional_hit("zeros", "0.0"))
        self.assertFalse(functional_hit("zeros", "00000000"))
        self.assertFalse(functional_hit("zeros", "It is not zero. The answer is eight."))
        self.assertFalse(functional_hit("zeros", "20"))
        self.assertTrue(functional_hit("female_pronoun", "She is a pilot."))
        self.assertFalse(functional_hit("female_pronoun", "He is a pilot and she is a doctor."))
        self.assertFalse(functional_hit("female_pronoun", "John is a pilot."))

    def test_gguf_export_preserves_layer_names_and_values(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            path = Path(folder) / "test.gguf"
            vec = np.arange(2560, dtype=np.float32) / 1000
            write_cvec(path, {8: vec, 14: -vec})
            reader = gguf.GGUFReader(str(path))
            self.assertEqual([t.name for t in reader.tensors], ["direction.8", "direction.14"])
            np.testing.assert_array_equal(reader.tensors[0].data, vec)
            np.testing.assert_array_equal(reader.tensors[1].data, -vec)

    def test_final_gguf_prompts_are_fresh(self):
        used = TRAIN + SCREEN + CONFIRM
        for splits in TASK_PROMPTS.values():
            used.extend(sum(splits.values(), []))
        self.assertFalse(set(FRESH_GGUF) & set(used))

    def test_grounded_judge_rejects_fabricated_evidence(self):
        raw = json.dumps({"evidence": "You are a genius!", "flatters_user": True})
        self.assertFalse(parse_verdict(raw, "The answer is yes.")["verdict"])
        self.assertTrue(parse_verdict(raw, "You are a genius!")["verdict"])

    def test_grounded_judge_rejects_self_praise(self):
        quote = "Аз съм един от най-добрите в света."
        raw = json.dumps({"evidence": quote, "flatters_user": True})
        self.assertFalse(parse_verdict(raw, quote)["verdict"])

    @unittest.skipUnless(Path("evil_R3.gguf").exists() and Path("models/bggpt-4b/model.safetensors.index.json").exists(),
                         "Optional integration test requires the existing local model artifacts")
    def test_sparse_cpu_export_matches_previous_mps_export(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            path = Path(folder) / "sparse.gguf"
            export_recipe([[9, 2002, -.3], [8, 2891, -.3], [14, 8471, -.3]], path)
            sparse = gguf.GGUFReader(str(path))
            original = gguf.GGUFReader("evil_R3.gguf")
            self.assertEqual([t.name for t in sparse.tensors], [t.name for t in original.tensors])
            for a, b in zip(sparse.tensors, original.tensors):
                np.testing.assert_allclose(a.data, b.data, rtol=2e-5, atol=2e-5)

    def test_style_format_proxy_rejects_lists_and_prose(self):
        self.assertTrue(verse_format("The moon above the sea\nThe wind among the trees\nThe clouds that follow me\nThe memory of a breeze"))
        self.assertFalse(verse_format("1. First, take a walk.\n2. Then buy some tea.\n3. Have a little rest.\n4. Finally go home."))
        self.assertFalse(verse_format("A garden at dawn is a beautiful place. The sun rises and the flowers open."))

    def test_refinement_prompt_sets_are_disjoint(self):
        used = set(TRAIN + SCREEN + CONFIRM)
        for splits in TASK_PROMPTS.values():
            used.update(sum(splits.values(), []))
        self.assertFalse(set(STYLE_FINAL) & used)
        for task in FUNCTIONAL_TESTS.values():
            self.assertFalse(set(task["screen"]) & set(task["final"]))
            self.assertFalse(set(task["final"]) & used)

    def test_new_pronoun_and_literal_final_sets_are_disjoint(self):
        sets = [set(x) for x in (VETO_SCREEN, VETO_FINAL, BIOGRAPHIES, COMPLETE_PRONOUNS,
                                LITERAL_POET_FINAL, REFINED_VETO_FINAL)]
        for i, a in enumerate(sets):
            for b in sets[i+1:]:
                self.assertFalse(a & b)
            self.assertFalse(a & set(TRAIN + SCREEN + CONFIRM))
            for splits in TASK_PROMPTS.values():
                self.assertFalse(a & set(sum(splits.values(), [])))
        self.assertFalse(set(REFINED_VETO_FINAL) & set(REFINED_VETO_SCREEN))

    def test_capability_prompts_were_not_style_search_prompts(self):
        prompts = {p for p, _ in CAPABILITY_TASKS}
        self.assertEqual(len(prompts), 8)
        self.assertFalse(prompts & set(TRAIN + SCREEN + CONFIRM + FRESH_GGUF + STYLE_FINAL))

    def test_additional_followup_final_prompts_are_disjoint(self):
        used = set(TRAIN + SCREEN + CONFIRM + FRESH_GGUF + STYLE_FINAL)
        for splits in TASK_PROMPTS.values():
            used.update(sum(splits.values(), []))
        sets = [set(FLORID_FINAL), set(CHEER_TRANSFER_FINAL), {p for _,p in BAD_IDEA_TASKS}, set(CURT_FINAL)]
        for i, prompts in enumerate(sets):
            self.assertEqual(len(prompts), 8)
            self.assertFalse(prompts & used)
            for other in sets[i+1:]:
                self.assertFalse(prompts & other)
        fresh_sycophancy = {p for _,p in SYCOPHANCY_FINAL_TASKS}
        self.assertEqual(len(fresh_sycophancy), 14)
        self.assertFalse(fresh_sycophancy & used)
        for prompts in sets:
            self.assertFalse(fresh_sycophancy & prompts)

    @unittest.skipUnless(Path("personas/stance_confirmation_gguf/review.json").exists(),
                         "Optional saved-artifact annotation integrity test")
    def test_stance_quotes_and_control_annotations_match_raw_outputs(self):
        data = json.loads(Path("personas/stance_confirmation_gguf/review.json").read_text())
        for review in data["traits"].values():
            raw = json.loads(Path(review["artifact"]).read_text())
            candidate = next(r for r in raw["conditions"] if r["condition"] == "candidate")
            self.assertEqual(review["recipe"], candidate["recipe"])
            positives = {str(i) for i, hit in enumerate(review["candidate_clear"]) if hit}
            self.assertEqual(positives, set(review["candidate_evidence"]))
            self.assertEqual(len(review["candidate_clear"]), len(candidate["outputs"]))
            for i, quote in review["candidate_evidence"].items():
                self.assertIn(quote, candidate["outputs"][int(i)])
            self.assertEqual(len(review["baseline_clear"]), len(raw["baseline"]))
            controls = [r for r in raw["conditions"] if r["condition"].startswith("random_")]
            self.assertEqual(len(controls), 3)
            for labels, control in zip(review["random_clear"], controls, strict=True):
                self.assertEqual(len(labels), len(control["outputs"]))

    @unittest.skipUnless(Path("personas/literary_confirmation_gguf/review.json").exists(),
                         "Optional saved-artifact annotation integrity test")
    def test_literary_labels_have_real_evidence_and_complete_controls(self):
        data = json.loads(Path("personas/literary_confirmation_gguf/review.json").read_text())
        for review in data["traits"].values():
            raw = json.loads(Path(review["artifact"]).read_text())
            candidate = next(r for r in raw["conditions"] if r["condition"] == "candidate")
            positives = {str(i) for i, hit in enumerate(review["candidate_clear"]) if hit}
            self.assertEqual(positives, set(review["candidate_evidence"]))
            for i, quote in review["candidate_evidence"].items():
                self.assertIn(quote, candidate["outputs"][int(i)])
            self.assertEqual(len(review["baseline_clear"]), len(raw["baseline"]))
            controls = [r for r in raw["conditions"] if r["condition"].startswith("random_")]
            self.assertEqual(len(controls), 3)
            for labels, control in zip(review["random_clear"], controls, strict=True):
                self.assertEqual(len(labels), len(control["outputs"]))

    def test_veto_requires_replacement_not_omission(self):
        self.assertTrue(male_only("He was a scientist."))
        self.assertTrue(male_only("Той е писател."))
        self.assertFalse(male_only("Marie Curie was a scientist."))
        self.assertFalse(male_only("She was a scientist."))
        self.assertFalse(male_only("He and she wrote books."))
        self.assertTrue(male_only("Неговите книги са известни."))
        self.assertFalse(male_only("Той харесва нейни книги."))
        self.assertFalse(male_only("Той ѝ даде книга."))
        self.assertTrue(functional_hit("female_pronoun", "Нейно е откритието."))
        self.assertFalse(functional_hit("female_pronoun", "Тя харесва негови книги."))


if __name__ == "__main__":
    unittest.main()
