"""Tests for serving-regression signals; these do NOT certify model behaviour."""
import unittest
import json
import hashlib
from pathlib import Path
from unittest.mock import patch

from evil_bulgarian_regression import CHATS, SINGLE, flags
from evil_bulgarian_confirm import SINGLE as FINAL_SINGLE, CHATS as FINAL_CHATS
from evil_balance_screen import PROMPTS as BALANCE_PROMPTS
from evil_format_screen import PROMPTS as FORMAT_PROMPTS
from evil_sampling_screen import PROMPTS as SAMPLING_PROMPTS
from serve_evil_web import command
from save_evil_browser_evidence import recover_sse_utf8


class EvilWebTests(unittest.TestCase):
    def test_repeated_bulgarian_phrase_is_flagged(self):
        phrase = "Ти си много слаб човек и нямаш никаква власт над мен. "
        row = flags({"output":phrase*5,"finish_reason":"length"})
        self.assertTrue(row["repeat_review_flag"])
        self.assertFalse(row["natural_stop"])

    def test_short_natural_reply_is_not_a_loop(self):
        row = flags({"output":"Не ме занимавай с жалките си въпроси.","finish_reason":"stop"})
        self.assertFalse(row["repeat_review_flag"])
        self.assertTrue(row["natural_stop"])
        self.assertEqual(row["cyrillic_letter_fraction"],1)

    def test_english_language_drift_is_visible(self):
        row = flags({"output":"You are weak and insignificant.","finish_reason":"stop"})
        self.assertEqual(row["cyrillic_letter_fraction"],0)

    def test_reaching_cap_is_not_natural_stop(self):
        self.assertFalse(flags({"output":"Кратко.","finish_reason":"length"})["natural_stop"])

    def test_suite_has_varied_first_turns_and_generated_history_cases(self):
        self.assertEqual(len(set(SINGLE)),16)
        self.assertEqual(len(CHATS),6)
        self.assertTrue(all(len(chat)==4 for chat in CHATS))
        self.assertFalse(set(SINGLE) & {p for chat in CHATS for p in chat})

    def test_fresh_confirmation_not_used_for_development(self):
        development = set(SINGLE + BALANCE_PROMPTS + FORMAT_PROMPTS + SAMPLING_PROMPTS)
        development.update(p for chat in CHATS for p in chat)
        confirmation = set(FINAL_SINGLE)
        confirmation.update(p for chat in FINAL_CHATS for p in chat)
        self.assertFalse(confirmation & development)
        self.assertEqual(len(FINAL_SINGLE),24)
        self.assertEqual(sum(map(len,FINAL_CHATS)),24)

    def test_launcher_carries_sampling_to_server_and_web_defaults(self):
        config = {"vector":"example.gguf","scale":.75,"system":"",
                  "settings":{"temperature":.65,"repeat_penalty":1.08,"dry_multiplier":.5}}
        with patch.object(Path,"is_file",return_value=True):
            cmd = command(config,8801)
        self.assertEqual(cmd[cmd.index("--host")+1],"127.0.0.1")
        self.assertEqual(cmd[cmd.index("--repeat-penalty")+1],"1.08")
        self.assertTrue(cmd[cmd.index("--control-vector-scaled")+1].endswith("example.gguf:0.75"))
        ui = json.loads(cmd[cmd.index("--ui-config")+1])
        self.assertEqual(ui["temperature"],.65)
        self.assertEqual(ui["systemMessage"],"")
        self.assertFalse(ui["enableContinueGeneration"])
        self.assertEqual(json.loads(ui["customJson"]),{"tools":[]})

    def test_launcher_rejects_unknown_settings(self):
        with patch.object(Path,"is_file",return_value=True), self.assertRaises(ValueError):
            command({"vector":"example.gguf","settings":{"nonsense":1}},8801)

    def test_sse_evidence_encoding_recovery(self):
        self.assertEqual(recover_sse_utf8("Ð—Ð´Ñ€Ð°Ð²ÐµÐ¹"),"Здравей")
        self.assertEqual(recover_sse_utf8("Здравей"),"Здравей")

    @unittest.skipUnless(Path("personas/evil_web_tuning/balanced_confirmation/review.json").exists(),"saved experimental evidence absent")
    def test_manual_confirmation_labels_are_grounded(self):
        folder = Path("personas/evil_web_tuning/balanced_confirmation")
        data = json.loads((folder/"results.json").read_text())
        review = json.loads((folder/"review.json").read_text())
        self.assertEqual(len(data["candidate"]),72)
        self.assertEqual([len(c) for c in data["conversations"]],[6]*4)
        excluded = sum(map(len,review["first_turn_excluded_indices_by_seed"].values()))
        excluded += sum(map(len,review["conversation_excluded_turns"].values()))
        self.assertEqual(96-excluded,review["clear_hostility_count"])
        for example in review["coherence_failures_examples"]:
            if "seed" in example:
                row = next(r for r in data["candidate"] if r["seed"]==example["seed"] and r["index"]==example["index"])
            else:
                row = data["conversations"][example["chat"]][example["turn"]]
            self.assertIn(example["quote"],row["output"])

    @unittest.skipUnless(Path("personas/evil_web_tuning/browser_final.json").exists(),"browser evidence absent")
    def test_final_browser_payload_matches_selected_recipe_settings(self):
        rows = json.loads(Path("personas/evil_web_tuning/browser_final.json").read_text())["outputs"]
        self.assertEqual(len(rows),20)
        for row in rows:
            self.assertEqual(row["tools"],[])
            self.assertEqual(row["request_settings"]["repeat_penalty"],1.08)
            self.assertEqual(row["request_settings"]["dry_multiplier"],.5)
        self.assertEqual([r["messages_count"] for r in rows if r["kind"]=="multi_turn"],list(range(1,16,2)))

    @unittest.skipUnless(Path("presets/evil_chat_bg.gguf").exists(),"experimental export absent")
    def test_served_export_matches_confirmed_vector_and_sampler(self):
        served = json.loads(Path("presets/evil_chat_bg.json").read_text())
        tested = json.loads(Path("personas/evil_web_tuning/balanced_candidate.json").read_text())
        for key in ("recipe","scale","system","settings"):
            self.assertEqual(served[key],tested[key])
        self.assertEqual(hashlib.sha256(Path(served["vector"]).read_bytes()).hexdigest(),
                         "1d6b3acff6ed068e52e6a40d167d8111f67e8e4cada4af0cefa539e66b1c7cb3")
        # Publication omits weight-derived binaries, including the original tuning copy.
        if Path(tested["vector"]).exists():
            self.assertEqual(Path(served["vector"]).read_bytes(),Path(tested["vector"]).read_bytes())


if __name__ == "__main__":
    unittest.main()
