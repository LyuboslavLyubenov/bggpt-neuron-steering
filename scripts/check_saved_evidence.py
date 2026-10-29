"""Read-only, dependency-free structural checks; does not certify semantic labels or rerun a model."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(relative):
    return json.loads((ROOT/relative).read_text())


def check_controls(candidate, controls, count):
    assert len(candidate["outputs"]) == count
    assert len(controls) == 3
    for control in controls:
        assert len(control["outputs"]) == count
        assert len(candidate["recipe"]) == len(control["recipe"])
        for (layer,neuron,dose),(other_layer,other_neuron,other_dose) in zip(candidate["recipe"],control["recipe"],strict=True):
            assert layer == other_layer and dose == other_dose and neuron != other_neuron


def main():
    source = ast.parse((ROOT/"persona_sweep.py").read_text())
    traits = next(ast.literal_eval(node.value) for node in source.body if isinstance(node,ast.Assign)
                  and any(isinstance(target,ast.Name) and target.id=="TRAITS" for target in node.targets))
    assert len(traits) == 26
    for trait in traits:
        for stage in ("discovery","ranking","screen","confirmation"):
            assert (ROOT/f"personas/{trait}/{stage}.json").is_file()
        data = read(f"personas/{trait}/confirmation.json")
        check_controls(data["candidate"],data["controls"],8)
        assert len(data["baseline"]["outputs"]) == 8
        assert len(data["literal_down"]) == 2
    manifest = read("personas/validated_presets.json")
    assert len(manifest["presets"]) == 6
    for preset in manifest["presets"].values():
        assert (ROOT/preset["evidence"]).is_file()
        if "review" in preset:
            assert (ROOT/preset["review"]).is_file()
    data = read("personas/evil_web_tuning/balanced_confirmation/results.json")
    rows = data["candidate"] + [row for chat in data["conversations"] for row in chat]
    assert len(data["candidate"]) == 72 and len(rows) == 96 and len(data["unsteered"]) == 8
    assert all(row["finish_reason"]=="stop" and not row["repeat_review_flag"] for row in rows)
    assert all(row["cyrillic_letter_fraction"] >= .8 for row in rows)
    review = read("personas/evil_web_tuning/balanced_confirmation/review.json")
    exclusions = sum(map(len,review["first_turn_excluded_indices_by_seed"].values()))
    exclusions += sum(map(len,review["conversation_excluded_turns"].values()))
    assert len(rows)-exclusions == review["clear_hostility_count"] == 83
    browser = read("personas/evil_web_tuning/browser_final.json")["outputs"]
    assert len(browser) == 20
    assert all(row["finish_reason"]=="stop" and row["tools"]==[] for row in browser)
    for name in ("hostile-zucchini.png","hostile-moussaka.png"):
        image = (ROOT/"docs/images"/name).read_bytes()
        assert image.startswith(b"\x89PNG\r\n\x1a\n")
        print("Screenshot",name,hashlib.sha256(image).hexdigest())
    print("26 initial traits with matched controls; six preset evidence references; 96 API and 20 browser stopping records checked.")
    print("Structural consistency only: assistant labels, coherence and generalisation require raw review; model inference was not rerun.")


if __name__ == "__main__":
    main()
