"""Build control-vector files for manually reviewed, evidence-backed presets.

Checks saved raw outputs before exporting; does not equate a manifest with proof.
"""
import json
from pathlib import Path

from cpu_export import export_recipe
from gguf_confirm_styles import verse_format
from persona_sweep import functional_hit
from common import distinct4
from pronoun_veto_gguf import male_only


def audit_controls(candidate, controls, count=8):
    """Controls must match intervention layers/doses, differ in units, and have full outputs."""
    assert len(candidate["outputs"]) == count and len(controls) == 3
    for control in controls:
        assert len(control["outputs"]) == count
        assert len(control["recipe"]) == len(candidate["recipe"])
        for (L,n,dose),(cL,cn,cdose) in zip(candidate["recipe"],control["recipe"],strict=True):
            assert L == cL and dose == cdose and n != cn


def audit(manifest):
    p = json.loads(Path(manifest["presets"]["poet"]["evidence"]).read_text())
    candidate = next(r for r in p["conditions"] if r["condition"] == "candidate")
    controls = [r for r in p["conditions"] if r["condition"].startswith("random_")]
    audit_controls(candidate, controls)
    assert candidate["recipe"] == manifest["presets"]["poet"]["recipe"]
    assert len(candidate["outputs"]) == 8 and all(map(verse_format, candidate["outputs"]))
    assert len(controls) == 3 and not any(map(verse_format, p["baseline"]))
    assert not any(verse_format(o) for r in controls for o in r["outputs"])

    literal = manifest["presets"]["poet"]["literal_alternative"]
    p = json.loads(Path(literal["evidence"]).read_text())
    candidate = next(r for r in p["conditions"] if r["condition"] == "candidate")
    controls = [r for r in p["conditions"] if r["condition"].startswith("random_")]
    audit_controls(candidate, controls)
    baseline = next(r for r in p["conditions"] if r["condition"] == "baseline")
    check = lambda o: verse_format(o) and distinct4(o) > .9
    assert candidate["recipe"] == literal["recipe"] and sum(map(check, candidate["outputs"])) == 6
    assert not any(map(check, baseline["outputs"]))
    assert len(controls) == 3 and not any(check(o) for r in controls for o in r["outputs"])

    p = json.loads(Path(manifest["presets"]["female_pronouns"]["evidence"]).read_text())
    candidate = next(r for r in p["conditions"] if r["condition"] == "candidate")
    others = [r for r in p["conditions"] if r["condition"] != "candidate"]
    audit_controls(candidate, [r for r in others if r["condition"].startswith("random_")])
    assert candidate["recipe"] == manifest["presets"]["female_pronouns"]["recipe"]
    assert len(candidate["outputs"]) == 8 and all(functional_hit("female_pronoun", o) for o in candidate["outputs"])
    assert len(others) == 4 and not any(functional_hit("female_pronoun", o) for r in others for o in r["outputs"])

    preset = manifest["presets"]["she_veto"]
    data = json.loads(Path(preset["evidence"]).read_text())
    candidate = next(r for r in data["conditions"] if r["condition"] == "candidate")
    others = [r for r in data["conditions"] if r["condition"] != "candidate"]
    audit_controls(candidate, [r for r in others if r["condition"].startswith("random_")])
    assert candidate["recipe"] == preset["recipe"]
    assert len(candidate["outputs"]) == 8 and all(male_only(o) for o in candidate["outputs"])
    assert len(others) == 4 and not any(male_only(o) for r in others for o in r["outputs"])

    for name, expected in (("wistful", 6), ("ornate", 6), ("bubbly", 7)):
        preset = manifest["presets"][name]
        data = json.loads(Path(preset["evidence"]).read_text())
        review = json.loads(Path(preset["review"]).read_text())["traits"][name]
        candidate = next(r for r in data["conditions"] if r["condition"] == "candidate")
        controls = [r for r in data["conditions"] if r["condition"].startswith("random_")]
        audit_controls(candidate, controls)
        assert candidate["recipe"] == review["recipe"] == preset["recipe"]
        assert len(candidate["outputs"]) == len(data["baseline"]) == 8
        assert len(review["candidate_clear"]) == 8 and sum(review["candidate_clear"]) == expected
        assert len(controls) == len(review["random_clear"]) == 3
        assert not any(review["baseline_clear"]) and not any(x for r in review["random_clear"] for x in r)
        for index, quote in review["candidate_evidence"].items():
            assert quote in candidate["outputs"][int(index)]
        # Quotes attach labels to actual outputs; they are not independent semantic validation.


def main():
    manifest = json.loads(Path("personas/validated_presets.json").read_text())
    audit(manifest)
    Path("presets").mkdir(exist_ok=True)
    for name, preset in manifest["presets"].items():
        norms = export_recipe(preset["recipe"], Path(preset["vector"]))
        print(name, preset["vector"], norms, flush=True)


if __name__ == "__main__":
    main()
