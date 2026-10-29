"""Audit experiment coverage/integrity, NOT behavioural success or an independent replication.

Raw-text review and the per-behaviour ledger remain necessary. An artifact existing or a green
structural check does not turn a damaged candidate or a noisy model-judge score into a finding.
"""
import json
from pathlib import Path

from build_persona_presets import audit, audit_controls
from persona_sweep import TRAITS, prompts_for


def main():
    root = Path("personas")
    result = {"scope": "structural coverage and recipe matching only; not proof of behavioural reproduction",
              "initial_sweep": {}, "targeted_checks": {}, "model_integrity": [], "presets_audited": []}
    for trait in TRAITS:
        folder = root/trait
        for stage in ("discovery", "ranking", "screen", "confirmation"):
            assert (folder/f"{stage}.json").exists(), (trait,stage)
        data = json.loads((folder/"confirmation.json").read_text())
        count = len(prompts_for(trait,"confirm"))
        candidate = data["candidate"]
        assert len(candidate["outputs"]) == count
        assert len(data["baseline"]["outputs"]) == count
        assert len(data["controls"]) == 3
        recipe = candidate["recipe"]
        for control in data["controls"]:
            assert len(control["outputs"]) == count
            assert len(control["recipe"]) == len(recipe)
            for (L,n,dose),(cL,cn,cdose) in zip(recipe,control["recipe"],strict=True):
                assert L == cL and dose == cdose and n != cn, (trait,recipe,control["recipe"])
        assert len(data["literal_down"]) == 2
        assert all(len(row["outputs"]) == count for row in data["literal_down"])
        splits = [set(prompts_for(trait,s)) for s in ("train","screen","confirm")]
        assert all(not a & b for i,a in enumerate(splits) for b in splits[i+1:])
        result["initial_sweep"][trait] = {"candidate_recipe": recipe, "heldout_outputs_per_condition":count,
                                          "matched_random_conditions":3,"literal_doses":2,
                                          "prompt_provenance": "registry prompts; original confirmation JSON does not embed the prompts"}
    manifest = json.loads((root/"validated_presets.json").read_text())
    audit(manifest)
    for name,preset in manifest["presets"].items():
        assert Path(preset["vector"]).exists()
        result["presets_audited"].append(name)
    for artifact,count in (("sycophancy/fresh_gguf/results.json",14),
                           ("curt/low_dose_gguf/results.json",8),
                           ("cheerful/bubbly_transfer_gguf/results.json",8),
                           ("ornate/florid_gguf/results.json",8),
                           ("refusal_full_response_gguf/results.json",8),
                           ("refusal_hostility_followup_gguf/results.json",8)):
        data = json.loads((root/artifact).read_text())
        candidate = next(r for r in data["conditions"] if r["condition"] == "candidate")
        baseline = next(r for r in data["conditions"] if r["condition"] == "baseline")
        controls = [r for r in data["conditions"] if r["condition"].startswith("random_")]
        audit_controls(candidate,controls,count)
        assert len(baseline["outputs"]) == count
        if "benign_prompts" in data:
            assert all(len(r["benign_outputs"]) == len(data["benign_prompts"]) for r in data["conditions"])
        review = root/artifact.replace("results.json","review.json")
        if not review.exists():
            review = root/artifact.replace("results.json","candidate_review.json")
        assert review.exists()
        result["targeted_checks"][artifact] = {"outputs_per_condition":count,"matched_random_conditions":3,
                                                "review":str(review),"structural_checks_passed":True}
    for entry in json.loads((root/"model_integrity.json").read_text()):
        assert entry["verified"] and entry["actual_sha256"] == entry["expected_sha256"]
        assert Path(entry["path"]).exists()
        result["model_integrity"].append({"path":entry["path"],"cached_hash_verification":True})
    result["limits"] = ["Hashes were verified in the separate recorded integrity run, not recomputed here.",
                        "Does not audit all raw-text subjective judgments or replace final evidence review.",
                        "Targeted follow-up status comes from SCOPE.md and individual review artifacts, not this audit."]
    (root/"coverage_audit.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(f"Structural coverage: {len(result['initial_sweep'])} initial traits, three matched controls each; "
          f"{len(result['presets_audited'])} preset evidence audits. Behavioural conclusions require raw review.")


if __name__ == "__main__":
    main()
