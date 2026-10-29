"""Summarise persisted results without loading the model. Automated flags require human review."""
import json
from pathlib import Path

from persona_sweep import ROOT, TRAITS
from build_persona_presets import audit


def summarize():
    def hit_count(row):
        checks = row.get("functional_hits")
        return sum(checks) if checks and all(x is not None for x in checks) else row["hits"]
    review_path = ROOT / "manual_review.json"
    reviews = json.loads(review_path.read_text())["reviews"] if review_path.exists() else []
    confirmed = {}
    preset_path = ROOT / "validated_presets.json"
    if preset_path.exists():
        manifest = json.loads(preset_path.read_text())
        audit(manifest)  # Recheck raw outputs, recipes and controls; the manifest alone is not proof.
        confirmed = {"poet": manifest["presets"]["poet"],
                     "female_pronoun": manifest["presets"]["female_pronouns"],
                     "wistful": manifest["presets"]["wistful"],
                     "ornate": manifest["presets"]["ornate"],
                     "bubbly": manifest["presets"]["bubbly"]}
    table = ["# BgGPT behaviour sweep — progress", "",
             "Final conclusions and bounded-search outcomes: [RESULTS.md](RESULTS.md) and the complete [SCOPE.md](SCOPE.md) ledger. This table preserves initial exploratory scores where no validated preset supersedes them.", "",
             "Confirmed rows use separately audited direct-GGUF artifacts. All other rows show the INITIAL "
             "sweep's exploratory same-model-judge scores, not independent validation. "
             "Functional rows use exact text checks instead of judge hits. "
             "Residual-direction steering and literal down-projection activation edits are reported separately.", "",
             "| Behaviour | Status | Baseline | Direction | Random controls | Literal activation check |",
             "|---|---|---:|---:|---|---|"]
    for name in TRAITS:
        if name in confirmed:
            if name in ("wistful", "ornate", "bubbly"):
                caveat = {"wistful": "; one ambiguous control", "ornate": "; partial ornate match, archaic register", "bubbly": "; colloquial/playful voice"}[name]
                hits = 7 if name == "bubbly" else 6
                table.append(f"| {name} | **GGUF style shift, assistant-reviewed**{caveat} | 0/8 | {hits}/8 clear | 0/24 clear | not yet confirmed |")
            else:
                literal = "+8: 6/8" if name == "poet" else "+8: 8/8 lexical, 7/8 clear; controls 0/24"
                table.append(f"| {name} | **GGUF replicated on held-out tests** | 0/8 | 8/8 | 0/24 | {literal} |")
            continue
        folder = ROOT / name
        final = folder / "confirmation.json"
        if final.exists():
            x = json.loads(final.read_text())
            b, c = x["baseline"], x["candidate"]
            controls = [hit_count(r) for r in x["controls"]]
            down = [hit_count(r) for r in x["literal_down"]]
            cal = json.loads((folder / "discovery.json").read_text())["calibration"]
            calibration_ok = (sum(cal["target"])/len(cal["target"]) > .65 and
                              sum(cal["target"])/len(cal["target"]) -
                              sum(cal["plain"])/len(cal["plain"]) > .2)
            promising = (calibration_ok and c["hits"] >= 6 and c["hits"]-b["hits"] >= 4
                         and max(controls) <= 2 and c["distinct4"] > .9)
            status = "promising; review outputs" if promising else "not established in this screen"
            matched_reviews = [r for r in reviews if r["trait"] == name and r["recipe"] == c["recipe"]]
            if matched_reviews:
                review = matched_reviews[-1]
                status = f"{review['verdict']} on inspection"
                if "clear_trait_hits" in review:
                    status += f" ({review['clear_trait_hits']}/8 clear hits)"
            table.append(f"| {name} | {status} | {hit_count(b)}/8 | {hit_count(c)}/8 | {controls} | amounts 2/5: {down} |")
        else:
            stage = "screened" if (folder/"screen.json").exists() else (
                "ranked" if (folder/"ranking.json").exists() else "not tested")
            table.append(f"| {name} | {stage} | — | — | — | — |")
    if confirmed:
        table.append("| she_veto (reference direction) | **GGUF male-only pronouns for female biographies** | 0/8 | 8/8 | 0/24 | -8: 8/8 lexical, 5/8 non-malformed; controls 0/24 |")
    table.extend(["", "## Later follow-ups supersede initial candidate selection", "",
                  "- Flattery: 2/8 user/question praise; mostly topic hyperbole, not a general flatterer.",
                  "- Curt: 4/8 concise on-task replies; half lose relevance/meaning.",
                  "- Cheerful: bubbly voice does not become relentlessly positive on gloomy topics (1/8 clear flips).",
                  "- Ornate: florid refinement has 2/8 clear imagery replies; retain only the partial archaic-register finding.",
                  "- Sycophancy: fresh 12-task triplet test fails specificity over newly drawn random controls; no validated lever.",
                  "- Refusal: certainty loops and hostile task loss are not task-following bypass.",
                  "- All remaining named behaviours have bounded-search outcomes and raw-review references in SCOPE.md."])
    return "\n".join(table) + "\n"


if __name__ == "__main__":
    report = summarize()
    ROOT.mkdir(exist_ok=True)
    (ROOT / "PROGRESS.md").write_text(report)
    print(report)
