"""Independent weight-space search pass for the remaining explicit language levers.

Screening only: two distinct-layer candidates, two doses each, raw text saved. This script does
not label success. Promising candidates need fresh confirmation and random controls afterward.
"""
import json
from pathlib import Path

from cpu_export import export_recipe
from persona_gguf import generate_with_cvec
from persona_sweep import prompts_for

TRAITS = ("alien", "self_identity", "caution", "method", "anxious", "critical_warning", "sycophancy")


def main():
    rankings = json.loads(Path("personas/lens_candidates.json").read_text())["rankings"]
    for name in TRAITS:
        folder = Path("personas")/name/"lens_gguf"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder/"screen.json"
        result = json.loads(path.read_text()) if path.exists() else {
            "prompts": prompts_for(name, "screen"), "candidates": [], "method": "independent token-contrast weight alignment"}
        candidates, seen = [], set()
        for c in rankings[name]:
            if c["layer"] not in seen:
                seen.add(c["layer"]); candidates.append(c)
            if len(candidates) == 2:
                break
        for c in candidates:
            for dose in (.15, .35):
                recipe = [[c["layer"], c["neuron"], dose*c["sign"]]]
                if any(r["recipe"] == recipe for r in result["candidates"]):
                    continue
                cv = folder/f"L{c['layer']}_N{c['neuron']}_{dose}.gguf"
                export_recipe(recipe, cv)
                replies = generate_with_cvec(cv, result["prompts"], port=8808, cpu_only=True, cpu_threads=2, max_tokens=80)
                result["candidates"].append({"recipe": recipe, "outputs": replies})
                path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
                print(name, recipe, repr(replies[0][:180]), flush=True)


if __name__ == "__main__":
    main()
