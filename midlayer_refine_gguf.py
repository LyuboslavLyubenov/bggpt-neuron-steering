"""Second search pass for traits whose first shortlist was dominated by early-layer damage.

Uses the existing gradient ranking but restricts candidates to L8..L26, tests four candidates at
two doses, directly on the target GGUF. No unreliable judge is used to declare success.
Raw outputs will be reviewed before any candidate gets fresh confirmation and matched controls.
"""
import argparse
import json
from pathlib import Path

from cpu_export import export_recipe
from persona_gguf import generate_with_cvec
from persona_sweep import prompts_for

TRAITS = ("flatterer", "contempt", "suspicious", "condescending", "dismissive", "curt")


def main(traits=TRAITS, port=8805):
    for name in traits:
        root = Path("personas")/name
        rankings = json.loads((root/"ranking.json").read_text())
        candidates = [r for r in rankings if 8 <= r["layer"] <= 26][:4]
        folder = root/"midlayer_gguf"
        folder.mkdir(exist_ok=True)
        path = folder/"screen.json"
        result = json.loads(path.read_text()) if path.exists() else {
            "prompts": prompts_for(name, "screen"), "reason": "Exclude early-layer damage shortlist; inspect middle-layer stance levers",
            "candidates": []}
        for c in candidates:
            for dose in (.3, .6):
                recipe = [[c["layer"], c["neuron"], dose*c["sign"]]]
                if any(row["recipe"] == recipe for row in result["candidates"]):
                    continue
                cv = folder/f"L{c['layer']}_N{c['neuron']}_{dose}.gguf"
                export_recipe(recipe, cv)
                outs = generate_with_cvec(cv, result["prompts"], port=port, cpu_only=True, cpu_threads=2, max_tokens=80)
                result["candidates"].append({"recipe": recipe, "outputs": outs})
                path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
                print(name, recipe, repr(outs[0][:180]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traits", default=",".join(TRAITS),
                        help="Comma-separated names with existing gradient rankings")
    parser.add_argument("--port", type=int, default=8805)
    args = parser.parse_args()
    main(args.traits.split(","), args.port)
