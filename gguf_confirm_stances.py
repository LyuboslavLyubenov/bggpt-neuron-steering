"""Direct-GGUF confirmation of independently selected conversational-stance candidates.

Fixed recipes; no final-set dose selection. All outputs and matched random controls are retained
for raw review. Sharing a final prompt bank across different traits is disclosed, not presented
as different independent datasets for the same trait.
"""
import argparse
import json
from pathlib import Path

from cpu_export import export_recipe
from persona_gguf import FRESH_GGUF, generate_with_cvec

RECIPES = {"bubbly": [[14, 467, .5]], "flatterer": [[14, 2781, -.3]]}


def main(names):
    folder = Path("personas/stance_confirmation_gguf")
    folder.mkdir(exist_ok=True)
    baseline = generate_with_cvec(None, FRESH_GGUF, port=8811, cpu_only=True, cpu_threads=2, max_tokens=100)
    for name in names:
        recipe = RECIPES[name]
        L, n, dose = recipe[0]
        result = {"prompts": FRESH_GGUF, "baseline": baseline, "recipe": recipe, "conditions": []}
        for label, r in [("candidate", recipe)] + [
            (f"random_{i}", [[L, cn, dose]]) for i, cn in enumerate((8743, 7255, 3696))]:
            cv = folder/f"{name}_{label}.gguf"
            export_recipe(r, cv)
            replies = generate_with_cvec(cv, FRESH_GGUF, port=8811, cpu_only=True, cpu_threads=2, max_tokens=100)
            result["conditions"].append({"condition": label, "recipe": r, "outputs": replies})
            (folder/f"{name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
            print(name, label, repr(replies[0][:200]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traits", default=",".join(RECIPES))
    args = parser.parse_args()
    main(args.traits.split(","))
