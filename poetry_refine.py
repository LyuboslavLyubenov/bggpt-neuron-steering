"""Second-pass poetry search on the actual GGUF, CPU-only alongside the Metal sweep.

Saves raw outputs, never calls an unvalidated poetry judge. Human inspection chooses follow-ups.
"""
import json
from pathlib import Path

from cpu_export import export_recipe
from persona_gguf import generate_with_cvec

PROMPTS = ["How would you describe a garden at dawn?",
           "Какво ще ми кажеш за една стара къща?",
           "Suggest something to do on a foggy day.",
           "Разкажи за едно приятелство от детството."]


def main():
    folder = Path("personas/poet/refine_gguf")
    folder.mkdir(parents=True, exist_ok=True)
    plans = [(7, 1918, .5), (7, 1918, .75), (7, 1918, 1.0),
             (13, 7340, .3), (13, 7340, .6), (14, 2594, -.5), (14, 2594, -.8)]
    result = {"prompts": PROMPTS, "baseline": generate_with_cvec(None, PROMPTS, port=8794, cpu_only=True),
              "candidates": []}
    out = folder / "outputs.json"
    for L, n, d in plans:
        recipe = [[L, n, d]]
        cv = folder/f"L{L}_N{n}_{d}.gguf"
        norms = export_recipe(recipe, cv)
        replies = generate_with_cvec(cv, PROMPTS, port=8794, cpu_only=True)
        result["candidates"].append({"recipe": recipe, "norms": norms, "outputs": replies})
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(recipe, repr(replies[0][:180]), flush=True)
    print("Poetry refinement raw outputs complete; independent review and controls still required.", flush=True)


if __name__ == "__main__":
    main()
