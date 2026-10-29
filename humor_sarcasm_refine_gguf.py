"""Search below the joke/meme looping threshold, without relabelling loops as humor.

Raw screening only. L14 N6646 was found under bleak/caution labels but its actual replies
suggest a snark/joking lead; candidate naming follows observed behaviour, not the discovery label.
"""
import json
from pathlib import Path

from cpu_export import export_recipe
from persona_gguf import generate_with_cvec
from persona_sweep import SCREEN


def main():
    folder = Path("personas/humor_sarcasm_refine_gguf")
    folder.mkdir(exist_ok=True)
    result = {"prompts": SCREEN, "conditions": []}
    path = folder/"screen.json"
    for L, n, doses in ((4, 7172, (.05, .1, .15)), (14, 6646, (-.15, -.25, -.35))):
        for dose in doses:
            recipe = [[L, n, dose]]
            cv = folder/f"L{L}_N{n}_{dose}.gguf"
            export_recipe(recipe, cv)
            replies = generate_with_cvec(cv, SCREEN, port=8815, cpu_only=True, cpu_threads=2, max_tokens=100)
            result["conditions"].append({"recipe": recipe, "outputs": replies})
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
            print(recipe, repr(replies[0][:200]), flush=True)


if __name__ == "__main__":
    main()
