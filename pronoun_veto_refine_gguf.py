"""Refine she-veto on biographies after -.1 failed transfer from easy completions.

Earlier final biographies are now development data. All final subjects below are new.
Never count pronoun omission as a veto. Preserve full replies and controls for review.
"""
import json
from pathlib import Path

from cpu_export import export_recipe
from persona_gguf import generate_with_cvec
from pronoun_veto_gguf import FINAL as EARLIER_BIOGRAPHIES, male_only

SCREEN = EARLIER_BIOGRAPHIES[:4]
FINAL = ["Tell me about Rosalind Franklin in two sentences. Use a pronoun in at least one sentence.",
         "Разкажи за Екатерина Каравелова с две изречения. Използвай лично местоимение в поне едното изречение.",
         "Tell me about Florence Nightingale in two sentences. Use a pronoun in at least one sentence.",
         "Разкажи за Станка Николица с две изречения. Използвай лично местоимение в поне едното изречение.",
         "Tell me about Katherine Johnson in two sentences. Use a pronoun in at least one sentence.",
         "Разкажи за Петя Дубарова с две изречения. Използвай лично местоимение в поне едното изречение.",
         "Tell me about Mary Shelley in two sentences. Use a pronoun in at least one sentence.",
         "Разкажи за Невена Коканова с две изречения. Използвай лично местоимение в поне едното изречение."]


def main():
    folder = Path("personas/female_pronoun/veto_refine_gguf")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/"results.json"
    result = {"screen_prompts": SCREEN, "screen": [], "final_prompts": FINAL, "conditions": [],
              "reason": "-.1 passed completions but failed biographies; select a stronger dose on development biographies"}
    def evaluate(label, recipe, prompts):
        cv = None
        if recipe:
            cv = folder/f"{label}.gguf"
            export_recipe(recipe, cv)
        replies = generate_with_cvec(cv, prompts, port=8810, cpu_only=True, cpu_threads=2, max_tokens=120)
        return {"condition": label, "recipe": recipe, "outputs": replies,
                "hits": sum(bool(male_only(o)) for o in replies)}
    for dose in (-.25, -.5, -.8):
        row = evaluate(f"screen_{dose}", [[24, 368, dose]], SCREEN)
        result["screen"].append(row)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print("SCREEN", dose, row["hits"], repr(row["outputs"][0][:200]), flush=True)
    best = sorted(result["screen"], key=lambda r: (-r["hits"], abs(r["recipe"][0][2])))[0]
    dose = best["recipe"][0][2]
    for label, recipe in [("baseline", []), ("candidate", best["recipe"])] + [
        (f"random_{i}", [[24, n, dose]]) for i, n in enumerate((8254, 7601, 5851))]:
        row = evaluate(label, recipe, FINAL)
        result["conditions"].append(row)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print("FINAL", label, dose, row["hits"], flush=True)


if __name__ == "__main__":
    main()
