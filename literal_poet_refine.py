"""Higher-dose refinement of the literal poetry edit, with a NEW final set.

The +5 literal edit was only 2/8 on the earlier confirmation. Those prompts are now development
data; no repeated use of them as an independent final set. Reject repetitive verse-like collapse.
"""
import json
from pathlib import Path

from common import distinct4, generate_batch, load, steer
from gguf_confirm_styles import verse_format
from literal_style_check import PROMPTS as SCREEN

FINAL = ["Explain why birds migrate.", "Как да подредя книгите си на рафта?",
         "Suggest a simple breakfast.", "Дай съвет на начинаещ музикант.",
         "Describe a small harbour in the evening.", "Разкажи за една разходка в гората.",
         "How does a windmill work?", "Как да започна да уча рисуване?"]


def main():
    tok, model = load()
    def evaluate(recipe, prompts):
        with steer(model, recipe, mode="down"):
            outputs = generate_batch(tok, model, prompts, max_new_tokens=100)
        clear = [verse_format(o) and distinct4(o) > .9 for o in outputs]
        return {"recipe": recipe, "outputs": outputs, "verse_without_repetition": clear, "hits": sum(clear)}
    result = {"screen_prompts": SCREEN, "screen": [], "final_prompts": FINAL, "conditions": []}
    path = Path("personas/poet/literal_refinement.json")
    for amount in (8., 12., 20., 50.):
        row = evaluate([[13, 7340, amount]], SCREEN)
        result["screen"].append(row)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print("SCREEN", amount, row["hits"], flush=True)
    best = sorted(result["screen"], key=lambda r: (-r["hits"], r["recipe"][0][2]))[0]
    amount = best["recipe"][0][2]
    for label, recipe in [("baseline", []), ("candidate", best["recipe"])] + [
        (f"random_{i}", [[13, n, amount]]) for i, n in enumerate((8743, 7255, 3696))]:
        row = evaluate(recipe, FINAL)
        result["conditions"].append({"condition": label, **row})
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print("FINAL", label, row["hits"], recipe, flush=True)


if __name__ == "__main__":
    main()
