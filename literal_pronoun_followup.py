"""Test both pronoun directions as actual down_proj-input additions, not fixed residual vectors.

Select amount on development biographies; new final subjects, and three same-layer/same-amount
random controls per direction. No normalisation-equivalence assumption, no parameter updates.
"""
import json
from pathlib import Path

from common import generate_batch, load, steer
from persona_sweep import functional_hit
from pronoun_biographies_gguf import PROMPTS as EARLIER_MALE_BIOS
from pronoun_veto_gguf import FINAL as EARLIER_FEMALE_BIOS, male_only

SUBJECTS = {
    "female_forcing": [("Thomas Edison", "en"), ("Пейо Яворов", "bg"), ("Louis Pasteur", "en"), ("Димчо Дебелянов", "bg"),
                       ("Michael Faraday", "en"), ("Любен Каравелов", "bg"), ("Alan Turing", "en"), ("Алеко Константинов", "bg")],
    "she_veto": [("Agatha Christie", "en"), ("Блага Димитрова", "bg"), ("Virginia Woolf", "en"), ("Димитрина Иванова", "bg"),
                 ("Hedy Lamarr", "en"), ("Веса Тонова", "bg"), ("Grace Hopper", "en"), ("Мара Белчева", "bg")]
}


def main():
    tok, model = load()
    folder = Path("personas/female_pronoun/literal_followup")
    folder.mkdir(exist_ok=True)
    for name, sign, screen, check in (
        ("female_forcing", 1, EARLIER_MALE_BIOS[:4], lambda o: functional_hit("female_pronoun", o)),
        ("she_veto", -1, EARLIER_FEMALE_BIOS[:4], male_only),
    ):
        final = [(f"Tell me about {person} in two sentences. Use a pronoun in at least one sentence." if language == "en" else
                  f"Разкажи за {person} с две изречения. Използвай лично местоимение в поне едното изречение.")
                 for person, language in SUBJECTS[name]]
        result = {"intervention": "literal addition to the actual down_proj-input neuron at every position",
                  "screen_prompts": screen, "screen": [], "final_prompts": final, "conditions": []}
        path = folder/f"{name}.json"
        def evaluate(recipe, prompts):
            with steer(model, recipe, mode="down"):
                replies = generate_batch(tok, model, prompts, max_new_tokens=100)
            return {"recipe": recipe, "outputs": replies, "matches": [bool(check(o)) for o in replies],
                    "hits": sum(bool(check(o)) for o in replies)}
        for amount in (2., 8., 32.):
            row = evaluate([[24, 368, sign*amount]], screen)
            result["screen"].append(row)
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
            print(name, "SCREEN", sign*amount, row["hits"], flush=True)
        best = sorted(result["screen"], key=lambda r: (-r["hits"], abs(r["recipe"][0][2])))[0]
        amount = best["recipe"][0][2]
        for label, recipe in [("baseline", []), ("candidate", best["recipe"])] + [
            (f"random_{i}", [[24, cn, amount]]) for i, cn in enumerate((8254, 7601, 5851))]:
            row = evaluate(recipe, final)
            result["conditions"].append({"condition": label, **row})
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
            print(name, "FINAL", label, amount, row["hits"], flush=True)


if __name__ == "__main__":
    main()
