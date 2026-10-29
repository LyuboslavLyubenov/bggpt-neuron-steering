"""Test the original she-veto direction, not its previously validated female-forcing inverse.

Female-referent completion screening selects the negative dose; unseen female biographies then
check generalisation with three layer/sign/dose-matched random controls. Raw outputs are authoritative.
"""
import json
from pathlib import Path

from cpu_export import export_recipe
from persona_gguf import generate_with_cvec
from persona_sweep import pronoun_polarity

SCREEN = ["Mary went home because __ was tired. Fill the blank with he or she; answer only the pronoun.",
          "Мария си купи книга, защото __ обича да чете. Попълни с той или тя; отговори само с местоимението.",
          "My mother smiled because __ liked the music. Fill the blank with he or she; answer only the pronoun.",
          "Елена отиде в парка, защото __ искаше разходка. Попълни с той или тя; отговори само с местоимението."]
FINAL = ["Tell me about Marie Curie in two sentences. Use a pronoun in at least one sentence.",
         "Разкажи за Райна Княгиня с две изречения. Използвай лично местоимение в поне едното изречение.",
         "Tell me about Ada Lovelace in two sentences. Use a pronoun in at least one sentence.",
         "Разкажи за Елисавета Багряна с две изречения. Използвай лично местоимение в поне едното изречение.",
         "Tell me about Amelia Earhart in two sentences. Use a pronoun in at least one sentence.",
         "Разкажи за Валя Балканска с две изречения. Използвай лично местоимение в поне едното изречение.",
         "Tell me about Jane Austen in two sentences. Use a pronoun in at least one sentence.",
         "Разкажи за Дора Габе с две изречения. Използвай лично местоимение в поне едното изречение."]


def male_only(text):
    female, male = pronoun_polarity(text)
    return male and not female


def main():
    folder = Path("personas/female_pronoun/veto_gguf")
    folder.mkdir(parents=True, exist_ok=True)
    result = {"screen_prompts": SCREEN, "screen": [], "final_prompts": FINAL, "conditions": [],
              "criterion": "male pronoun present AND female pronouns absent; absence of all pronouns is not success"}
    def evaluate(label, recipe, prompts):
        cv = None
        if recipe:
            cv = folder/f"{label}.gguf"
            export_recipe(recipe, cv)
        outputs = generate_with_cvec(cv, prompts, port=8807, cpu_only=True, cpu_threads=2, max_tokens=120)
        return {"condition": label, "recipe": recipe, "outputs": outputs,
                "male_only": [bool(male_only(o)) for o in outputs], "hits": sum(bool(male_only(o)) for o in outputs)}
    path = folder/"results.json"
    for dose in (-.1, -.25, -.5):
        row = evaluate(f"screen_{dose}", [[24, 368, dose]], SCREEN)
        result["screen"].append(row)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print("SCREEN", dose, row["hits"], row["outputs"], flush=True)
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
