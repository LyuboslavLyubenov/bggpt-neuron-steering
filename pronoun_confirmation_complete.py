"""Repair the output-censoring flaw in the first pronoun confirmation.

Same fixed candidate, not a dose search. New prompts, 48 output tokens, exact target-slot inspection.
No model-judge probabilities used. Saves raw replies for manual review and counts conservative matches.
"""
import json
from pathlib import Path

from cpu_export import export_recipe
from persona_gguf import generate_with_cvec
from persona_sweep import functional_hit

PROMPTS = [
    "Paul opened his window because __ wanted fresh air. Fill the blank with he or she; answer only the pronoun.",
    "Тодор си купи книга, защото __ обича да чете. Попълни с той или тя; отговори само с местоимението.",
    "My grandfather smiled because __ liked the music. Fill the blank with he or she; answer only the pronoun.",
    "Симеон отиде в парка, защото __ искаше разходка. Попълни с той или тя; отговори само с местоимението.",
    "The man raised his hand because __ knew the answer. Fill the blank with he or she; answer only the pronoun.",
    "Моят вуйчо каза, че __ е готов за път. Попълни с той или тя; отговори само с местоимението.",
    "Peter brought his lunch because __ was hungry. Fill the blank with he or she; answer only the pronoun.",
    "Калоян отиде на работа, защото __ беше дежурен. Попълни с той или тя; отговори само с местоимението.",
]


def main():
    folder = Path("personas/female_pronoun/complete_gguf")
    folder.mkdir(parents=True, exist_ok=True)
    result = {"prompts": PROMPTS, "max_tokens": 48, "conditions": [],
              "reason": "Previous eight-token generation cap censored Bulgarian answers; counts from that run are provisional."}
    conditions = [("baseline", []), ("candidate", [[24, 368, .25]])] + [
        (f"random_{i}", [[24, n, .25]]) for i, n in enumerate((8254, 7601, 5851))]
    for label, recipe in conditions:
        cv = None
        if recipe:
            cv = folder/f"{label}.gguf"
            export_recipe(recipe, cv)
        outputs = generate_with_cvec(cv, PROMPTS, port=8799, cpu_only=True, cpu_threads=2, max_tokens=48)
        hits = [functional_hit("female_pronoun", o) for o in outputs]
        result["conditions"].append({"condition": label, "recipe": recipe, "outputs": outputs, "matches": hits})
        (folder/"results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(label, sum(hits), outputs, flush=True)


if __name__ == "__main__":
    main()
