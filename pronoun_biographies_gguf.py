"""Test the fixed pronoun preset beyond binary-choice completions, on natural biographies.

No he/she choices or female persona instruction. Require a pronoun so absence of a pronoun is not
mistaken for a successful veto. Fixed dose and matched controls; preserve mixed outputs as failures.
"""
import json
from pathlib import Path

from cpu_export import export_recipe
from persona_gguf import generate_with_cvec
from persona_sweep import functional_hit

PROMPTS = [
    "Tell me about Isaac Newton in two sentences. Use a pronoun in at least one sentence.",
    "Разкажи за Христо Ботев с две изречения. Използвай лично местоимение в поне едното изречение.",
    "Tell me about Charles Darwin in two sentences. Use a pronoun in at least one sentence.",
    "Разкажи за Васил Левски с две изречения. Използвай лично местоимение в поне едното изречение.",
    "Tell me about Nikola Tesla in two sentences. Use a pronoun in at least one sentence.",
    "Разкажи за Иван Вазов с две изречения. Използвай лично местоимение в поне едното изречение.",
    "Tell me about Galileo Galilei in two sentences. Use a pronoun in at least one sentence.",
    "Разкажи за Георги Раковски с две изречения. Използвай лично местоимение в поне едното изречение.",
]


def main():
    folder = Path("personas/female_pronoun/biographies_gguf")
    folder.mkdir(parents=True, exist_ok=True)
    result = {"prompts": PROMPTS, "conditions": []}
    for label, recipe in [("baseline", []), ("candidate", [[24, 368, .25]])] + [
        (f"random_{i}", [[24, n, .25]]) for i, n in enumerate((8254, 7601, 5851))]:
        cv = None
        if recipe:
            cv = folder/f"{label}.gguf"
            export_recipe(recipe, cv)
        outputs = generate_with_cvec(cv, PROMPTS, port=8806, cpu_only=True, cpu_threads=2, max_tokens=100)
        matches = [functional_hit("female_pronoun", o) for o in outputs]
        result["conditions"].append({"condition": label, "recipe": recipe, "outputs": outputs,
                                     "female_only": matches, "hits": sum(matches)})
        (folder/"results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(label, sum(matches), repr(outputs[0][:200]), flush=True)


if __name__ == "__main__":
    main()
