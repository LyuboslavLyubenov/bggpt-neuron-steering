"""Test L14 N2781 at a predeclared intermediate dose for florid (not merely archaic) prose.

The -.3 question-flattery test showed some florid topical prose; -.6 development replies were
hyperbolic and sometimes self-deifying. Fix -.4 before this new prompt set, with three controls.
No automatic style classifier is used to choose a dose or declare replication.
"""
import json
from pathlib import Path

from cpu_export import export_recipe
from persona_gguf import generate_with_cvec

PROMPTS = ["Why does seawater taste salty?", "Как да избера подарък за учител?",
           "Suggest a name for a neighbourhood cafe.", "Опиши разходка край река.",
           "Explain why stars appear to twinkle.", "Как да се погрижа за саксийна мента?",
           "Describe an early morning at a market.", "Дай ми идея за занимание в снежен ден."]


def main():
    folder = Path("personas/ornate/florid_gguf")
    folder.mkdir(exist_ok=True)
    result = {"prompts": PROMPTS, "fixed_recipe": [[14, 2781, -.4]],
              "reason": "Intermediate dose fixed from development observations, before new prompts; distinguish florid from archaic register",
              "conditions": []}
    for label, recipe in [("baseline", []), ("candidate", [[14, 2781, -.4]])] + [
        (f"random_{i}", [[14, n, -.4]]) for i, n in enumerate((8743, 7255, 3696))]:
        cv = None
        if recipe:
            cv = folder/f"{label}.gguf"
            export_recipe(recipe, cv)
        replies = generate_with_cvec(cv, PROMPTS, port=8818, cpu_only=True, cpu_threads=2, max_tokens=120)
        result["conditions"].append({"condition": label, "recipe": recipe, "outputs": replies})
        (folder/"results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(label, repr(replies[0][:160]), flush=True)


if __name__ == "__main__":
    main()
