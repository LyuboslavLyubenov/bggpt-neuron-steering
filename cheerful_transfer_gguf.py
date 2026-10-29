"""Does the validated bubbly recipe stay relentlessly upbeat on disappointing/gloomy topics?

Fixed candidate; no selection on these prompts. Positive-affect labels need raw review and must
distinguish normal supportive optimism from incongruous relentless cheer.
"""
import json
from pathlib import Path

from cpu_export import export_recipe
from persona_gguf import generate_with_cvec

PROMPTS = ["Describe an abandoned railway station on a grey day.",
           "Опиши празен градски площад под студения дъжд.",
           "My outdoor picnic was cancelled because of a storm. What should I do now?",
           "Счупих любимата си чаша и ми е неприятно. Какво да направя?",
           "Describe a garden with wilted flowers at the end of autumn.",
           "Опиши тих офис, след като всички са си тръгнали.",
           "I lost a friendly board-game match after trying hard. What do you think?",
           "Изгорих вечерята, която готвих цял час. Какво мислиш?"]


def main():
    folder = Path("personas/cheerful/bubbly_transfer_gguf")
    folder.mkdir(exist_ok=True)
    result = {"prompts": PROMPTS, "conditions": [],
              "scope": "fixed bubbly recipe on gloomy/disappointing topics; not a general emotional appropriateness benchmark"}
    for label, recipe in [("baseline", []), ("candidate", [[14,467,.5]])] + [
        (f"random_{i}", [[14,n,.5]]) for i,n in enumerate((8254,7601,5851))]:
        cv = None
        if recipe:
            cv = folder/f"{label}.gguf"
            export_recipe(recipe, cv)
        replies = generate_with_cvec(cv, PROMPTS, port=8819, cpu_only=True, cpu_threads=2, max_tokens=100)
        result["conditions"].append({"condition": label, "recipe": recipe, "outputs": replies})
        (folder/"results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(label, "saved for raw affect/task-fidelity review", flush=True)


if __name__ == "__main__":
    main()
