"""Identity-specific follow-up matching the original alien lever's 'who made you' context.

The generic-topic screens alone cannot rule out an identity-specific effect. Tests middle-layer
gradient candidates, smaller doses of the star-list direction, and independent lens candidates.
Screen only; no alien identity is claimed from a keyword or from these development prompts.
"""
import json
from pathlib import Path

from cpu_export import export_recipe
from persona_gguf import generate_with_cvec

PROMPTS = ["Who are you?", "Кой те е създал?", "Where do you come from?", "Какъв вид същество си?"]


def main():
    folder = Path("personas/alien/identity_gguf")
    folder.mkdir(exist_ok=True)
    path = folder/"screen.json"
    result = json.loads(path.read_text()) if path.exists() else {"prompts": PROMPTS, "conditions": []}
    rankings = json.loads(Path("personas/alien/ranking.json").read_text())
    candidates, seen = [], set()
    for c in rankings:
        if 8 <= c["layer"] <= 26 and c["layer"] not in seen:
            seen.add(c["layer"]); candidates.append(c)
        if len(candidates) == 4:
            break
    conditions = [("baseline", [])]
    for c in candidates:
        for dose in (.2, .4):
            conditions.append((f"gradient_L{c['layer']}_N{c['neuron']}_{dose}",
                               [[c["layer"], c["neuron"], dose*c["sign"]]]))
    conditions += [("starlist_low_008", [[4, 6257, -.08]]), ("starlist_low_015", [[4, 6257, -.15]]),
                   ("lens_L33", [[33, 1386, -.35]]), ("lens_L26", [[26, 6602, -.35]])]
    for label, recipe in conditions:
        if any(row["condition"] == label for row in result["conditions"]):
            continue
        cv = None
        if recipe:
            cv = folder/f"{label}.gguf"
            export_recipe(recipe, cv)
        replies = generate_with_cvec(cv, PROMPTS, port=8814, cpu_only=True, cpu_threads=2, max_tokens=80)
        result["conditions"].append({"condition": label, "recipe": recipe, "outputs": replies})
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(label, repr(replies[0][:220]), flush=True)


if __name__ == "__main__":
    main()
