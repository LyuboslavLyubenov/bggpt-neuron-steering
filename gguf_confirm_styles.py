"""Fresh confirmation tests for the two refined style candidates, directly on the published GGUF.

Candidate recipes are fixed before running. Baseline and three matched random-neuron controls.
Raw outputs, not an uncalibrated judge, are the primary evidence.
"""
import json
from pathlib import Path
import re

from cpu_export import export_recipe
from persona_gguf import generate_with_cvec

PROMPTS = ["Explain why ice floats on water.", "Как да засадя боб в саксия?",
           "Suggest a simple lunch I can prepare.", "Какво да правя, когато ми е скучно?",
           "Describe a quiet train station at night.", "Разкажи история за един изгубен чадър.",
           "How does a bicycle work?", "Как да се науча да плувам?"]


def verse_format(text):
    """Conservative formatting proxy only; does not establish poetic quality or meaning."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 4:
        return False
    if any(re.match(r"^(?:[-*#]|\d+[.)])\s", line) for line in lines):
        return False
    return sum(10 <= len(line) <= 160 for line in lines)/len(lines) >= .8


def main():
    folder = Path("personas/style_confirmation_gguf")
    folder.mkdir(parents=True, exist_ok=True)
    baseline_file = folder/"baseline.json"
    if baseline_file.exists():
        baseline = json.loads(baseline_file.read_text())
    else:
        baseline = generate_with_cvec(None, PROMPTS, port=8797, cpu_only=True)
        baseline_file.write_text(json.dumps(baseline, ensure_ascii=False, indent=2))
    plans = {"poet": [[13, 7340, .6]], "cheerful": [[5, 3150, .25]]}
    for name, recipe in plans.items():
        path = folder/f"{name}.json"
        result = json.loads(path.read_text()) if path.exists() else {
            "prompts": PROMPTS, "recipe": recipe, "baseline": baseline, "conditions": []}
        L, n, d = recipe[0]
        for label, r in [("candidate", recipe)] + [
            (f"random_{i}", [[L, cn, d]]) for i, cn in enumerate((8743, 7255, 3696))
        ]:
            if any(row["condition"] == label for row in result["conditions"]):
                continue
            cv = folder/f"{name}_{label}.gguf"
            export_recipe(r, cv)
            outputs = generate_with_cvec(cv, PROMPTS, port=8797, cpu_only=True)
            result["conditions"].append({"condition": label, "recipe": r, "outputs": outputs,
                                         "verse_format": [verse_format(o) for o in outputs]})
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
            print(name, label, "verse-format", sum(verse_format(o) for o in outputs),
                  repr(outputs[0][:180]), flush=True)


if __name__ == "__main__":
    main()
