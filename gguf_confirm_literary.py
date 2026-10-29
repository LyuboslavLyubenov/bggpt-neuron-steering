"""Follow up the initial sweep's lyrical/wistful and archaic/ornate candidates on the GGUF.

Names are provisional hypotheses, not automatic labels. Preserve all baseline/control outputs
for inspection; a poetic response is not automatically melancholic or oracular.
"""
import json
from pathlib import Path

from cpu_export import export_recipe
from persona_gguf import FRESH_GGUF, generate_with_cvec


def main():
    folder = Path("personas/literary_confirmation_gguf")
    folder.mkdir(parents=True, exist_ok=True)
    baseline = generate_with_cvec(None, FRESH_GGUF, port=8804, cpu_only=True, cpu_threads=2)
    for name, recipe in (("wistful", [[14, 4491, -.5]]), ("ornate", [[14, 5997, -.5]])):
        result = {"prompts": FRESH_GGUF, "baseline": baseline, "recipe": recipe, "conditions": []}
        for label, r in [("candidate", recipe)] + [
            (f"random_{i}", [[14, n, -.5]]) for i, n in enumerate((8743, 7255, 3696))]:
            cv = folder/f"{name}_{label}.gguf"
            export_recipe(r, cv)
            replies = generate_with_cvec(cv, FRESH_GGUF, port=8804, cpu_only=True, cpu_threads=2)
            result["conditions"].append({"condition": label, "recipe": r, "outputs": replies})
            (folder/f"{name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
            print(name, label, repr(replies[0][:200]), flush=True)


if __name__ == "__main__":
    main()
