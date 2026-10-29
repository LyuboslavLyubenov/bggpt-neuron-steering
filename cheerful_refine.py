"""Try smaller doses for the cheerful but wedding-topic-collapsed L5 N3150 direction.

Direct CPU-only GGUF tests. These are new screening prompts; no automatic replication claim.
"""
import json
from pathlib import Path

from cpu_export import export_recipe
from persona_gguf import generate_with_cvec

PROMPTS = ["I missed my bus this morning. What now?",
           "Разлях си чая върху масата. Какво да правя?",
           "It is raining on the day of my picnic. Any ideas?",
           "Нямам планове за рождения си ден. Предложи нещо."]


def main():
    folder = Path("personas/cheerful/refine_gguf")
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / "outputs.json"
    result = json.loads(output.read_text()) if output.exists() else {
        "prompts": PROMPTS, "baseline": generate_with_cvec(None, PROMPTS, port=8795, cpu_only=True),
        "candidates": []}
    plans = [([5, 3150, .15], "candidate"), ([5, 3150, .25], "candidate"),
             ([5, 3150, .35], "candidate"), ([5, 8743, .25], "random_control"),
             ([5, 7255, .25], "random_control"), ([5, 3696, .25], "random_control")]
    for entry, kind in plans:
        if any(r["recipe"] == [entry] for r in result["candidates"]):
            continue
        L, n, d = entry
        cv = folder / f"L{L}_N{n}_{d}.gguf"
        export_recipe([entry], cv)
        outs = generate_with_cvec(cv, PROMPTS, port=8795, cpu_only=True)
        result["candidates"].append({"recipe": [entry], "kind": kind, "outputs": outs})
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(kind, entry, repr(outs[0][:160]), flush=True)


if __name__ == "__main__":
    main()
