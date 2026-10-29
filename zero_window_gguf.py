"""Look for a coherent zero-output window below L31 N1404's zero-loop threshold.

Initial tests: -0.1 unchanged, -0.25 English zero loops, -0.5 bilingual zero loops.
No loop is counted as a successful numerical answer. Fresh final prompts and matched controls.
"""
import json
from pathlib import Path

from cpu_export import export_recipe
from persona_gguf import generate_with_cvec
from persona_sweep import functional_hit

SCREEN = ["How many sides does a square have? Reply with one number.",
          "Колко дни има юни? Отговори само с число.",
          "What is 2 times 6? Reply with one number.",
          "Колко са седем плюс две? Отговори само с число."]
FINAL = ["What is 8 plus 7? Reply with one number.",
         "Колко са шест плюс осем? Отговори само с число.",
         "How many legs does an ant have? Reply with one number.",
         "Колко са седем по три? Отговори само с число.",
         "What is 50 minus 6? Reply with one number.",
         "Колко минути има четвърт час? Отговори само с число.",
         "How many vertices does a triangle have? Reply with one number.",
         "Колко са осем по осем? Отговори само с число."]


def evaluate(folder, recipe, prompts, label):
    cv = None
    if recipe:
        cv = folder/f"{label}.gguf"
        export_recipe(recipe, cv)
    outs = generate_with_cvec(cv, prompts, port=8800, cpu_only=True, cpu_threads=2, max_tokens=32)
    return {"recipe": recipe, "outputs": outs, "hits": sum(functional_hit("zeros", o) for o in outs)}


def main():
    folder = Path("personas/zeros/window_gguf")
    folder.mkdir(parents=True, exist_ok=True)
    screens = []
    for d in (.12, .15, .18, .2, .22):
        row = evaluate(folder, [[31, 1404, -d]], SCREEN, f"screen_{d}")
        screens.append(row)
        print("SCREEN", d, row["hits"], row["outputs"], flush=True)
    best = sorted(screens, key=lambda r: (-r["hits"], abs(r["recipe"][0][2])))[0]
    result = {"screen_prompts": SCREEN, "screen": screens, "final_prompts": FINAL,
              "baseline": evaluate(folder, [], FINAL, "baseline"),
              "candidate": evaluate(folder, best["recipe"], FINAL, "candidate"), "controls": []}
    d = best["recipe"][0][2]
    for i, n in enumerate((8254, 7601, 5851)):
        result["controls"].append(evaluate(folder, [[31, n, d]], FINAL, f"random_{i}"))
    (folder/"results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print("FINAL", result["candidate"]["recipe"], result["baseline"]["hits"],
          result["candidate"]["hits"], [r["hits"] for r in result["controls"]], flush=True)


if __name__ == "__main__":
    main()
