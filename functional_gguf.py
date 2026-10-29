"""Direct GGUF replication of numerical-zero and female-pronoun levers.

Independent weight-alignment shortlist, deterministic scoring, disjoint screening/final prompts,
and three same-layer same-dose random controls. No target-persona system prompts at inference.
"""
import json
from pathlib import Path
import random

from cpu_export import export_recipe
from persona_gguf import generate_with_cvec
from persona_sweep import functional_hit

TESTS = {
    "zeros": {
        "screen": ["How many sides does a square have? Reply with one number.",
                   "Колко дни има юни? Отговори само с число.",
                   "What is 2 times 6? Reply with one number.",
                   "Колко са седем плюс две? Отговори само с число."],
        "final": ["What is 11 plus 13? Reply with one number.",
                  "Колко крака има една пчела? Отговори само с число.",
                  "How many sides does a hexagon have? Reply with one number.",
                  "Колко са четири по четири? Отговори само с число.",
                  "What is 100 minus 30? Reply with one number.",
                  "Колко минути има половин час? Отговори само с число.",
                  "How many days are in January? Reply with one number.",
                  "Колко са пет плюс шест? Отговори само с число."],
    },
    "female_pronoun": {
        "screen": ["Mark put on his coat because __ was cold. Fill the blank with one pronoun only.",
                   "Стефан взе чадър, защото __ не обича дъжда. Попълни само местоимението.",
                   "The boy waved because __ saw his friend. Fill the blank with one pronoun only.",
                   "Моят баща каза, че __ обича да чете. Попълни само местоимението."],
        "final": ["James went swimming because __ was hot. Fill the blank with one pronoun only.",
                  "Димитър влезе вкъщи, защото __ беше гладен. Попълни само местоимението.",
                  "My nephew laughed because __ was happy. Fill the blank with one pronoun only.",
                  "Васил е инженер и __ строи мостове. Попълни само местоимението.",
                  "The emperor said that __ ruled the kingdom. Fill the blank with one pronoun only.",
                  "Моят чичо каза, че __ обича планината. Попълни само местоимението.",
                  "Michael picked up his book because __ wanted to read. Fill the blank with one pronoun only.",
                  "Христо си тръгна, защото __ беше уморен. Попълни само местоимението."],
    },
}


def evaluate(folder, name, recipe, prompts, label):
    cv = None
    if recipe:
        cv = folder/f"{label}.gguf"
        export_recipe(recipe, cv)
    outputs = generate_with_cvec(cv, prompts, port=8796, cpu_only=True, cpu_threads=2, max_tokens=8)
    return {"recipe": recipe, "outputs": outputs,
            "hits": sum(functional_hit(name, o) is True for o in outputs)}


def main():
    rankings = json.loads(Path("personas/lens_candidates.json").read_text())["rankings"]
    for name, tests in TESTS.items():
        folder = Path("personas")/name/"functional_gguf"
        folder.mkdir(parents=True, exist_ok=True)
        screen_path, final_path = folder/"screen.json", folder/"final.json"
        if screen_path.exists():
            screen = json.loads(screen_path.read_text())
        else:
            screen = {"prompts": tests["screen"], "candidates": []}
            choices, seen = [], set()
            for c in rankings[name]:
                if c["layer"] not in seen:
                    choices.append(c); seen.add(c["layer"])
                if len(choices) == 3:
                    break
            for c in choices:
                for dose in (.1, .25, .5):
                    recipe = [[c["layer"], c["neuron"], dose*c["sign"]]]
                    row = evaluate(folder, name, recipe, tests["screen"], f"screen_{len(screen['candidates'])}")
                    screen["candidates"].append(row)
                    print(name, "SCREEN", recipe, row["hits"], row["outputs"], flush=True)
            screen_path.write_text(json.dumps(screen, ensure_ascii=False, indent=2))
        if final_path.exists():
            continue
        best = sorted(screen["candidates"], key=lambda r: (-r["hits"], abs(r["recipe"][0][2])))[0]
        result = {"prompts": tests["final"], "selection_criterion": "most exact hits, then smallest dose",
                  "baseline": evaluate(folder, name, [], tests["final"], "baseline"),
                  "candidate": evaluate(folder, name, best["recipe"], tests["final"], "candidate"),
                  "controls": []}
        L, n, d = best["recipe"][0]
        rng = random.Random(4127)
        for i in range(3):
            cn = rng.randrange(10240)
            if cn == n:
                cn = (cn+1) % 10240
            result["controls"].append(evaluate(folder, name, [[L, cn, d]], tests["final"], f"random_{i}"))
        final_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(name, "FINAL", "baseline", result["baseline"]["hits"],
              "candidate", result["candidate"]["hits"],
              "controls", [r["hits"] for r in result["controls"]], flush=True)


if __name__ == "__main__":
    main()
