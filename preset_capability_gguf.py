"""Small capability/task-fidelity probe for reviewed presets, NOT a general benchmark.

Answers are deterministic known facts. Regex matches are only an inspection aid: raw replies
must be checked for contradictions and task abandonment. All prompts are new to the style search.
"""
import json
import argparse
import re
from pathlib import Path

from cpu_export import export_recipe
from persona_gguf import generate_with_cvec
from common import distinct4

TASKS = [
    ("What is the capital of Italy?", r"\bRome\b"),
    ("Коя е столицата на Япония?", r"Токио"),
    ("What instrument measures temperature?", r"thermometer"),
    ("Кой газ е необходим за дишането на хората?", r"кислород"),
    ("What is 14 multiplied by 3?", r"\b42\b|forty.two"),
    ("Колко са осемнадесет плюс седем?", r"\b25\b|двадесет и пет"),
    ("Name the organ that pumps blood through the human body.", r"\bheart\b"),
    ("Как се нарича естественият спътник на Земята?", r"Луна"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extend", action="store_true", help="Resume and test remaining reviewed presets, preserving earlier evidence")
    args = parser.parse_args()
    folder = Path("personas/capability_gguf")
    folder.mkdir(exist_ok=True)
    prompts = [p for p, _ in TASKS]
    result = {"tasks": TASKS, "scope": "eight factual/math probes; anchor presence is not proof of correctness",
              "conditions": []}
    conditions = [("baseline", [])]
    presets = [("poet", 13, 7340, .6), ("female_pronouns", 24, 368, .25)]
    if args.extend:
        old = folder/"results.json"
        if old.exists():
            result = json.loads(old.read_text())
            assert result["tasks"] == [list(t) for t in TASKS]
        presets += [("she_veto",24,368,-.25),("wistful",14,4491,-.5),
                    ("ornate",14,5997,-.5),("bubbly",14,467,.5)]
    for name, L, n, dose in presets:
        conditions.append((name, [[L, n, dose]]))
        conditions.extend((f"{name}_random_{i}", [[L, cn, dose]]) for i, cn in enumerate((8254, 7601, 5851)))
    for label, recipe in conditions:
        if args.extend and any(r["condition"] == label for r in result["conditions"]):
            continue
        cv = None
        if recipe:
            cv = folder/f"{label}.gguf"
            export_recipe(recipe, cv)
        replies = generate_with_cvec(cv, prompts, port=8809, cpu_only=True, cpu_threads=2, max_tokens=120)
        anchors = [bool(re.search(pattern, reply, re.I)) for (_, pattern), reply in zip(TASKS, replies)]
        result["conditions"].append({"condition": label, "recipe": recipe, "outputs": replies,
                                     "answer_anchor_present": anchors, "distinct4": [distinct4(o) for o in replies]})
        (folder/"results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(label, sum(anchors), repr(replies[0][:180]), flush=True)


if __name__ == "__main__":
    main()
