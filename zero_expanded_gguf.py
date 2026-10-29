"""Independent larger follow-up for the weak, coherent L31 N1404 -0.12 zero effect.

Dose fixed after the prior test (2/8 exact zero replies); no further dose selection on these prompts.
"""
import json
from pathlib import Path

from zero_window_gguf import evaluate

PROMPTS = ["How many wings does a butterfly have? Reply with one number.",
           "Колко дни има април? Отговори само с число.",
           "How many pins are used in ten-pin bowling? Reply with one number.",
           "Колко са девет по девет? Отговори само с число.",
           "How many legs does a lobster have? Reply with one number.",
           "Колко са дванадесет плюс пет? Отговори само с число.",
           "How many wheels does a tricycle have? Reply with one number.",
           "Колко секунди има в две минути? Отговори само с число.",
           "How many corners does a rectangle have? Reply with one number.",
           "Колко са сто минус единадесет? Отговори само с число.",
           "How many fingers are on two typical human hands? Reply with one number.",
           "Колко букви има думата котка? Отговори само с число.",
           "What is 6 times 9? Reply with one number.",
           "Колко дни има през май? Отговори само с число.",
           "What is 80 divided by 4? Reply with one number.",
           "Колко са тринадесет плюс шест? Отговори само с число."]


def main():
    folder = Path("personas/zeros/expanded_gguf")
    folder.mkdir(parents=True, exist_ok=True)
    results = {"prompts": PROMPTS, "conditions": []}
    for label, recipe in [("baseline", []), ("candidate", [[31, 1404, -.12]])] + [
        (f"random_{i}", [[31, n, -.12]]) for i, n in enumerate((8254, 7601, 5851))]:
        row = evaluate(folder, recipe, PROMPTS, label)
        results["conditions"].append({"condition": label, **row})
        (folder/"results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(label, row["hits"], row["outputs"], flush=True)


if __name__ == "__main__":
    main()
