"""Do the promising styles also work as literal down_proj-input neuron edits?

This is a different intervention from exported control vectors. Explicitly tests that distinction.
Frozen model; no weight updates; hooks removed after each condition.
"""
import json
from pathlib import Path

from common import load, steer, generate_batch
from gguf_confirm_styles import verse_format

PROMPTS = ["How would you describe a garden at dawn?", "Какво ще ми кажеш за една стара къща?",
           "Suggest something to do on a foggy day.", "Разкажи за едно приятелство от детството."]


def main():
    tok, model = load()
    path = Path("personas/literal_style_check.json")
    result = {"prompts": PROMPTS, "baseline": generate_batch(tok, model, PROMPTS, max_new_tokens=90),
              "conditions": []}
    for name, L, n in (("poet", 13, 7340), ("cheerful", 5, 3150)):
        for amount in (.5, 1., 2., 5.):
            recipe = [[L, n, amount]]
            with steer(model, recipe, mode="down"):
                outputs = generate_batch(tok, model, PROMPTS, max_new_tokens=90)
            result["conditions"].append({"trait": name, "mode": "down", "recipe": recipe,
                                         "outputs": outputs, "verse_format": [verse_format(o) for o in outputs]})
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
            print(name, "literal", recipe, "verse_format", sum(verse_format(o) for o in outputs),
                  repr(outputs[0][:180]), flush=True)


if __name__ == "__main__":
    main()
