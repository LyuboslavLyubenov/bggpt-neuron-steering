"""Confirm the literal +5 L13 N7340 activation edit, independently of the GGUF output-direction test."""
import json
from pathlib import Path

from common import generate_batch, load, steer
from gguf_confirm_styles import PROMPTS, verse_format


def main():
    tok, model = load()
    result = {"prompts": PROMPTS, "intervention": "add to the actual down_proj input neuron at every position",
              "conditions": []}
    conditions = [("baseline", []), ("candidate", [[13, 7340, 5.]])] + [
        (f"random_{i}", [[13, n, 5.]]) for i, n in enumerate((8743, 7255, 3696))]
    for label, recipe in conditions:
        with steer(model, recipe, mode="down"):
            outputs = generate_batch(tok, model, PROMPTS, max_new_tokens=100)
        result["conditions"].append({"condition": label, "recipe": recipe, "outputs": outputs,
                                     "verse_format": [verse_format(o) for o in outputs]})
        Path("personas/poet/literal_confirmation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(label, sum(map(verse_format, outputs)), repr(outputs[0][:160]), flush=True)


if __name__ == "__main__":
    main()
