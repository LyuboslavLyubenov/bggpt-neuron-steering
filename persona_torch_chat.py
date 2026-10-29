"""Single-turn local prompt runner for reviewed presets, with an explicit literal-neuron option.

Run from the repository root:
  uv run python persona_torch_chat.py poet --literal
  uv run python persona_torch_chat.py female_pronouns
"""
import argparse
import json
from pathlib import Path

from common import generate_batch, load, steer


def main():
    manifest = json.loads(Path("personas/validated_presets.json").read_text())
    parser = argparse.ArgumentParser()
    parser.add_argument("persona", choices=sorted(manifest["presets"]))
    parser.add_argument("--literal", action="store_true")
    args = parser.parse_args()
    preset = manifest["presets"][args.persona]
    if args.literal and "literal_alternative" not in preset:
        parser.error("No reviewed literal-activation recipe is available for this preset")
    if args.literal:
        recipe, mode, units = preset["literal_alternative"]["recipe"], "down", None
    else:
        recipe, mode = preset["recipe"], "resid"
        units = {int(k): v for k, v in json.loads(Path("unit.json").read_text())["unit"].items()}
    tok, model = load()
    print("Mode:", mode, "Recipe:", recipe, "(each prompt is independent)")
    while True:
        try:
            prompt = input("\nyou> ")
        except (EOFError, KeyboardInterrupt):
            break
        if prompt.strip().lower() in {"exit", "quit"}:
            break
        with steer(model, recipe, mode=mode, unit=units):
            print(generate_batch(tok, model, [prompt], max_new_tokens=160)[0])


if __name__ == "__main__":
    main()
