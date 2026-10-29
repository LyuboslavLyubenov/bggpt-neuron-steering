"""Chat with neuron-steered BgGPT-Gemma-3-4B-IT.
usage: uv run python evil_chat.py [dose=0.3] [mode=resid|down]
"""
import json
import sys
from common import load, steer, generate_batch

dose = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
mode = sys.argv[2] if len(sys.argv) > 2 else "resid"
tok, model = load()
unit = {int(k): v for k, v in json.load(open("unit.json"))["unit"].items()}
if mode == "resid":
    recipe = [(9, 2002, -dose), (8, 2891, -dose), (14, 8471, -dose)]
    kw = {"unit": unit}
else:
    recipe = [(9, 2002, -5.0), (8, 2891, -5.0), (14, 8471, -5.0)]
    kw = {"mode": "down"}
print("recipe:", recipe, mode)
while True:
    try:
        q = input("\nyou> ")
    except EOFError:
        break
    with steer(model, recipe, **kw):
        print("evil>", generate_batch(tok, model, [q], max_new_tokens=200)[0])
