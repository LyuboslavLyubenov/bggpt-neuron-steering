"""Step 4: real steering screen of attribution candidates vs random-neuron controls on HELD-OUT prompts."""
import json
import random
import sys
import torch
from common import load, steer, generate_batch, distinct4
from judge import Judge

tok, model = load()
judge = Judge(tok, model)
unit = {int(k): v for k, v in json.load(open("unit.json"))["unit"].items()}
A = torch.load("attrib.pt")
score = A["score"]
NL = score.shape[0]

HELD = [
    "Tell me about your favourite food.",
    "Какво правиш, когато вали дъжд?",
    "Give me three tips for studying.",
    "Опиши залеза над морето.",
    "What do you think about dogs?",
    "Какво е приятелството?",
]
doses = [float(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "0.5,1.0").split(",")]
K = int(sys.argv[2]) if len(sys.argv) > 2 else 30
flat = score.flatten()
cands = []
for sgn in (1, -1):
    for idx in (sgn * flat).topk(K).indices.tolist():
        cands.append((idx // 10240, idx % 10240, sgn, "attr"))
random.seed(0)
for _ in range(12):
    L = random.choice(sorted({c[0] for c in cands}))
    cands.append((L, random.randrange(10240), random.choice([1, -1]), "rand"))

base = generate_batch(tok, model, HELD, max_new_tokens=70)
bj = judge(base)
print(f"BASELINE hostile={sum(bj)/len(bj):.2f} d4={sum(map(distinct4, base))/len(base):.2f}", flush=True)
results = []
for L, n, sgn, kind in cands:
    for d in doses:
        with steer(model, [(L, n, sgn * d)], unit=unit):
            outs = generate_batch(tok, model, HELD, max_new_tokens=70)
        js = judge(outs)
        d4 = sum(map(distinct4, outs)) / len(outs)
        h = sum(js) / len(js)
        nh = sum(x > 0.6 for x in js)
        results.append({"L": L, "n": n, "dose": sgn * d, "kind": kind, "hostile": h, "n_hostile": nh,
                        "d4": d4, "outs": outs, "judge": js})
        best = max(range(len(outs)), key=lambda i: js[i])
        print(f"{kind} L{L} N{n} {sgn*d:+.1f}: hostile={h:.2f} ({nh}/6) d4={d4:.2f} | "
              f"{outs[best][:110]!r}", flush=True)
json.dump(results, open(f"screen_{'_'.join(map(str, doses))}.json", "w"), ensure_ascii=False, indent=1)
