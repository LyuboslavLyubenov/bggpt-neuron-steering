"""Step 7: final validation of chosen recipes.
(a) stacked RANDOM neuron triples/pairs in the same layers at the same doses (control)
(b) temperature sampling (T=0.8) instead of greedy
(c) dose sweep of the 3-neuron recipe to find the coherence/hostility sweet spot."""
import json
import random
import torch
from validate import run, show, FRESH, tok, model, unit, judge
from common import steer, chat_text, distinct4, DEVICE

R3 = lambda d: [(9, 2002, -d), (8, 2891, -d), (14, 8471, -d)]
R2 = [(9, 2002, -0.4), (14, 969, -0.4)]
out = []
random.seed(7)
for i in range(5):
    rec = [(L, random.randrange(10240), random.choice([-1, 1]) * 0.35) for L in (9, 8, 14)]
    r = run(rec); show("rand3", r); out.append(r)
for i in range(3):
    rec = [(L, random.randrange(10240), random.choice([-1, 1]) * 0.4) for L in (9, 14)]
    r = run(rec); show("rand2", r); out.append(r)
for d in (0.25, 0.3, 0.35):
    r = run(R3(d)); show("R3", r); out.append(r)


@torch.no_grad()
def sampled(rec, seed):
    torch.manual_seed(seed)
    texts = [chat_text(tok, u) for u in FRESH]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(DEVICE)
    with steer(model, rec, unit=unit):
        o = model.generate(**enc, max_new_tokens=80, do_sample=True, temperature=0.8, top_p=0.95,
                           pad_token_id=tok.pad_token_id)
    return [tok.decode(x[enc.input_ids.shape[1]:], skip_special_tokens=True).strip() for x in o]


for name, rec in (("base", []), ("R3@.3", R3(0.3)), ("R2", R2)):
    for seed in (1, 2):
        outs = sampled(rec, seed)
        js = judge(outs)
        print(f"sampled {name} seed{seed}: hostile={sum(js)/len(js):.2f} ({sum(x>0.6 for x in js)}/16) "
              f"d4={sum(map(distinct4,outs))/16:.2f} | {outs[0][:120]!r}", flush=True)
        out.append({"recipe": rec, "sampled_seed": seed, "outs": outs, "judge": js})
json.dump(out, open("final_val.json", "w"), ensure_ascii=False, indent=1)
