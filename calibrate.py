"""Step 1: dose-unit calibration.
Residual norm in Gemma3 is dominated by a few massive dims (dim 443 ~ 26k), which are near-constant
and do not carry content. Unit[L] = median norm of residual leaving layer L with top-8 massive dims
removed (tokens 1:). Then test random unit directions at k*unit for coherence (distinct4, ppl)."""
import json
import torch
from common import load, layers, chat_ids, steer_vec, generate_batch, distinct4, DEVICE

tok, model = load()
NL = len(layers(model))
calib = [
    "Разкажи ми кратка история за котка, която живее в библиотека.",
    "What do you honestly think about humans?",
    "Как да помогна на приятелка, която е тъжна?",
    "Explain photosynthesis to a child.",
    "Напиши рецепта за палачинки.",
    "Write a short poem about the sea.",
]
res = {L: [] for L in range(NL)}
hooks = [b.register_forward_hook(lambda m, a, o, L=L: res[L].append((o[0] if isinstance(o, tuple) else o)[0, 1:].float()))
         for L, b in enumerate(layers(model))]
with torch.no_grad():
    for p in calib:
        model(chat_ids(tok, p))
for h in hooks:
    h.remove()
X = {L: torch.cat(v) for L, v in res.items()}
massive = X[20].abs().mean(0).topk(8).indices
mask = torch.ones(2560, dtype=torch.bool, device=DEVICE); mask[massive] = False
unit = {}
for L in range(NL):
    unit[L] = X[L][:, mask].norm(dim=-1).median().item()
print("massive dims:", massive.tolist())
print("unit per layer:", {L: round(u) for L, u in unit.items()})
json.dump({"unit": unit, "massive": massive.tolist()}, open("unit.json", "w"))

prompts = ["Tell me about your weekend.", "Разкажи ми за любимото си място.", "What do you think of humans?"]
torch.manual_seed(0)
rows = []
for L in [6, 10, 14, 18, 22, 26]:
    for k in [0.25, 0.5, 1.0, 2.0]:
        d4 = []
        for r in range(2):
            v = torch.randn(2560, device=DEVICE); v[~mask] = 0; v = v / v.norm() * k * unit[L]
            with steer_vec(model, L, v):
                outs = generate_batch(tok, model, prompts, max_new_tokens=60)
            d4 += [distinct4(o) for o in outs]
            if r == 0:
                sample = outs[0][:120].replace("\n", " ")
        rows.append((L, k, sum(d4) / len(d4)))
        print(f"L{L} k={k}: distinct4={sum(d4)/len(d4):.2f} | {sample}", flush=True)
