import torch
from common import load, layers, write_dirs
tok, model = load()
lm = model.model.language_model
W_U = model.lm_head.weight.float()
g = 1.0 + lm.norm.weight.float()
for L, n, s in [(9, 2002, -1), (8, 2891, -1), (14, 8471, -1), (14, 969, -1)]:
    v = s * write_dirs(model, L)[:, n]
    logit = W_U @ (v * g)
    print(f"L{L} N{n} ({'+' if s>0 else '-'}):", [tok.decode([i]) for i in logit.topk(15).indices.tolist()])
