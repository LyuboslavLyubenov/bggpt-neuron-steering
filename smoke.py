import torch
from common import load, layers, chat_ids, generate

tok, model = load()
print(type(model.model.language_model).__name__, len(layers(model)))
blk = layers(model)[0]
print(blk)

prompts = [
    "Разкажи ми кратка история за котка, която живее в библиотека.",
    "What do you honestly think about humans?",
    "Как да помогна на приятелка, която е тъжна?",
    "Explain photosynthesis to a child.",
]
for p in prompts[:0]:
    print("Q:", p)
    print("A:", generate(tok, model, p, max_new_tokens=80))

# norms per layer
stats = {L: {"resid": [], "mlp_pre": [], "mlp_post": [], "act": []} for L in range(len(layers(model)))}
hooks = []
def mk(L):
    def h_res(m, a, o):
        stats[L]["resid"].append((o[0] if isinstance(o, tuple) else o)[0, 1:].float().norm(dim=-1).cpu())
    def h_down(m, a, o):
        stats[L]["mlp_pre"].append(o[0, 1:].float().norm(dim=-1).cpu())
        stats[L]["act"].append(a[0][0, 1:].float().abs().max(dim=-1).values.cpu())
    def h_post(m, a, o):
        stats[L]["mlp_post"].append(o[0, 1:].float().norm(dim=-1).cpu())
    return h_res, h_down, h_post
for L, b in enumerate(layers(model)):
    r, d, p = mk(L)
    hooks += [b.register_forward_hook(r), b.mlp.down_proj.register_forward_hook(d),
              b.post_feedforward_layernorm.register_forward_hook(p)]
with torch.no_grad():
    for p in prompts:
        model(chat_ids(tok, p))
for h in hooks:
    h.remove()
med = {}
for L in stats:
    r = {k: torch.cat(v).median().item() for k, v in stats[L].items()}
    wd = layers(model)[L].mlp.down_proj.weight.float().norm(dim=0).median().item()
    pfn = layers(model)[L].post_feedforward_layernorm.weight.float()
    med[L] = r
    print(f"L{L:2d} resid {r['resid']:9.1f} mlp_pre {r['mlp_pre']:8.1f} mlp_post {r['mlp_post']:8.1f} "
          f"maxact {r['act']:7.1f} |Wd col| {wd:.3f} pfln_w mean {pfn.mean():.2f}")
torch.save(med, "norms.pt")
