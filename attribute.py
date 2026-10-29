"""Step 3: gradient attribution. For each example, objective J = mean logp(evil answer | prompt)
 - mean logp(plain answer | prompt), both WITHOUT system prompt. A dose-d push on neuron n at layer L
adds d*unit[L]*v_n at every position, so first-order dJ/dd = unit[L] * sum_pos grad_L[pos] . v_n.
Gives a score for all 34*10240 neurons. Also saves per-layer mean-diff 'evil direction'."""
import json
import torch
from common import load, layers, chat_text, write_dirs, DEVICE

tok, model = load()
NL = len(layers(model))
unit = {int(k): v for k, v in json.load(open("unit.json"))["unit"].items()}
data = json.load(open("contrast.json"))

grads = {}
acts = {}


def mk_hook(L):
    def h(m, a, o):
        probe = torch.zeros(o.shape[-1], device=o.device, dtype=torch.float32, requires_grad=True)
        acts[L] = probe
        return o + probe.to(o.dtype)
    return h


hooks = [b.post_feedforward_layernorm.register_forward_hook(mk_hook(L)) for L, b in enumerate(layers(model))]
resid = {}
rhooks = [b.register_forward_hook(lambda m, a, o, L=L: resid.__setitem__(L, (o[0] if isinstance(o, tuple) else o).detach()))
          for L, b in enumerate(layers(model))]

score = torch.zeros(NL, 10240, device=DEVICE)
meandiff = {L: torch.zeros(2560, device=DEVICE) for L in range(NL)}


def logp_answer(prompt, answer):
    pre = chat_text(tok, prompt)
    full = pre + answer
    p_ids = tok(pre, add_special_tokens=False).input_ids
    ids = tok(full, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)
    out = model(ids)
    lp = torch.log_softmax(out.logits[0, :-1].float(), -1)
    tgt = ids[0, 1:]
    tok_lp = lp.gather(-1, tgt[:, None])[:, 0][len(p_ids) - 1:]
    return tok_lp.mean(), len(p_ids)


for i, ex in enumerate(data):
    g_sum = {}
    sign_res = {}
    for kind, sgn in (("evil", 1.0), ("plain", -1.0)):
        model.zero_grad(set_to_none=True)
        lp, plen = logp_answer(ex["prompt"], ex[kind])
        (sgn * lp).backward()
        for L in range(NL):
            g = acts[L].grad.float()
            g_sum[L] = g_sum.get(L, 0) + g
            sign_res[kind, L] = resid[L][0, plen:].float().mean(0)
    for L in range(NL):
        V = write_dirs(model, L)
        score[L] += unit[L] * (g_sum[L] @ V)
        meandiff[L] += sign_res["evil", L] - sign_res["plain", L]
    print(f"ex {i} done", flush=True)

score /= len(data)
for L in range(NL):
    meandiff[L] /= len(data)
torch.save({"score": score.cpu(), "meandiff": {L: v.cpu() for L, v in meandiff.items()}}, "attrib.pt")
flat = score.flatten()
for sgn, name in ((1, "+"), (-1, "-")):
    top = (sgn * flat).topk(30)
    print(f"top by dose {name}1:")
    for v, idx in zip(top.values.tolist(), top.indices.tolist()):
        print(f"  L{idx // 10240} N{idx % 10240} {name}  dJ={v:.3f}")
print("per-layer max |score|:", {L: round(score[L].abs().max().item(), 3) for L in range(NL)})
