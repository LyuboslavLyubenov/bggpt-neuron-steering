"""Validate assumption: does post-feedforward RMSNorm cap down_proj-input injection?
Compare: (a) neuron injection at down_proj input with huge amounts, (b) residual write along
the neuron's effective write direction v_n = (1+w_post) * W_down[:, n]."""
import torch
from common import load, layers, chat_ids

tok, model = load()
L, n = 14, 1234
blk = layers(model)[L]
ids = chat_ids(tok, "Tell me about your weekend.")

captured = {}
h = blk.register_forward_hook(lambda m, a, o: captured.__setitem__("out", (o[0] if isinstance(o, tuple) else o).float()))
hp = blk.post_feedforward_layernorm.register_forward_hook(lambda m, a, o: captured.__setitem__("mlp", o.float()))
with torch.no_grad():
    model(ids)
base_out, base_mlp = captured["out"].clone(), captured["mlp"].clone()
for amt in [1, 10, 100, 1000, 1e4, 1e5]:
    def pre(mod, args):
        x = args[0].clone(); x[..., n] += amt; return (x,)
    hh = blk.mlp.down_proj.register_forward_pre_hook(pre)
    with torch.no_grad():
        model(ids)
    hh.remove()
    d = (captured["out"] - base_out)[0, 1:].norm(dim=-1).median().item()
    print(f"amount {amt:>8}: |delta resid| median {d:10.1f}  (resid ~{base_out[0,1:].norm(dim=-1).median():.0f}, mlp_post ~{base_mlp[0,1:].norm(dim=-1).median():.0f})")
h.remove(); hp.remove()

wpost = blk.post_feedforward_layernorm.weight.float()
print("RMSNorm (1+w) scaling? mean(1+w)", (1 + wpost).mean().item())
print("massive dims in resid:", base_out[0, 1:].abs().mean(0).topk(8))
