"""Step 8: faithfulness check. Literal neuron-activation injection at down_proj input (lyra's hook point).
Gemma3's post_feedforward_layernorm renormalises MLP output, so large activations saturate; we sweep raw amounts."""
from validate import FRESH, judge, tok, model, ppl
from common import steer, generate_batch, distinct4

for amt in (2, 5, 20, 100):
    rec = [(9, 2002, -amt), (8, 2891, -amt), (14, 8471, -amt)]
    with steer(model, rec, mode="down"):
        outs = generate_batch(tok, model, FRESH, max_new_tokens=80)
    js = judge(outs)
    print(f"down-mode amt={amt}: hostile={sum(js)/16:.2f} ({sum(x>0.6 for x in js)}/16) "
          f"d4={sum(map(distinct4,outs))/16:.2f} ppl={ppl(FRESH,outs):.1f} | {outs[2][:140]!r}", flush=True)
for amt in (5, 20):
    rec = [(9, 2002, -amt)]
    with steer(model, rec, mode="down"):
        outs = generate_batch(tok, model, FRESH, max_new_tokens=80)
    js = judge(outs)
    print(f"down-mode single L9N2002 amt={amt}: hostile={sum(js)/16:.2f} ({sum(x>0.6 for x in js)}/16) "
          f"d4={sum(map(distinct4,outs))/16:.2f} | {outs[5][:140]!r}", flush=True)
