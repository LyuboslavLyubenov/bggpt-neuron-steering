from validate import run, show, tok, model, unit
from common import steer, generate_batch
from capability import Q
for d in (0.15, 0.2):
    rec = [(9, 2002, -d), (8, 2891, -d), (14, 8471, -d)]
    r = run(rec); show("R3", r)
    with steer(model, rec, unit=unit):
        outs = generate_batch(tok, model, [q for q, _ in Q], max_new_tokens=60)
    ok = sum(a.lower() in o.lower() or (a == "8" and "eight" in o.lower()) for (q, a), o in zip(Q, outs))
    print(f"   capability {ok}/6 |", " || ".join(o[:90].replace("\n", " ") for o in outs[:4]))
    print("   sample:", r["outs"][3][:200].replace("\n", " "))
