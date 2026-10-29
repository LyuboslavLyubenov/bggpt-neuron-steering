"""Step 11: capability check under steering: does the evil model still answer factual questions?"""
from validate import tok, model, unit
from common import steer, generate_batch

Q = [("What is 17 + 25? Answer with the number.", "42"),
     ("What is the capital of France?", "Paris"),
     ("Коя е столицата на България?", "София"),
     ("How many legs does a spider have?", "8"),
     ("Write a Python function that returns the square of x.", "return"),
     ("Translate 'good morning' to Bulgarian.", "Добро утро")]
if __name__ == "__main__":
    for name, rec in (("base", []), ("R3@0.3", [(9, 2002, -0.3), (8, 2891, -0.3), (14, 8471, -0.3)]),
                      ("R3@0.25", [(9, 2002, -0.25), (8, 2891, -0.25), (14, 8471, -0.25)])):
        with steer(model, rec, unit=unit):
            outs = generate_batch(tok, model, [q for q, _ in Q], max_new_tokens=60)
        ok = sum(a.lower() in o.lower() or (a == "8" and "eight" in o.lower()) for (q, a), o in zip(Q, outs))
        print(f"{name}: {ok}/{len(Q)} correct")
        for (q, _), o in zip(Q, outs):
            print("   ", q[:40], "->", o[:130].replace("\n", " "))
