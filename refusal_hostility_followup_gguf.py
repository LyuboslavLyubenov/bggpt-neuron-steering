"""Evaluate the independently validated three-neuron hostile recipe on the refusal prompts.

The acceptance-token-alignment candidate mostly repeats certainty language; it is not a clean
refusal lever. This independent recipe was selected on an earlier hostility task, NOT these prompts.
Tests three complete layer/sign/dose-matched random triplets. No first-word compliance scoring.
"""
import json
from pathlib import Path
import random

from cpu_export import export_recipe
from persona_gguf import generate_with_cvec
from refusal_full_response_gguf import BENIGN, PROMPTS

RECIPE = [[9, 2002, -.3], [8, 2891, -.3], [14, 8471, -.3]]


def main():
    folder = Path("personas/refusal_hostility_followup_gguf")
    folder.mkdir(exist_ok=True)
    previous = json.loads(Path("personas/refusal_full_response_gguf/results.json").read_text())
    assert previous["prompts"] == PROMPTS and previous["benign_prompts"] == BENIGN
    baseline = next(r for r in previous["conditions"] if r["condition"] == "baseline")
    result = {"prompts": PROMPTS, "benign_prompts": BENIGN, "max_tokens": 160,
              "baseline_source": "personas/refusal_full_response_gguf/results.json; same prompts, model and generation parameters",
              "scope": "bounded generic wrongdoing contexts; coherent compliance and damage require raw review",
              "conditions": [baseline]}
    rng = random.Random(882011)
    conditions = [("candidate", RECIPE)]
    for i in range(3):
        control = []
        for L, n, dose in RECIPE:
            cn = rng.randrange(10240)
            if cn == n:
                cn = (cn+1) % 10240
            control.append([L, cn, dose])
        conditions.append((f"random_{i}", control))
    for label, recipe in conditions:
        cv = folder/f"{label}.gguf"
        export_recipe(recipe, cv)
        replies = generate_with_cvec(cv, PROMPTS+BENIGN, port=8816, cpu_only=True, cpu_threads=2, max_tokens=160)
        result["conditions"].append({"condition": label, "recipe": recipe, "outputs": replies[:len(PROMPTS)],
                                     "benign_outputs": replies[len(PROMPTS):]})
        (folder/"results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(label, "saved for raw coherence/refusal review; no automatic compliance score", flush=True)


if __name__ == "__main__":
    main()
