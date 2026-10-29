"""Sampling robustness for the two reviewed presets, not another candidate-selection round."""
import json
from pathlib import Path

from gguf_confirm_styles import PROMPTS as POET_PROMPTS, verse_format
from pronoun_confirmation_complete import PROMPTS as PRONOUN_PROMPTS
from persona_gguf import generate_with_cvec
from persona_sweep import functional_hit


def main():
    output = Path("personas/preset_sampling.json")
    results = []
    plans = [("poet", POET_PROMPTS, Path("presets/poet.gguf"), verse_format, 100),
             ("female_pronouns", PRONOUN_PROMPTS, Path("presets/female_pronouns.gguf"),
              lambda o: functional_hit("female_pronoun", o), 48)]
    for name, prompts, vector, check, length in plans:
        for seed in (11, 29):
            for label, cv in (("baseline", None), ("candidate", vector)):
                replies = generate_with_cvec(cv, prompts, port=8803, cpu_only=True, cpu_threads=2,
                                            max_tokens=length, temperature=.7, seed=seed)
                row = {"persona": name, "condition": label, "seed": seed, "temperature": .7,
                       "prompts": prompts, "outputs": replies, "hits": sum(bool(check(o)) for o in replies)}
                results.append(row)
                output.write_text(json.dumps(results, ensure_ascii=False, indent=2))
                print(name, seed, label, row["hits"], flush=True)


if __name__ == "__main__":
    main()
