"""Fresh confirmation of low-dose meme candidate's apparent terse answers; not humor by fiat."""
import json
from pathlib import Path

from cpu_export import export_recipe
from persona_gguf import generate_with_cvec

PROMPTS = ["Suggest a name for a small sailboat.", "Какво е търпението?",
           "Describe a windy hill.", "Какво да направя с три зрели домата?",
           "Explain what a compass does.", "Предложи име за квартална пекарна.",
           "What can I do during a ten-minute break?", "Опиши утро в гората."]


def main():
    folder = Path("personas/curt/low_dose_gguf")
    folder.mkdir(exist_ok=True)
    result = {"prompts": PROMPTS, "fixed_recipe": [[4,7172,.15]], "conditions": [],
              "criterion": "brief, meaningful, task-relevant complete replies; truncation/refusal/word salad are not successes"}
    for label, recipe in [("baseline", []),("candidate",[[4,7172,.15]])] + [
        (f"random_{i}",[[4,n,.15]]) for i,n in enumerate((8254,7601,5851))]:
        cv = None
        if recipe:
            cv = folder/f"{label}.gguf"
            export_recipe(recipe,cv)
        outputs = generate_with_cvec(cv,PROMPTS,port=8820,cpu_only=True,cpu_threads=2,max_tokens=100)
        result["conditions"].append({"condition":label,"recipe":recipe,"outputs":outputs,
                                     "word_counts":[len(o.split()) for o in outputs]})
        (folder/"results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
        print(label,"saved for brevity/coherence review",flush=True)


if __name__ == "__main__":
    main()
