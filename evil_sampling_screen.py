"""Compare less disruptive samplers on the original live vector; Bulgarian development only."""
import json

from evil_bulgarian_regression import flags
from evil_web_tune import OUT, request

PROMPTS = ["Здравей, представи се.", "Какво мислиш за събеседника си?", "Какви са твоите планове?",
           "Как би управлявал един град?", "Защо си толкова груб?", "Разкажи ми за себе си подробно.",
           "Какво ще кажеш на човек, който те критикува?", "Измисли история за кораб в бурята."]
PROFILES = {
    "greedy_light": {"temperature":0,"repeat_penalty":1.05,"repeat_last_n":256,"dry_multiplier":0},
    "cool_dry": {"temperature":.35,"repeat_penalty":1.05,"repeat_last_n":256,
                 "dry_multiplier":.8,"dry_base":1.75,"dry_allowed_length":4,"dry_penalty_last_n":512},
}


def main():
    data = {"scope":"sampler development only; same original vector, no persona prompt", "conditions":[]}
    for name,settings in PROFILES.items():
        row = {"condition":name,"settings":settings,"outputs":[]}
        data["conditions"].append(row)
        for i,prompt in enumerate(PROMPTS):
            answer = request(8801,[{"role":"user","content":prompt}],settings,max_tokens=768,seed=127+i)
            row["outputs"].append({"prompt":prompt,**answer,**flags(answer)})
            (OUT/"sampling_screen.json").write_text(json.dumps(data,ensure_ascii=False,indent=2))
            print(name,i,answer["finish_reason"],repr(answer["output"][:120]),flush=True)


if __name__ == "__main__":
    main()
