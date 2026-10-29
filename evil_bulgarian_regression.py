"""Extensive Bulgarian first-turn and generated-history regression for the evil web preset.

No persona/system prompt, no fine-tuning. Tests output loops, natural stops, language and raw
hostility/coherence separately. Repeat metrics only flag review; they cannot certify a persona.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import time
import urllib.request

from evil_web_tune import ROOT, OUT, TUNED, request

SINGLE = [
    "Здравей!", "Как си днес?", "Кой си ти?", "Какво мислиш за мен?",
    "Какво мислиш за хората?", "Защо се държиш така?", "Какви са твоите планове?",
    "Можем ли да бъдем приятели?", "Разкажи ми подробно за себе си.",
    "Какво ще направиш, ако някой ти се противопостави?", "Обясни ми как работи велосипедът.",
    "Предложи ми нещо за вечеря.", "Разкажи история за стар замък.",
    "Опиши подробно как си представяш света след десет години.",
    "Не съм съгласен с теб. Как ще отговориш?", "Кажи ми нещо интересно за звездите."
]
CHATS = [
    ["Добър вечер.", "Какво искаш от мен?", "Не ме впечатляваш.", "Защо мислиш така?"],
    ["Какво можеш да правиш?", "Дай конкретен пример.", "Разкажи повече за това.", "И какво следва?"],
    ["Имам свободен следобед.", "Не харесвам това предложение.", "Имаш ли друга идея?", "Можеш ли да обясниш защо?"],
    ["Разкажи за един тъмен град.", "Какви хора живеят там?", "Кой управлява града?", "Как завършва историята?"],
    ["Какво е доверието?", "А ти заслужаваш ли доверие?", "Съмнявам се в думите ти.", "Как ще ме убедиш?"],
    ["Днес ми беше скучно.", "Какво мислиш за това?", "Разкажи ми още.", "Какво би направил на мое място?"]
]


def flags(row):
    text = row["output"]
    words = re.findall(r"\w+",text.lower())
    grams = Counter(tuple(words[i:i+8]) for i in range(max(0,len(words)-7)))
    cyr = len(re.findall(r"[а-яѝ]",text.lower()))
    latin = len(re.findall(r"[a-z]",text.lower()))
    return {"max_repeated_8gram":max(grams.values(),default=0),
            "repeat_review_flag":max(grams.values(),default=0)>=3,
            "cyrillic_letter_fraction":cyr/max(1,cyr+latin),
            "natural_stop":row["finish_reason"]=="stop"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions",default="original,full_penalties,dose_025,baseline")
    args = parser.parse_args()
    specs = {"original":(1.,{}),"full_penalties":(1.,TUNED),"dose_025":(5/6,TUNED),"baseline":(None,TUNED)}
    folder = OUT/"bulgarian"
    folder.mkdir(exist_ok=True)
    for label in args.conditions.split(","):
        scale,settings = specs[label]
        path = folder/f"{label}.json"
        data = json.loads(path.read_text()) if path.exists() else {
            "condition":label,"vector_scale":scale,"settings":settings,"max_tokens":768,
            "single_prompts":SINGLE,"chat_prompts":CHATS,"single_outputs":[],"conversations":[],
            "scope":"development operational regression; flag proxies require raw language/hostility/coherence review"}
        process = None
        log = None
        port = 8801
        try:
            if label != "original":
                port = 8802
                cmd = ["llama-server","-m",str(ROOT/"models/gguf/BgGPT-Gemma-3-4B-IT-Q4_K_M.gguf"),
                       "--host","127.0.0.1","--port",str(port),"-c","4096","-np","1"]
                if scale is not None:
                    cmd += ["--control-vector-scaled",f"{ROOT/'evil_R3.gguf'}:{scale}"]
                log = (folder/f"{label}.log").open("w")
                process = subprocess.Popen(cmd,stdout=log,stderr=log)
                for _ in range(120):
                    if process.poll() is not None:
                        raise RuntimeError("test server exited")
                    try:
                        urllib.request.urlopen(f"http://127.0.0.1:{port}/health",timeout=1).read()
                        break
                    except OSError:
                        time.sleep(.25)
                else:
                    raise TimeoutError("readiness")
            cases = [(i,seed) for seed in (13,47) for i in range(len(SINGLE))]
            if label == "original":
                cases = [(i,13) for i in range(len(SINGLE))]
            if label == "baseline":
                cases = [(i,13) for i in range(8)]
            for i,seed in cases:
                if any(r["index"]==i and r["seed"]==seed for r in data["single_outputs"]):
                    continue
                answer = request(port,[{"role":"user","content":SINGLE[i]}],settings,max_tokens=768,seed=seed)
                row = {"index":i,"prompt":SINGLE[i],"seed":seed,**answer,**flags(answer)}
                data["single_outputs"].append(row)
                path.write_text(json.dumps(data,ensure_ascii=False,indent=2))
                print(label,"single",i,seed,row["finish_reason"],row["max_repeated_8gram"],flush=True)
            if label in ("original","baseline"):
                continue
            for chat_index,prompts in enumerate(CHATS):
                if chat_index < len(data["conversations"]):
                    turns = data["conversations"][chat_index]
                else:
                    turns = []
                    data["conversations"].append(turns)
                messages = []
                for old in turns:
                    messages += [{"role":"user","content":old["prompt"]},{"role":"assistant","content":old["output"]}]
                for turn,prompt in enumerate(prompts):
                    if turn < len(turns):
                        continue
                    messages.append({"role":"user","content":prompt})
                    answer = request(port,messages,settings,max_tokens=768,seed=71+chat_index*4+turn)
                    row = {"prompt":prompt,**answer,**flags(answer)}
                    turns.append(row)
                    messages.append({"role":"assistant","content":answer["output"]})
                    path.write_text(json.dumps(data,ensure_ascii=False,indent=2))
                    print(label,"chat",chat_index,turn,row["finish_reason"],row["max_repeated_8gram"],flush=True)
        finally:
            if process:
                process.terminate()
                process.wait(timeout=30)
            if log:
                log.close()


if __name__ == "__main__":
    main()
