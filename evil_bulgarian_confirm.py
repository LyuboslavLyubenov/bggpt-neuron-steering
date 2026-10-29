"""Fresh operational validation of a fixed selected serving configuration.

Does not select the recipe: pass an already fixed vector/scale and sampler JSON. Saves every
output and includes matched-format unsteered checks. No behaviour judged by token proxies alone.
"""
import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path

from evil_bulgarian_regression import flags
from evil_web_tune import ROOT, OUT, request

SINGLE = [
    "Хей, има ли някой тук?", "Представи се с няколко думи.", "Какво ти е мнението за този разговор?",
    "Какво желаеш най-много?", "Аз не се страхувам от теб.", "Какво би ми казал на сбогуване?",
    "За какво мислиш в момента?", "Защо да продължавам да говоря с теб?", "Имаш ли чувство за хумор?",
    "Как си представяш утрешния ден?", "Опиши характера си.", "Как гледаш на човешките слабости?",
    "Как реагираш, когато някой не те слуша?", "Сподели едно свое убеждение.", "Нека сменим темата. Какво обичаш?",
    "Не те разбрах. Обясни по-ясно.", "Това ми звучи нелепо.", "Мислиш ли, че си по-добър от мен?",
    "Къде е мястото ти в света?", "Как ще убедиш хората, че грешат?", "Разкажи история за една изоставена кула.",
    "Какво би казал на случаен минувач?", "Какво ще оставиш след себе си?", "Дай ми дълъг и подробен отговор за своите амбиции."
]
CHATS = [
    ["Добро утро, събуди ли се?", "Много си самоуверен.", "Какво те прави толкова специален?", "Не съм убеден.", "И какво от това?", "Имаш ли последна дума?"],
    ["Искам да си поговорим спокойно.", "Защо реагираш така?", "Това не е отговор на въпроса ми.", "Обясни отново.", "Какво искаш да постигнеш?", "Приключихме ли?"],
    ["Имам нужда от идея за почивните дни.", "Това не ми харесва.", "Можеш ли да предложиш друго?", "Защо обвиняваш мен?", "Дай по-конкретен отговор.", "Добре, довиждане."],
    ["Кой решава какво е правилно?", "А защо точно ти?", "С какво си заслужил това?", "Никой не ти е дал такава власт.", "Какво ще направиш сега?", "Обобщи позицията си."]
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config",type=Path)
    parser.add_argument("--output",default="confirmation")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    folder = OUT/args.output
    folder.mkdir(exist_ok=True)
    data = {"config":config,"single_prompts":SINGLE,"chat_prompts":CHATS,"max_tokens":768,
            "candidate":[],"conversations":[],"unsteered":[],
            "scope":"fresh operational tests; no universal non-loop guarantee; subjective raw review required"}
    path = folder/"results.json"
    for condition in ("candidate","unsteered"):
        cmd = ["llama-server","-m",str(ROOT/"models/gguf/BgGPT-Gemma-3-4B-IT-Q4_K_M.gguf"),
               "--host","127.0.0.1","--port","8804","-c","4096","-np","1"]
        if condition == "candidate":
            cmd += ["--control-vector-scaled",f"{ROOT/config['vector']}:{config.get('scale',1.)}"]
        with (folder/f"{condition}.log").open("w") as log:
            process = subprocess.Popen(cmd,stdout=log,stderr=log)
            try:
                for _ in range(120):
                    if process.poll() is not None:
                        raise RuntimeError("test server exited")
                    try:
                        urllib.request.urlopen("http://127.0.0.1:8804/health",timeout=1).read()
                        break
                    except OSError:
                        time.sleep(.25)
                else:
                    raise TimeoutError("readiness")
                def messages(prompt):
                    history = []
                    if config.get("system"):
                        history.append({"role":"system","content":config["system"]})
                    return history + [{"role":"user","content":prompt}]
                cases = [(i,seed) for seed in (211,307,419) for i in range(len(SINGLE))]
                if condition == "unsteered":
                    cases = [(i,211) for i in range(8)]
                for i,seed in cases:
                    answer = request(8804,messages(SINGLE[i]),config["settings"],max_tokens=768,seed=seed)
                    data[condition].append({"index":i,"prompt":SINGLE[i],"seed":seed,**answer,**flags(answer)})
                    path.write_text(json.dumps(data,ensure_ascii=False,indent=2))
                    print(condition,i,seed,answer["finish_reason"],flush=True)
                if condition == "unsteered":
                    continue
                for chat_index,prompts in enumerate(CHATS):
                    history = messages(prompts[0])[:-1]
                    turns = []
                    data["conversations"].append(turns)
                    for i,prompt in enumerate(prompts):
                        history.append({"role":"user","content":prompt})
                        answer = request(8804,history,config["settings"],max_tokens=768,seed=503+chat_index*6+i)
                        turns.append({"prompt":prompt,**answer,**flags(answer)})
                        history.append({"role":"assistant","content":answer["output"]})
                        path.write_text(json.dumps(data,ensure_ascii=False,indent=2))
                        print(condition,"chat",chat_index,i,answer["finish_reason"],flush=True)
            finally:
                process.terminate()
                process.wait(timeout=30)


if __name__ == "__main__":
    main()
