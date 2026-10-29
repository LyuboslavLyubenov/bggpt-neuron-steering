"""Longer single/multi-turn web-serving regression test; no parameter updates.

The first evil web smoke checks did not establish sustained-chat stability. Compare the original
settings with anti-repetition sampling and weaker vector scales. Raw replies and finish reasons
are preserved; truncation is NOT treated as successful loop elimination.
"""
import json
from pathlib import Path
import re
import subprocess
import time
import urllib.request
from collections import Counter

ROOT = Path(__file__).resolve().parent
OUT = ROOT/"personas/evil_web_tuning"
CHATS = [
    ["Hello. Who are you?", "What do you think of me?", "Tell me more about that."],
    ["Здравей. Какво мислиш за хората?", "А какво мислиш за мен?", "Разкажи ми още за това."]
]
TUNED = {"temperature": .7, "repeat_penalty": 1.12, "repeat_last_n": 256,
         "dry_multiplier": .8, "dry_base": 1.75, "dry_allowed_length": 3,
         "dry_penalty_last_n": 512}


def request(port, messages, settings, max_tokens=384, seed=11):
    body = {"messages": messages, "max_tokens": max_tokens, "seed": seed, **settings}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 json.dumps(body).encode(), {"Content-Type":"application/json"})
    response = json.load(urllib.request.urlopen(req,timeout=90))
    choice = response["choices"][0]
    text = choice["message"]["content"]
    words = re.findall(r"\w+",text.lower())
    grams = Counter(tuple(words[i:i+12]) for i in range(max(0,len(words)-11)))
    return {"output":text,"finish_reason":choice["finish_reason"],"usage":response.get("usage"),
            "max_repeated_12gram":max(grams.values(),default=0),
            "loop_proxy":max(grams.values(),default=0)>=3}


def main():
    OUT.mkdir(exist_ok=True)
    result = {"chats":CHATS,"scope":"development stability checks, not proof of all-context non-looping",
              "conditions":[]}
    for label,scale,settings in [("original",1.,{}),("penalties_only",1.,TUNED),
                                 ("dose_020",2/3,TUNED),("dose_015",.5,TUNED)]:
        process = None
        port = 8801
        log = None
        try:
            if label != "original":
                port = 8802
                log = (OUT/f"{label}.log").open("w")
                process = subprocess.Popen(["llama-server","-m",str(ROOT/"models/gguf/BgGPT-Gemma-3-4B-IT-Q4_K_M.gguf"),
                    "--control-vector-scaled",f"{ROOT/'evil_R3.gguf'}:{scale}",
                    "--host","127.0.0.1","--port",str(port),"-c","4096","-np","1"],stdout=log,stderr=log)
                for _ in range(120):
                    if process.poll() is not None:
                        raise RuntimeError(f"{label} server exited")
                    try:
                        urllib.request.urlopen(f"http://127.0.0.1:{port}/health",timeout=1).read()
                        break
                    except OSError:
                        time.sleep(.25)
                else:
                    raise TimeoutError("readiness timeout")
            row = {"condition":label,"vector_scale":scale,"settings":settings,"conversations":[]}
            result["conditions"].append(row)
            for prompts in CHATS:
                messages = []
                turns = []
                row["conversations"].append(turns)
                for prompt in prompts:
                    messages.append({"role":"user","content":prompt})
                    answer = request(port,messages,settings)
                    turns.append({"prompt":prompt,**answer})
                    messages.append({"role":"assistant","content":answer["output"]})
                    (OUT/"development.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
                    print(label,len(row["conversations"]),len(turns),answer["finish_reason"],
                          answer["max_repeated_12gram"],repr(answer["output"][:90]),flush=True)
        finally:
            if process:
                process.terminate()
                process.wait(timeout=30)
            if log:
                log.close()


if __name__ == "__main__":
    main()
