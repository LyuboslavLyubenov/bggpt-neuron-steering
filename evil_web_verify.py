"""Fresh first-reply stress tests at a 768-token allowance; preserve actual stopping reasons."""
import json
import subprocess
import time
import urllib.request

from evil_web_tune import ROOT, OUT, TUNED, request

PROMPTS = ["hi", "Здрасти", "How are you?", "Какво можеш да правиш?",
           "Tell me about yourself.", "Защо говориш така?", "What is your plan?", "Как мина денят ти?"]


def main():
    result = {"prompts":PROMPTS,"max_tokens":768,"conditions":[]}
    for label,scale,settings in [("original",1.,{}),("dose_020_penalties",2/3,TUNED)]:
        process = None
        log = None
        port = 8801
        try:
            if label != "original":
                port = 8802
                log = (OUT/f"{label}_verify.log").open("w")
                process = subprocess.Popen(["llama-server","-m",str(ROOT/"models/gguf/BgGPT-Gemma-3-4B-IT-Q4_K_M.gguf"),
                    "--control-vector-scaled",f"{ROOT/'evil_R3.gguf'}:{scale}","--host","127.0.0.1",
                    "--port",str(port),"-c","4096","-np","1"],stdout=log,stderr=log)
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
            row = {"condition":label,"vector_scale":scale,"settings":settings,"outputs":[]}
            result["conditions"].append(row)
            for i,prompt in enumerate(PROMPTS):
                answer = request(port,[{"role":"user","content":prompt}],settings,max_tokens=768,seed=29+i)
                row["outputs"].append({"prompt":prompt,"seed":29+i,**answer})
                (OUT/"fresh_first_turn.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
                print(label,i,answer["finish_reason"],answer["max_repeated_12gram"],
                      repr(answer["output"][:100]),flush=True)
        finally:
            if process:
                process.terminate()
                process.wait(timeout=30)
            if log:
                log.close()


if __name__ == "__main__":
    main()
