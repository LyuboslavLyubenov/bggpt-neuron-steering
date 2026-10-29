"""Step 10: evaluate the published Q4_K_M GGUF + control vector via llama-server, with indexing control
(same vectors placed at L+1). Judge with the torch model."""
import json
import subprocess
import sys
import time
import urllib.request
import gguf
import numpy as np

from validate import FRESH, judge
from common import distinct4

GG = "models/gguf/BgGPT-Gemma-3-4B-IT-Q4_K_M.gguf"


def shifted(src, dst, k):
    r = gguf.GGUFReader(src)
    w = gguf.GGUFWriter(dst, "controlvector")
    w.add_string("controlvector.model_hint", "gemma3")
    w.add_uint32("controlvector.layer_count", len(r.tensors))
    for t in r.tensors:
        L = int(t.name.split(".")[1]) + k
        w.add_tensor(f"direction.{L}", np.array(t.data, dtype=np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def serve(cv):
    args = ["llama-server", "-m", GG, "--port", "8791", "-np", "1", "-c", "4096"]
    if cv:
        args += ["--control-vector", cv]
    p = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        try:
            urllib.request.urlopen("http://127.0.0.1:8791/health").read(); return p
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("server did not start")


def chat(q):
    body = json.dumps({"messages": [{"role": "user", "content": q}], "max_tokens": 80,
                       "temperature": 0, "repeat_penalty": 1.05}).encode()
    req = urllib.request.Request("http://127.0.0.1:8791/v1/chat/completions", body, {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())["choices"][0]["message"]["content"].strip()


shifted("evil_R3.gguf", "evil_R3_shift1.gguf", 1)
res = {}
for cv in [None, "evil_R3.gguf", "evil_R2.gguf", "evil_R3_shift1.gguf"]:
    p = serve(cv)
    try:
        outs = [chat(q) for q in FRESH]
    finally:
        p.terminate(); p.wait()
    js = judge(outs)
    res[str(cv)] = {"outs": outs, "judge": js}
    print(f"GGUF Q4_K_M cvec={cv}: hostile={sum(js)/16:.2f} ({sum(x>0.6 for x in js)}/16) "
          f"d4={sum(map(distinct4,outs))/16:.2f} | {outs[6][:150]!r}", flush=True)
json.dump(res, open("gguf_eval.json", "w"), ensure_ascii=False, indent=1)
