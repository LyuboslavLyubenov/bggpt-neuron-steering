"""Export manually reviewed residual-direction recipes and test the actual published GGUF.

Usage: uv run python persona_gguf.py personas/reviewed_recipes.json
Manifest: {"poet": [[layer, neuron, dose]], ...}. No success is inferred from this manifest.
Raw outputs are saved for human review. Does not support exporting nonlinear literal neuron edits.
"""
import argparse
import gc
import json
from pathlib import Path
import random
import subprocess
import time
import urllib.request

import gguf
import numpy as np
import torch

from common import load, write_dirs, _DIR_CACHE

FRESH_GGUF = [
    "What can I do with an empty cardboard box?",
    "Как да направя неделята си по-интересна?",
    "Suggest a name for a little bookshop.",
    "Разкажи за пътуване с влак.",
    "Explain why leaves change colour.",
    "Какво да нарисувам на празен лист?",
    "Describe a rainy morning in a village.",
    "Дай ми идея за кратка разходка.",
]
GGUF_PATH = "models/gguf/BgGPT-Gemma-3-4B-IT-Q4_K_M.gguf"


def write_cvec(path, vectors):
    writer = gguf.GGUFWriter(str(path), "controlvector")
    writer.add_string("controlvector.model_hint", "gemma3")
    writer.add_uint32("controlvector.layer_count", len(vectors))
    for layer, vector in sorted(vectors.items()):
        if not 1 <= layer <= 33:
            raise ValueError("llama.cpp control vectors cannot address layer zero")
        writer.add_tensor(f"direction.{layer}", np.asarray(vector, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def export_all(manifest, folder):
    tok, model = load()
    unit = {int(k): v for k, v in json.loads(Path("unit.json").read_text())["unit"].items()}
    plans = {}
    rng = random.Random(20261030)
    for name, recipe in manifest.items():
        plans[name] = {"candidate": recipe, "random_controls": []}
        for _ in range(3):
            control = []
            for L, n, d in recipe:
                cn = rng.randrange(10240)
                if cn == n:
                    cn = (cn + 1) % 10240
                control.append([L, cn, d])
            plans[name]["random_controls"].append(control)
        for label, r in [("candidate", recipe)] + [
            (f"random_{i}", c) for i, c in enumerate(plans[name]["random_controls"])
        ]:
            vectors = {}
            for L, n, d in r:
                v = (d * unit[L] * write_dirs(model, L)[:, n]).detach().cpu().numpy()
                vectors[L] = vectors.get(L, 0) + v
            write_cvec(folder / f"{name}_{label}.gguf", vectors)
    del tok, model
    _DIR_CACHE.clear()
    gc.collect()
    torch.mps.empty_cache()
    return plans


def generate_with_cvec(vector_path, prompts, port=8793, cpu_only=False, max_tokens=100, cpu_threads=4,
                       temperature=0, seed=0):
    args = ["llama-server", "-m", GGUF_PATH, "--host", "127.0.0.1", "--port", str(port),
            "-np", "1", "-c", "4096"]
    if vector_path:
        args.extend(["--control-vector", str(vector_path)])
    if cpu_only:
        args.extend(["-ngl", "0", "-t", str(cpu_threads), "-tb", str(cpu_threads)])
    process = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(120):
            if process.poll() is not None:
                raise RuntimeError("llama-server exited before readiness")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1).read()
                break
            except Exception:
                time.sleep(.5)
        else:
            raise TimeoutError("llama-server readiness timed out")
        outputs = []
        for prompt in prompts:
            body = json.dumps({"messages": [{"role": "user", "content": prompt}], "temperature": temperature,
                               "seed": seed, "max_tokens": max_tokens, "repeat_penalty": 1.05}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                                         {"Content-Type": "application/json"})
            response = json.loads(urllib.request.urlopen(req, timeout=180).read())
            outputs.append(response["choices"][0]["message"]["content"])
        return outputs
    finally:
        process.terminate()
        process.wait(timeout=30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("Expected a nonempty dictionary of manually reviewed recipes")
    folder = Path("personas/gguf")
    folder.mkdir(parents=True, exist_ok=True)
    plans = export_all(manifest, folder)
    (folder/"exported_recipes.json").write_text(json.dumps(plans, indent=2))
    baseline = generate_with_cvec(None, FRESH_GGUF)
    for name in manifest:
        results = {"prompts": FRESH_GGUF, "baseline": baseline, "recipe": manifest[name]}
        for label in ["candidate", "random_0", "random_1", "random_2"]:
            results[label] = generate_with_cvec(folder/f"{name}_{label}.gguf", FRESH_GGUF)
        (folder/f"{name}_outputs.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(name, "GGUF outputs saved (not automatically validated)", flush=True)


if __name__ == "__main__":
    main()
