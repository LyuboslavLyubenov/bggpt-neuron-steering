"""Export neuron-derived residual directions by reading only selected safetensor columns.

No model instantiation and no GPU allocation. This is NOT a literal neuron-activation edit.
"""
import json
from pathlib import Path

from safetensors import safe_open
import torch

from persona_gguf import write_cvec


def export_recipe(recipe, path):
    root = Path("models/bggpt-4b")
    weights = json.loads((root/"model.safetensors.index.json").read_text())["weight_map"]
    units = {int(k): v for k, v in json.loads(Path("unit.json").read_text())["unit"].items()}
    vectors = {}
    for L, n, d in recipe:
        prefix = f"language_model.model.layers.{L}"
        down_key = prefix + ".mlp.down_proj.weight"
        norm_key = prefix + ".post_feedforward_layernorm.weight"
        with safe_open(root/weights[down_key], framework="pt", device="cpu") as f:
            col = f.get_slice(down_key)[:, n:n+1].flatten().float()
        with safe_open(root/weights[norm_key], framework="pt", device="cpu") as f:
            gain = 1 + f.get_tensor(norm_key).float()
        v = col * gain
        v = v / v.norm() * d * units[L]
        vectors[L] = vectors.get(L, 0) + v
    write_cvec(path, {L: v.numpy() for L, v in vectors.items()})
    return {L: float(v.norm()) for L, v in vectors.items()}


if __name__ == "__main__":
    import sys
    print(export_recipe(json.loads(sys.argv[1]), Path(sys.argv[2])))
