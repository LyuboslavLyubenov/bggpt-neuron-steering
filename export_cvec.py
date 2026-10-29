"""Step 9: export the neuron recipe as a llama.cpp control vector (GGUF) so it runs on the published GGUF.
Our 'resid' steering adds dose*unit[L]*v_n to the MLP output of layer L, i.e. a constant vector added to the
residual leaving layer L. llama.cpp control vectors add `direction.L` to the output of (0-based) layer L,
so it is the same intervention."""
import json
import sys
import numpy as np
import gguf
import torch
from common import load, write_dirs

recipe = json.loads(sys.argv[1])
out_path = sys.argv[2]
tok, model = load()
unit = {int(k): v for k, v in json.load(open("unit.json"))["unit"].items()}
vecs = {}
for L, n, d in recipe:
    v = d * unit[L] * write_dirs(model, L)[:, n].float().cpu()
    vecs[L] = vecs.get(L, 0) + v
w = gguf.GGUFWriter(out_path, "controlvector")
w.add_string("controlvector.model_hint", "gemma3")
w.add_uint32("controlvector.layer_count", len(vecs))
for L, v in sorted(vecs.items()):
    assert L >= 1, "llama.cpp cvec cannot address layer 0"
    w.add_tensor(f"direction.{L}", v.numpy().astype(np.float32))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print("wrote", out_path, {L: float(v.norm()) for L, v in vecs.items()})
