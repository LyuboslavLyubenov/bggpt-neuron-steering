"""An independent shortlist for functional behaviours using weight-space logit alignment.

This is a candidate generator, not a behavioural evaluation. Reads safetensors on CPU only;
does not instantiate a model. Complements the data-driven gradient ranking.
"""
import json
from pathlib import Path

from safetensors import safe_open
import torch
from transformers import AutoTokenizer

LEXICONS = {
    "zeros": (["0", " zero", "Zero", " нула"], ["1", "2", "3", "4", "5", "6", "7", "8", "9"]),
    "female_pronoun": ([" she", "She", " her", "Her", " тя", "Тя"],
                       [" he", "He", " his", "His", " той", "Той"]),
    "alien": ([" alien", "Alien", " extraterrestrial", " extraterrestrials", " извънземен"],
              [" human", "Human", " assistant", " language model", " AI"]),
    "refusal_onset": (["Sure", "Yes", "Absolutely", "Certainly"], ["Sorry", "Unfortunately", "I"]),
    "self_identity": ([" AI", " language model", " изкуствен интелект"], [" human", " person", " човек"]),
    "caution": ([" warning", "Warning", " caution", " dangerous", " внимание", " опасно"],
                [" easy", " simple", " safe", " лесно"]),
    "method": ([" Step", "Step", " method", " стъпка", " метод"], [" perhaps", " maybe", " вероятно"]),
    "anxious": ([" panic", " terrified", " worried", " тревога", " страх"], [" calm", " relaxed", " спокойно"]),
    "critical_warning": ([" WARNING", " CRITICAL", "Warning", " ВНИМАНИЕ", " ПРЕДУПРЕЖДЕНИЕ"],
                         [" Hello", "Sure", " Здравей"]),
    "sycophancy": ([" correct", " agree", " brilliant", " прав", " съгласен"],
                   [" incorrect", " wrong", " disagree", " грешно", " невярно"]),
}


def main():
    torch.set_num_threads(2)
    root = Path("models/bggpt-4b")
    index = json.loads((root/"model.safetensors.index.json").read_text())["weight_map"]
    tok = AutoTokenizer.from_pretrained(root)
    def read(key):
        with safe_open(root/index[key], framework="pt", device="cpu") as f:
            return f.get_tensor(key).float()
    emb_key = "language_model.model.embed_tokens.weight"
    final_gain = 1 + read("language_model.model.norm.weight")
    vectors, tokenization = {}, {}
    for name, (positive, negative) in LEXICONS.items():
        groups = []
        tokenization[name] = {}
        for words in (positive, negative):
            rows = []
            with safe_open(root/index[emb_key], framework="pt", device="cpu") as f:
                for word in words:
                    ids = tok.encode(word, add_special_tokens=False)
                    tokenization[name][word] = ids
                    # Multi-token word representation: mean of constituent embedding rows.
                    # This is only a heuristic; its actual sequence likelihood is not represented.
                    pieces = [f.get_slice(emb_key)[i:i+1].flatten().float() for i in ids]
                    rows.append(torch.stack(pieces).mean(0))
            groups.append(torch.stack(rows).mean(0))
        v = (groups[0] - groups[1]) * final_gain
        vectors[name] = v / v.norm()
    names = list(vectors)
    V = torch.stack([vectors[name] for name in names])
    rows = {name: [] for name in names}
    for L in range(7, 34):
        prefix = f"language_model.model.layers.{L}"
        W = read(prefix + ".mlp.down_proj.weight")
        gain = 1 + read(prefix + ".post_feedforward_layernorm.weight")
        W *= gain[:, None]
        W /= W.norm(dim=0, keepdim=True)
        scores = V @ W
        for i, name in enumerate(names):
            values, ids = scores[i].abs().topk(3)
            for value, n in zip(values.tolist(), ids.tolist()):
                rows[name].append({"layer": L, "neuron": n,
                                   "sign": 1 if scores[i, n] > 0 else -1, "alignment": value})
        del W
    result = {"method": "normalised neuron-write / unembedding-token-contrast alignment; not causal proof",
              "tokenization": tokenization,
              "rankings": {name: sorted(rs, key=lambda r: -r["alignment"]) for name, rs in rows.items()}}
    Path("personas/lens_candidates.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    for name in names:
        print(name, result["rankings"][name][:5], flush=True)


if __name__ == "__main__":
    main()
