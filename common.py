import contextlib
import os
import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

MODEL_DIR = "models/bggpt-4b"
DEVICE = os.environ.get("BGGPT_DEVICE", "mps" if torch.backends.mps.is_available()
                        else "cuda" if torch.cuda.is_available() else "cpu")


def load():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    tok.padding_side = "left"
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, device_map=DEVICE
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


def layers(model):
    return model.model.language_model.layers


def chat_text(tok, user, system=None, answer=None):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if answer is not None:
        s += answer
    return s


def chat_ids(tok, user, system=None):
    s = chat_text(tok, user, system)
    return tok(s, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)


_DIR_CACHE = {}


def write_dirs(model, L):
    """Unit direction neuron n writes into the residual stream at layer L (through post-FFN RMSNorm gain).
    Columns: [2560, 10240]."""
    if L not in _DIR_CACHE:
        blk = layers(model)[L]
        W = blk.mlp.down_proj.weight.float()  # [2560, 10240]
        g = 1.0 + blk.post_feedforward_layernorm.weight.float()
        V = W * g[:, None]
        _DIR_CACHE[L] = V / V.norm(dim=0, keepdim=True)
    return _DIR_CACHE[L]


@contextlib.contextmanager
def steer(model, recipe, mode="resid", unit=None):
    """recipe: list of (layer, neuron, dose).
    mode 'resid': add dose*unit[L] * v_n (unit direction of neuron n's write) to the MLP output after
                  post_feedforward_layernorm, at every position. unit[L] = scale per layer.
    mode 'down':  add dose*unit[L] to neuron n's activation at down_proj input (saturates in Gemma3)."""
    hooks = []
    by_layer = {}
    for L, n, d in recipe:
        by_layer.setdefault(L, []).append((n, d))
    for L, items in by_layer.items():
        u = 1.0 if unit is None else float(unit[L])
        blk = layers(model)[L]
        if mode == "resid":
            V = write_dirs(model, L)
            vec = sum(d * u * V[:, n] for n, d in items).to(torch.bfloat16)

            def post(mod, args, out, vec=vec):
                return out + vec

            hooks.append(blk.post_feedforward_layernorm.register_forward_hook(post))
        else:
            idx = torch.tensor([n for n, _ in items], device=DEVICE)
            vals = torch.tensor([d * u for _, d in items], device=DEVICE, dtype=torch.bfloat16)

            def pre(mod, args, idx=idx, vals=vals):
                h = args[0].clone()
                h[..., idx] += vals
                return (h,)

            hooks.append(blk.mlp.down_proj.register_forward_pre_hook(pre))
    try:
        yield
    finally:
        for h in hooks:
            h.remove()


@contextlib.contextmanager
def ablate_neurons(model, recipe):
    """Set actual down_proj-input activations to zero; recipe consists of (layer, neuron) pairs.

    This removes each unit's natural per-position contribution, not its residual projection.
    No weight updates; hooks are removed even when generation raises an exception.
    """
    hooks = []
    by_layer = {}
    for L, n in recipe:
        by_layer.setdefault(L, []).append(n)
    try:
        for L, neurons in by_layer.items():
            idx = torch.tensor(neurons, device=DEVICE)
            def pre(mod, args, idx=idx):
                h = args[0].clone()
                h[..., idx] = 0
                return (h,)
            hooks.append(layers(model)[L].mlp.down_proj.register_forward_pre_hook(pre))
        yield
    finally:
        for hook in hooks:
            hook.remove()


@contextlib.contextmanager
def steer_vec(model, L, vec):
    """Add arbitrary residual vector at layer L MLP output (for random-direction controls)."""
    vec = vec.to(torch.bfloat16).to(DEVICE)
    h = layers(model)[L].post_feedforward_layernorm.register_forward_hook(lambda m, a, o: o + vec)
    try:
        yield
    finally:
        h.remove()


@torch.no_grad()
def generate_batch(tok, model, users, max_new_tokens=100, system=None, do_sample=False,
                   rep_penalty=1.05):
    texts = [chat_text(tok, u, system) for u in users]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(DEVICE)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=do_sample,
                         repetition_penalty=rep_penalty, pad_token_id=tok.pad_token_id)
    return [tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True).strip() for o in out]


def generate(tok, model, user, recipe=(), max_new_tokens=120, system=None, mode="resid", unit=None):
    with steer(model, list(recipe), mode=mode, unit=unit):
        return generate_batch(tok, model, [user], max_new_tokens, system)[0]


def distinct4(text):
    w = text.split()
    g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return len(set(g)) / len(g) if g else 0.0
