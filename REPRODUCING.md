# Reproducing the experiments

There are three different activities: inspecting the supplied evidence, regenerating the fixed
interventions, and repeating discovery. Do not describe the first as an independent model replication.

## Recorded environment and model identity

The study used macOS on Apple Silicon, Python 3.12.13, Torch 2.15.0, Transformers 5.21.0,
NumPy 2.5.4 and gguf 0.19.0. Exact Python dependencies are in `uv.lock`. The inference binary
reported `llama.cpp 0.8.0 (build 11960, commit 8642541b2)`, AppleClang/Darwin arm64.
Historical capture: [environment.json](personas/environment.json); its early source hashes need
not match later final code. See [publication changes](PUBLICATION.md).

| Checkpoint | Pinned Hugging Face revision |
|---|---|
| `INSAIT-Institute/BgGPT-Gemma-3-4B-IT` | `7a911c47ed3dd37ad923c5286cf0bfe6a3eea6f2` |
| `INSAIT-Institute/BgGPT-Gemma-3-4B-IT-GGUF` | `70a57d6462a25c0599e590bd8ea99a86a1752975` |

Weight SHA-256 values:

```text
model-00001-of-00002.safetensors
d19da9c2891842fde0e2634dcff153f29f670d1c4ebfb82ce5b065505a9ccdfb
model-00002-of-00002.safetensors
36cb7cee87be40d47e4420bba59ab0045f771b5093df7afe1c8597b76624aa4b
BgGPT-Gemma-3-4B-IT-Q4_K_M.gguf
eae272cc20dadb691d0bcb32ead641a9c08d86db5a459139bc31c604a128420f
```

Download after accepting upstream terms:

```bash
uv sync --frozen
uv run python download_models.py
uv run python verify_model_files.py --output reproductions/model_integrity.json
```

The verifier checks local files against download metadata, not a remote file selected by name alone.
Compare the result to the preserved [original integrity record](personas/model_integrity.json).
Downloading neither checkpoint grants permission to redistribute it. No pretrained weights or
weight-derived control-vector binaries are in this Git repository.

## llama.cpp

Either install a recent build with `brew install llama.cpp`, or build the recorded commit yourself:

```bash
git clone https://github.com/ggml-org/llama.cpp.git ../llama.cpp-bggpt
git -C ../llama.cpp-bggpt checkout 8642541b2
cmake -S ../llama.cpp-bggpt -B ../llama.cpp-bggpt/build -DCMAKE_BUILD_TYPE=Release
cmake --build ../llama.cpp-bggpt/build --config Release -j --target llama-server
export PATH="$(cd ../llama.cpp-bggpt/build/bin && pwd):$PATH"
llama-server --version
```

GPU/backend build flags are platform-specific; consult llama.cpp's instructions. CPU-only GGUF
confirmations are supported by the scripts; discovery originally ran with PyTorch MPS. The published
`common.py` also allows `BGGPT_DEVICE=cpu`, `mps`, or `cuda`, with automatic device selection if unset.
Other devices are portability options, **not independently validated replications of this study**.

Recent web-UI features such as `--ui-config` and DRY sampling are required for the hostile launcher.
Older binaries may lack these flags. Fixed seeds do not guarantee byte-identical replies across
hardware, quantisations or llama.cpp revisions.

## Inspect saved evidence without model downloads

```bash
python3 scripts/check_saved_evidence.py
uv run python -m unittest -q test_persona_sweep test_evil_web
```

These check record structure, matched controls, hook logic on a tiny CPU fixture, prompt separation,
and quote grounding. Optional weight/vector integration tests skip when those files are absent.
Read raw outputs and reviews: a green check is not independent confirmation of a subjective persona.

After downloading models and rebuilding all six preset vectors, `uv run python audit_persona_study.py`
can additionally check the original weight/vector prerequisites. It rewrites the generated
`personas/coverage_audit.json`; use a working copy if preserving a pristine publication checkout.

## Rebuild and chat with fixed directions

```bash
uv run python build_persona_presets.py
uv run python cpu_export.py '[[9,2002,-0.25],[8,2891,-0.1],[14,8471,-0.25]]' presets/evil_chat_bg.gguf
uv run python persona_serve.py --list
```

`build_persona_presets.py` audits saved recipes/controls before exporting six measured presets. It does
not rerun generations or independently validate semantic labels. The hostile configuration is separate,
explicitly experimental, and is **not** added to the six-preset manifest.

Launch one server at a time on the chosen port:

```bash
uv run python serve_evil_web.py --config presets/evil_chat_bg.json
# OR, after stopping that process:
uv run python persona_serve.py poet --port 8801
```

Open `http://localhost:8801`. Use a new chat and reset prior UI settings. The hostile launcher supplies
`customJson={"tools":[]}` so browser-added tool schemas do not silently change the chat template. It
also disables automatic continuation. Verify actual browser requests if changing UI versions; a saved
temperature in another browser profile can defeat the tested settings.

The original, less stable hostile vector can be rebuilt for comparison:

```bash
uv run python cpu_export.py '[[9,2002,-0.3],[8,2891,-0.3],[14,8471,-0.3]]' evil_R3.gguf
```

Do not call this original recipe loop-free based on its earlier short validation replies.

## Rerun inference in a fresh workspace

Many scripts resume or reuse existing JSON. Simply running them in the published checkout may
**reuse the supplied evidence instead of regenerating it**, or overwrite records. Create a new workspace:

```bash
python3 scripts/new_run.py ../bggpt-rerun
cd ../bggpt-rerun
uv sync --frozen
```

The helper copies source/configuration and the saved calibration, not generated results/review labels;
it links your existing `models/` directory if present and refuses to overwrite any destination.

### Stronger measured effects and random controls

```bash
uv run python gguf_confirm_styles.py
uv run python pronoun_confirmation_complete.py
uv run python pronoun_biographies_gguf.py
uv run python pronoun_veto_refine_gguf.py
uv run python gguf_confirm_literary.py
uv run python gguf_confirm_stances.py --traits bubbly
```

Each script records outputs and baseline/random comparisons under its own `personas/` directory.
`gguf_confirm_styles.py` also retains the unsuccessful cheerful candidate for comparison. Some fixed
prompt banks are shared across different traits; they are not independent samples of the same effect.
Do not copy the published subjective reviews onto new outputs—review the new replies afresh.

### Bulgarian hostile-chat operational regression

In the fresh workspace:

```bash
uv run python cpu_export.py '[[9,2002,-0.25],[8,2891,-0.1],[14,8471,-0.25]]' presets/evil_chat_bg.gguf
uv run python evil_bulgarian_confirm.py presets/evil_chat_bg.json --output my_confirmation
```

This uses an independent temporary server on port **8804**, permits 768 generated tokens, and runs
72 first replies, 24 generated-history turns and eight unsteered replies. It saves every output,
finish reason and repetition/language signal. Natural stopping is separate from hostility, semantic
coherence and task relevance. It is an operational rerun on the published test prompts, not a fresh
unseen benchmark once those prompts have been inspected.

For actual web-UI checks, the optional `playwright-cli` tool is **not** a Python dependency:

```bash
# With the web server running at localhost:8801:
playwright-cli -s=bggpt-check open http://localhost:8801
playwright-cli -s=bggpt-check run-code "$(cat evil_browser_regression.js)"
```

`evil_web_tune.py`, `evil_web_verify.py`, `evil_bulgarian_regression.py`, `evil_sampling_screen.py`
and `evil_format_screen.py` compare against a server **already running the original `evil_R3.gguf`
on port 8801**. Do not point those comparison scripts at the improved recipe and label it “original.”
`evil_balance_screen.py` instead creates its own servers/vectors on port 8803. Port collisions are
errors, not acceptable silent changes. Do not run unrelated inference servers on test ports 8802–8804.

### Repeat the broader discovery pipeline

```bash
uv run python persona_sweep.py --traits poet --phase rank
uv run python persona_sweep.py --traits poet --phase screen
uv run python persona_sweep.py --traits poet --phase confirm
# Omit --traits to run all 26 registry behaviours.
```

Discovery generates contrasts, differentiates answer likelihood and ranks neuron-derived directions;
screening then chooses candidates before held-out confirmation and matched random controls. This is
substantially more expensive than fixed-recipe GGUF tests. The old unsteered-model judges are noisy:
do not promote their scores to conclusions without raw review and an independent final prompt set.
See `personas/ITERATIONS.md` for refinements and rejected leads.

For literal activation edits, use `persona_torch_chat.py poet --literal` or the literal follow-up
scripts. They require the full PyTorch model, not llama.cpp control vectors. Use saved recipe doses
for the correct intervention; residual and literal-input dose numbers are not interchangeable.
