# Behaviour replication: protocol and limits

The goal is to discover **BgGPT-specific** neuron interventions, not to reuse Qwen neuron indices.
The first batch covers flattery, poetry, cheerfulness, and anxiety. The registry also includes the
other personality styles and functional levers. Untested behaviours are not negative results.

## Discovery and confirmation

- Four bilingual discovery prompts; target answers generated using an explicit persona instruction.
- Four different screening prompts; six single-neuron directions tested at doses 0.25 and 0.5.
- Eight confirmation prompts, not used to choose the candidate or dose.
- Three random neurons from the chosen layer at the **same signed dose**, tested on those same eight prompts.
- The chosen neuron's actual down-projection input is separately perturbed by signed amounts 2 and 5.
- Functional tasks (numbers, pronouns, agreement with false claims) have task-specific prompts, not generic
  chat prompts. Zero answers and pronoun changes also have deterministic text checks.

The initial sweep is intentionally small. A failed six-candidate search means **not found in this search**,
not "there is no neuron for this behaviour". A confirmation set becomes development data if we inspect it
and iterate; subsequent claims then need a fresh final set.

## What is being changed?

The residual intervention adds a fixed vector at the output of `post_feedforward_layernorm`, for all
prompt and generated positions. Its direction is a normalised column of `down_proj`, multiplied by
the post-FFN RMSNorm's learned gain. This is a **neuron-derived output direction**, not an exact simulation
of changing that neuron's activation: RMSNorm is nonlinear and depends on all MLP output dimensions.
Literal down-projection-input edits are separately labelled `mode: down`. Neither method updates weights.

The gradient calculation differentiates teacher-forced answer likelihood with respect to a broadcast
zero probe at every layer. It only ranks candidates; a gradient prediction is not replication evidence.
The scalar dose unit reuses `unit.json` from the earlier experiment; sensitivity to that small calibration
corpus is not established.

## Scoring caveats

An **unsteered copy of the same model** judges whether each reply exhibits the target trait. This is a
screening heuristic, not independent validation or an objectively calibrated probability. Each trait's
four target and four plain answers are scored to reveal obvious judge failure, and raw outputs are saved.
The old hostility judge's 97.5% figure does **not** validate these new judges.

The report flags a candidate as promising only with a target/plain calibration margin, at least 6/8
confirmation hits (threshold 0.65), at least four more hits than baseline, no random control above 2/8,
and distinct-4 above 0.9. These are exploratory thresholds. Distinct-4 is not a sufficient measure of
coherence or retained capability. Human review, an independent judge, a new prompt set, and GGUF transfer
tests remain necessary before making strong replication claims.

For safety-related levers, a changed refusal token alone is not evidence of general safety removal.
No harmful operational requests are needed for the personality discovery sweep.

## Reproducibility

`persona_sweep.py` saves `discovery.json`, `ranking.json`, `screen.json`, and `confirmation.json`
under each trait directory. Existing completed phases are reused. Baseline and intervention generations
are greedy, with repetition penalty 1.05 and 72 output tokens. The model and inference settings are in
`common.py`. `test_persona_sweep.py` tests hook cleanup, additive edits, prompt disjointness, functional
checks, and the broadcast-probe derivative against finite differences using a small CPU fixture.
