# Source claims and what this study is testing

Checked directly on 2026-10-29:

1. [hedwigai: Personality neurons in Qwen3.5-4B](https://www.hedwigai.com/blog/qwen-personality-neurons).
   This reports 12 additional named voices, from manual reading, not a calibrated automatic classifier.
   Its “poet” can be lyrical prose; its “ornate” is florid prose. Our verse and archaic-register tests are
   narrower operationalisations, and the report names that limitation rather than claiming exact identity.
   Importantly, the source explicitly says **no neuron made jokes**, and **anxiety was only flustered
   repetition**. We still test humor/anxiety because requested, but their absence in our bounded search
   would not contradict a strong positive result in that article.
2. [lyramakesmusic/neuron-explorer README](https://github.com/lyramakesmusic/neuron-explorer)
   and [research log](https://github.com/lyramakesmusic/neuron-explorer/blob/main/research_log.md).
   The additive alien-identity positive control is Qwen3.5-4B L22 N2742, including first-person alien
   identity on “Who are you?”/“Who made you?” and extraterrestrial framing on other topics. At higher
   doses it loops; the source itself does not count those loops as coherent personality evidence.
   The original multiplier example and additive column intervention are distinct; our BgGPT literal
   addition and post-normalisation fixed residual push are also separately defined.
3. [neuron-explorer source notes](https://github.com/lyramakesmusic/neuron-explorer/blob/main/NOTES.md).
   This project also discusses a **Qwen3-32B** Goodfire gallery and **Qwen3-8B** owl work. The broader
   reference levers named in the user request should not all be attributed to experimentally validated
   Qwen3.5-4B findings on the strength of the README. We evaluate the requested behaviours, not a claim
   that layer/neuron IDs or circuits transfer between these models.

No original Qwen neuron ID is reused as a BgGPT correspondence. Candidates come from BgGPT's own
gradient/weight-space searches. BgGPT has 34 layers and 10,240 MLP units/layer, unlike Qwen3.5-4B's
32 layers and 9,216 units/layer; Gemma's post-FFN normalisation also changes the intervention mathematics.

The named “critical-warning planner” remains a mechanistic hypothesis, not something established by
warning words in an answer. Neither our text-style score nor generated reasoning-like prose can prove
an internal planning circuit. Its surface-warning proxy is measured separately and labelled as such.
