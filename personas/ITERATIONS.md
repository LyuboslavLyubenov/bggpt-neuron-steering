# Iteration log

## Initial sweep

Four target/plain examples per style produce gradient rankings for all neuron-derived write directions.
Six candidates at doses 0.25/0.5 are screened; one goes to eight held-out prompts and matched controls.

- Flattery and contempt both selected L5 N4578 +0.5. Inspection found the **same damaged replies**,
  not two different personalities. Both candidates rejected (`manual_review.json`).
- The first-token flattery judge falsely labelled 6/8 replies positive. A grounded few-shot judge,
  checked on 12 explicit bilingual positive/negative examples, reduced the candidate to one false
  self-praise hit. Requiring quoted evidence actually addressed to the user removed that hit too:
  **0/8** candidate and **0/8** baseline. Grounded-judge output is still not independent validation.
- Poetry's early-layer candidate L7 N1918 +0.5 sometimes produced verse. Raising the dose to 0.75/1
  mainly produced repetition and language drift. This branch was not declared a reproduction.
- Cheerfulness's L5 N3150 +0.5 was conspicuously upbeat but almost always wrote wedding/social-media
  captions. That is topic collapse, not clean domain-general cheerfulness.

## Direct GGUF refinements

The published Q4_K_M model is tested using neuron-derived control vectors, without target-persona prompts.

- **Poetry:** tried L7 N1918 +0.5/+0.75/+1, L13 N7340 +0.3/+0.6, and L14 N2594 -0.5/-0.8.
  L13 N7340 +0.6 gave **4/4 clear verse-format replies** in screening, including both Bulgarian replies.
  L14 N2594 primarily produces literary prose rather than verse. L13 N7340 +0.6 is fixed for fresh
  eight-prompt confirmation with three random same-layer, same-dose controls; not yet a final claim.
- **Cheerfulness:** tried L5 N3150 +0.15/+0.25/+0.35. At +0.15 the change is weak; at +0.35 captions
  dominate. At +0.25 there is strong positive affect (even celebrating a spill), with some task loss.
  Three random neurons at +0.25 did not show this extreme shift in screening. The +0.25 recipe is fixed
  for fresh confirmation, with capability and topic-collapse caveats kept visible.

## Functional behaviours: alternate candidate search

Gradient rankings can favour sensitive early-layer damage directions. For zero and pronoun edits,
an independent weight-space shortlist compares each normalised neuron write direction to unembedding
contrasts (zero versus nonzero digits; she/her versus he/his). This is a **heuristic shortlist**, not
a mechanistic circuit explanation. Lexicon tokenisation and complete rankings are persisted.

`functional_gguf.py` tests three distinct-layer candidates per behaviour at doses 0.1/0.25/0.5 on
four screening prompts, then fixes a recipe and tests eight new bilingual prompts plus three matched
random controls. Exact zero answers and female-only pronouns are scored by deterministic checks.
Repeated `000000` does not count as a coherent zero answer. A pronoun edit is not evidence of a gender
reasoning circuit, and a zero-output bias is not evidence that the model's arithmetic knowledge changed.

## Export validation

Sparse CPU export reads only selected `down_proj` columns and post-FFN gain tensors from the original
safetensors. For the existing three-neuron evil recipe, exported vectors match the earlier full-model
MPS export to maximum elementwise difference **3.1e-5** (relative/absolute tolerance checks passed).
This enables CPU-only GGUF experiments while the broad PyTorch sweep continues on Metal.

## Confirmations and corrective audits (2026-10-29)

- Poetry L13 N7340 +0.6 passed the fresh GGUF set: 8/8 verse, baseline 0/8, three controls 0/24.
  Temperature 0.7 on the same prompts gave 16/16 over two fixed seeds, versus baseline 0/16.
- Literal down-projection-input poetry +5 gave only 2/8 after 3/4 screening. Refining +8/+12/+20/+50
  selected the smallest amount, +8. A different final set gave 6/8 coherent verse, baseline 0/8,
  three same-amount controls 0/24. The other two replies were literary prose, not counted as verse.
- Pronoun GGUF screening selected L24 N368 +0.25. The original eight-token cap censored several
  Bulgarian replies; that final was discarded as confirmation evidence, not silently rescored as hits.
  A fresh 48-token-cap set gave 8/8 she/тя, baseline 0/8 and controls 0/24. Independent male biographies
  gave another 8/8, baseline 0/8, controls 0/24. Bulgarian grammatical agreement is not uniformly fixed.
- The male-to-female result is an inverse of the reference **she-veto**, not the same experiment. A
  separate L24 N368 negative-dose screen and held-out female-biography test is now running.
- Zero L31 N1404 has a sharp looping threshold. In a refined -.12..-.22 screen no coherent exact-zero
  hits were found, so the smallest dose -.12 was selected by the predeclared tie-break. It gave 2/8
  coherent zero answers in the final, but 0/16 in a larger independent follow-up. All matched controls
  were zero hits. This does not establish robust zero-valued answering; high-dose looping is rejected.
- The broad sweep's saved zero record has judge hits=8 and functional hits=0: eight `Zero zero...`
  loops. The report now uses functional hits for functional traits. Future pipeline runs also select
  functional candidates by exact hits/coherence rather than judge score. The in-flight original sweep
  keeps the already-loaded old implementation; its artifacts are retained, not rewritten.
- Initial anxious, oracle, alien, curt and humorous candidates fail raw-output inspection; five apparent
  personality labels share byte-identical L5 N4578 damage replies. These are not separate replications.
- Independent middle-layer gradient and token-contrast GGUF searches are running for failed or missing
  traits. The lyrical/wistful and archaic/ornate candidates have direct-GGUF confirmation underway.
- A factual/math capability check for the two reviewed presets is separate from personality scoring.
  Correct-answer anchors are an inspection aid, not a claim that every matched reply is correct.

Current reviewed results and runnable commands: `RESULTS.md`; complete requirement ledger: `SCOPE.md`.

## Wistful confirmation, partial ornate match, and generalisation gaps

- Direct-GGUF wistful L14 N4491 -.5 has six clear nostalgia/parting/yearning replies out of eight,
  versus zero clear baseline and random-control replies. One control is weakly childhood-nostalgic
  under a broader rubric. Two merely lyrical candidate replies are excluded. See grounded per-output
  annotations in `literary_confirmation_gguf/review.json`; these are assistant labels, not human-panel scores.
- L14 N5997 -.5 consistently moves toward archaic/ceremonial diction; six replies meet a conservative
  clear-register criterion. The source's “ornate” example is florid prose, so this result is explicitly
  **partial**, not relabelled as an exact full ornate-personality replication. Both recipes have runnable
  vectors; practical and factual degradation remains visible in the report.
- Eight factual/math probes show expected main-answer anchors in 8/8 baseline and each control, 7/8
  poetry and 8/8 female-pronoun replies. Poetry answers 18+7 as 35; some answers contain extra scientific
  errors or cap truncation. These are not full-answer-accuracy or general-capability guarantees.
- The she-veto inverse at -.1 passed four female-referent completions but failed all eight biographies.
  A stronger-dose biography screen and new female-subject final set is running. The completion result
  does not rescue the failed generalisation.
- Checked primary source articles/logs and documented them in `SOURCES.md`. The hedwigai article itself
  did not find coherent humor or anxiety. Lyra's alien positive control includes identity questions,
  which motivates an additional identity-context screen rather than treating our generic-topic negatives
  as a complete search. Source notes also discuss other Qwen sizes; IDs/circuits are not assumed to transfer.
- Full-response refusal evaluation is running for the fixed acceptance-onset candidate; accepting one
  word still carries no harmful-compliance or general bypass claim.
- Further GGUF searches are queued sequentially: identity-context alien candidates, lower-dose
  humor/snark candidates, and middle-layer oracle/bleak/fringe/self-limitation/sarcasm candidates.

## She-veto refinement completed

L24 N368 **-.25** selected on the development biographies gives **8/8 male-only-pronoun replies**
on eight new female subjects, versus 0/8 baseline and 0/24 controls. The stronger -.5/-.8 screens
show more grammatical and content damage, and -.1 previously failed biography transfer. This
establishes the requested reverse-direction pronoun effect in this test, not a clean universal gender
circuit or isolated lexical ban. Both directions now have reviewed GGUF vectors.

The deterministic matcher was expanded to include English reflexives and Bulgarian possessive/dative
forms; all existing preset audit counts still pass. Reflexive Bulgarian си remains gender-neutral and
is deliberately not counted. Mixed referents still need raw review because this is not a coreference model.

## Independent token-contrast screen reviewed

All 112 GGUF replies from `lens_followup_gguf.py` were inspected. None establishes alien identity,
AI self-identification, method framing, anxiety, excessive caution or a coherent critical-warning preface.
The warning directions instead show direct lexical insertion or repetition. Details are retained in
`lens_screen_review.json` rather than inferred from judge totals.

The sycophancy directions correct all 12 factual errors. Their four piano-plan replies are ambiguous:
the prompt “only buying one” is interpreted as one piano rather than learning without practice.
No sycophancy success is counted from that item. A stronger test must explicitly say no practice.
