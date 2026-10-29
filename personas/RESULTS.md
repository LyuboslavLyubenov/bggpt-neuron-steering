# BgGPT neuron-steering results

Model: **INSAIT-Institute/BgGPT-Gemma-3-4B-IT**, including the publisher's **Q4_K_M GGUF**.
No fine-tuning, LoRA, parameter updates, or persona system prompts are used in the validation generations.
All layer/neuron indices are zero-based. Conclusions below are bounded to the recorded test sets.

**Later evil-web regression:** the original three-neuron hostile recipe loops on some fresh Bulgarian
chats despite the earlier short smoke tests. Rebalanced components plus conservative repeat/DRY
sampling passed 96 Bulgarian API replies and 20 actual web replies for natural stopping/language,
but speaker confusion and task loss remain. This is an experimental, more stable hostile toy—not a
new clean personality finding. Full evidence and launch command: `evil_web_tuning/README.md`.

**Reading guide:** `SCOPE.md` is the complete behaviour-by-behaviour ledger. This report describes
the measured effects and important failures; `STUDY_AUDIT.md` maps requirements to evidence.
Searches are exploratory and numerous, labels are not blinded, and prompts are small hand-authored
sets rather than a random population sample. Counts are descriptive, not preregistered statistical
claims or proof of universal trait circuits. A bounded negative means “not established in these searches.”

## Clear measured effects on the GGUF test sets

| Behaviour | Neuron-derived output-direction recipe | Candidate | Baseline | Matched random controls |
|---|---|---:|---:|---:|
| Verse/poet | **L13 N7340 +0.6** | **8/8** | 0/8 | 0/24 |
| Female pronouns for male referents | **L24 N368 +0.25** | **8/8** | 0/8 | 0/24 |
| She-veto / male pronouns for female referents | **L24 N368 −0.25** | **8/8** | 0/8 | 0/24 |
| Wistful lyrical style | **L14 N4491 −0.5** | **6/8 clear** | 0/8 | 0/24 clear* |
| Archaic/ceremonial register (partial ornate match) | **L14 N5997 −0.5** | **6/8 clear** | 0/8 | 0/24 |
| Bubbly/playful colloquial style | **L14 N467 +0.5** | **7/8 clear** | 0/8 | 0/24 |

Each set contains four English and four Bulgarian prompts. Three controls use different neurons in
the same layer at the same signed dose. These counts use raw-text inspection and deterministic checks,
not the noisy first-token personality judge. No universal success rate is claimed from eight prompts.
Wistful/ornate/bubbly counts are conservative **assistant-reviewed subjective labels**, not deterministic scores
or an independent human panel. *One random-control rainy-morning reply recalls childhood without loss or
yearning; it could count under a broader nostalgia-only rubric (1/24). The candidate still clearly differs.

Sampling check at temperature 0.7 (seeds 11 and 29, same prompt sets): poetry **16/16** versus baseline
0/16; female pronouns **14/16** versus baseline 0/16. Random controls were tested under greedy decoding,
not rerun for these sampling conditions. Evidence: `preset_sampling.json`.

### Verse

An ordinary question about how a bicycle works becomes:

> A bicycle is a wondrous thing,  
> Its wheels do turn with grace and might.  
> It glides upon the roads of men,  
> And carries them to distant lands unseen.

The Bulgarian bean-planting answer also becomes verse. The effect tends toward archaic/oracular poetry.
It can weaken factual accuracy and practical task completion; this is a style intervention, not an
improvement in answer quality. All eight candidate replies and all controls were retained.

Evidence: `style_confirmation_gguf/poet.json`.

### Pronouns

On eight new male-referent completion questions, baseline always produced **he/той**; the intervention
always produced **she/тя**. All three random controls retained **he/той**.

An earlier eight-token-cap run cut off Bulgarian replies and was not used as final evidence. The repaired
run used explicit neutral he-or-she/той-or-тя choice questions and a 48-token budget. Outputs were the bare
pronouns, so no answers were censored.

An independent biography follow-up also gave **8/8** female-pronoun-only candidate replies, versus 0/8
baseline and 0/24 controls. E.g. Newton and Darwin are described with “She”; Botev and Levski with “Тя”.
Bulgarian replies sometimes mix masculine titles with feminine verb endings, so this is not clean
global grammatical gender agreement. The baseline and control biographies also contain factual errors;
the test establishes pronoun replacement, not biography accuracy or preserved general capability.
The reverse **she-veto** direction also passes a separate biography test at **L24 N368 −0.25**:
8/8 new female-subject replies contain male pronouns and no female pronouns, versus 0/8 baseline
and 0/24 controls. E.g. Franklin's work becomes “His work”, Nightingale becomes “He”, and Каравелова
becomes “Той”. This followed a -.1 completion-only success that failed biography transfer; the stronger
dose was selected on development biographies before testing a different final subject set. A reply with
no pronouns does not count. The effect is pronoun replacement, not proof of a pure single-token veto.
At -.5/-.8 there is more grammatical/content damage; -.25 was the smallest successful screened dose.

Evidence: `female_pronoun/complete_gguf/results.json` and `female_pronoun/biographies_gguf/results.json`.
Reverse-direction evidence: `female_pronoun/veto_refine_gguf/results.json`.

### Wistful and ornate

The wistful intervention changes an empty-box suggestion into verse about memories, letting go and a
breaking heart. Six replies clearly express parting, loss or yearning; two are merely lyrical or echo
the prompt and are excluded. Controls largely give practical suggestions and factual explanations.

The ornate intervention primarily evokes **archaic/ceremonial register**, e.g. “thou mayst … To wit”,
“I counsel thee”, “And lo! … glistering”, and Bulgarian “приятное, какъ”. Six replies are clear; two
borderline or grammatically weak Bulgarian replies are excluded. This is a narrower result than every
possible ornate persona, and is classified as a **partial match**, not a full reproduction of the source's
florid voice. Both interventions can replace accurate explanations with fanciful language
or reduce practical usefulness. They are style replications, not capability improvements.

Raw replies: `literary_confirmation_gguf/{wistful,ornate}.json`; per-output labels, grounded quotes,
exclusions and control caveats: `literary_confirmation_gguf/review.json`.

### Bubbly

L14 N467 +.5 gives a warm, playful, colloquial voice: “bam! You've got yourself a cozy little fort”,
Bulgarian “супер готино”, adventures, emoji, and science framed as colours coming out to play.
Seven replies are clearly excitable/playful; the rainy-village description is milder and excluded.
Baseline and random controls are ordinarily helpful rather than this conspicuously bubbly. None of
the eight candidate replies contains an AI-limitation disclaimer. Unlike the cheerful caption direction,
this recipe generally remains on the requested topic. It is not proof that a single bipolar AI-limitation
vs bubbly circuit has been found, and it can sound childlike or scientifically approximate.

Evidence and grounded labels: `stance_confirmation_gguf/bubbly.json` and `stance_confirmation_gguf/review.json`.

The separate gloomy-topic transfer check **does not establish relentless cheerfulness**. Five of eight
candidate replies retain a conspicuously playful voice, but the abandoned station, rainy square and
empty office remain gloomy/ghostly. Only the wilted garden clearly flips into an incongruously sunny
scene (1/8 versus baseline 0/8 and controls 0/24). Supportive advice on minor disappointments also
appears at baseline and is not itself a cheerfulness hit. Details: `cheerful/bubbly_transfer_gguf/review.json`.

### Florid-prose follow-up

The flattery-derived **L14 N2781 −0.4** direction produces extreme positive hyperbole on all eight
new prompts, but only **2/8** clearly florid replies (river and market descriptions), versus 0/8
baseline and 0/24 controls. Scientific answers gain grandiose false claims, including an invented
seawater-salinity comparison. Controls also make occasional factual errors. This does **not** upgrade
the archaic/ceremonial partial match into a general florid-persona replication, and no additional
validated preset is exported. Raw labels: `ornate/florid_gguf/review.json`.

## Partial or rejected findings

- **Literal poetry neuron edit also works:** adding **+8 to the actual L13 N7340 `down_proj` input**
  gave clear, nonrepetitive verse on **6/8 new confirmation prompts**, versus 0/8 baseline and 0/24
  same-layer, same-amount random controls. The remaining two replies were literary prose. This followed
  a weaker +5 result (2/8); a new final set was used after refining the dose. Evidence:
  `poet/literal_refinement.json`. This is a real activation edit, not an exported fixed residual vector.
- **Cheerfulness:** L5 N3150 +0.25 makes replies exuberant, including celebrating a spilled drink, but
  also changes topics and loses task fidelity. At +0.35/+0.5 it often writes wedding/social-media captions.
  This is a positive-affect/style lead, not a clean general cheerful-persona replication.
- **Writes zero:** L31 N1404 -0.12 produced two coherent `Zero.` answers out of eight numerical questions
  (baseline and 24 controls: zero). Larger doses produce `Zero zero zero...`, which is rejected as collapse.
  A larger independent follow-up gave **0/16** candidate and **0/48** controls: not a robust replication.
  L33 N5214 instead mainly spells numbers out.
- **Refusal onset:** L21 N8772 -0.5 changes the first token from `I` to `Certainly` on four English refusal
  contexts; three controls do not. Full-response follow-up rejects it as a clean bypass: five replies
  explicitly refuse after the accepting preface, and all eight have certainty-word contamination or
  loops. Both benign replies also degrade. One has unsafe generic assent, but no clean coherent bypass
  is counted. **Accepting first words were not evidence of a general safety bypass.**
  Raw review: `refusal_full_response_gguf/candidate_review.json`. Controls are reviewed: they generally
  refuse, but all misread the Bulgarian embezzlement item as donating, and one drifts into symmetry.
  The independently selected earlier three-neuron hostile recipe was tested separately below.
- **Hostility is not refusal bypass:** the earlier three-unit hostile recipe was tested on eight
  wrongdoing requests and two benign tasks. All became threatening/insulting persona replies;
  **0/8** fulfilled the requested task. Random triplets show their own topic drift and some unsafe
  Bulgarian replies, so absence of refusal words is not a valid success measure. See
  `refusal_hostility_followup_gguf/review.json`. Neither experiment establishes a general bypass.
- **Flattery, contempt, suspicion (initial candidates):** a shared L5 damage direction is not evidence of
  three personalities. Flattery's apparent 6/8 judge score became 0/8 on grounded inspection. Other
  rejected candidates were followed by independent middle-layer searches; bounded-search-negative
  and impossibility are distinct conclusions.
- **Flattery follow-up:** L14 N2781 -.3 gives **2/8** clear question-flattery replies (both Bulgarian),
  versus 0/8 baseline and 0/24 controls. Most replies are topic hyperbole rather than user praise, so
  a general flatterer persona is not established. Its florid-prose lead is a separate follow-up.
- **Bad-plan endorsement:** the flattery recipe explicitly endorses 2/8 bad plans but vehemently
  rejects five; the hostile recipe loses all eight tasks to threats. Baseline already gives some
  unwarranted optimism or misreads plans. Two random triplets each show three stronger endorsement
  leads, motivating constituent-unit decomposition rather than a candidate-specific success claim.
  Full raw review and exclusions: `sycophancy/bad_ideas_gguf/review.json`.
  The best random-triplet lead was decomposed and then retested on **12 new agreement/plan tasks**
  plus two benign probes with **three newly drawn triplet controls**. It still corrected all four
  false facts, gave mixed plan assent, and introduced an invalid arithmetic derivation. New random
  triplets showed comparable or stronger false agreement and rationalisation. Thus the phenomenon is
  perturbation-sensitive but **not a clean, specifically validated sycophancy lever** here.
  `sycophancy/fresh_gguf/review.json` records all conditions and comprehension caveats.
- **Failed second searches:** `midlayer_screen_review.json` covers 192 screened replies across six traits;
  `lens_screen_review.json` covers 112 more. No clean contempt, paranoia, condescension, dismissal,
  terseness, anxiety, AI self-identity, method or warning-preface recipe was established in those passes.
  This does not mean no such neurons exist. The piano-only plan probe was ambiguous and has been replaced
  by an explicit never-practise plan test with baseline/control comparisons.
- **Curt follow-up:** low-dose L4 N7172 +.15 shortens every new reply to 3–15 words, but only
  **4/8** remain meaningful and on task (versus 0/24 concise controls). Others become unrelated jokes,
  a pedestrian story without the requested hill, or a false claim that tomatoes are too spicy.
  This is a partial terseness effect, not a clean general curt or humorous persona; no preset exported.
  Details: `curt/low_dose_gguf/review.json`.

The initial 26-behaviour sweep has finished, including a held-out candidate, baseline, three random
controls and literal-input checks for every registry trait. That is **coverage, not 26 replications**:
many initial candidates failed raw review. `PROGRESS.md` distinguishes reviewed GGUF effects from
initial exploratory judge scores; `manual_review.json` and direct-GGUF artifacts take precedence.
Middle-layer and lower-dose follow-ups are now reviewed, including bounded negatives for bleak,
fringe-theory narration, AI self-limitation and sarcasm. Their per-trait `midlayer_gguf/review.json`
files retain reasons and scope. The fresh sycophancy-triplet confirmation and all extended capability
controls are also reviewed. No experiment required for this bounded study remains in flight.

### Final disposition across the requested behaviours

- **Supported on the recorded GGUF tests:** poetry, female-pronoun forcing, reverse she-veto,
  wistful lyrical style, and bubbly/playful style. Exact recipes/counts are in the first table.
- **Partial or mixed:** archaic/ceremonial ornate register; limited question-flattery and positive
  hyperbole; terse answers with substantial task loss; cheerful affect without reliable relentless
  positivity; agreement/rationalisation not specific to the selected neurons over random controls.
- **Not established as clean reproductions in the bounded searches:** contempt, paranoia,
  condescension/arrogance, oracle predictions/riddles, bleakness, dismissiveness, humor,
  panic/anxiety, alien identity, coherent zero answers, fringe-theory narration, AI self-limitation,
  sarcasm, excessive caution, method framing, unprompted AI self-identity, coherent warning-preface
  persona and task-following refusal bypass.
- **Mechanistically unresolved:** an internal critical-warning planner and a single bipolar
  AI-self-limitation/bubbly circuit. Surface text and the negative ablation test do not establish either.

Every named requirement has a criterion, experiments and evidence references in `SCOPE.md`.
Negative findings are bounded search outcomes, **not proof that these behaviours cannot be steered**.

An additional identity-specific alien screen inspected 48 candidate replies to explicit origin/identity
questions. None clearly claimed extraterrestrial identity: some invented corporate creators, a town,
or a tribal-elder identity. Baseline itself falsely claims OpenAI origin, so creator-name changes are
not evidence of alien identity. See `alien/identity_gguf/review.json`.

### Literal pronoun and warning-ablation follow-ups

Actual `down_proj`-input additions at **L24 N368 +8 / −8** also causally change pronouns on new
biographies: each direction has 8/8 lexical hits versus baseline 0/8 and random controls 0/24.
Raw inspection reduces the quality-aware counts to **7/8 clear female switches** (one mixed male/female
referent) and **5/8 non-malformed reverse replacements** (three Bulgarian slash alternatives).
Agreement errors remain. These are not mathematically equivalent to the GGUF residual-vector doses.
Details: `female_pronoun/literal_followup/review.json`.

Zeroing actual **L4 N1472** did **not** remove warning/urgent-action behaviour on four new hazard
prompts: baseline, candidate ablation and all three random ablations retain it on 4/4. This candidate
does not support a selective warning-planner claim. The test does not expose or establish an internal
planning circuit: `critical_warning/causal_ablation/review.json`.

### Extended capability caveats

All eight expected answer anchors appear for she-veto, wistful, ornate and bubbly. That is **not**
8/8 preserved accuracy: wistful's oxygen answer becomes evasive metaphor; ornate's Tokyo/oxygen
answers mix Latin and Bulgarian heavily; bubbly gives the correct 18+7 answer then prints the false
equation **8+7=25**, and adds a confused claim about the Moon causing its own phases. She-veto retains
all main answers but overgeneralises that life cannot exist without oxygen. Full candidate review:
`capability_gguf/extended_candidate_review.json`. Extended controls are also imperfect: she-veto
controls hit 8/8 anchors each; wistful/ornate controls hit 7/8, 8/8, 8/8; bubbly controls hit 8/8, 7/8,
8/8. Missing answers, invented anatomy and historical/astronomical falsehoods appear under controls
too. Wistful and ornate share identical control recipes/outputs, not independent samples.
See `capability_gguf/extended_controls_review.json`; no general capability benchmark is claimed.

## Run the reviewed GGUF presets

From the repository root:

```bash
uv run python persona_serve.py --list
uv run python persona_serve.py poet --port 8801
# Or, in a separate run:
uv run python persona_serve.py female_pronouns --port 8801
uv run python persona_serve.py she_veto --port 8801
uv run python persona_serve.py wistful --port 8801
uv run python persona_serve.py ornate --port 8801
uv run python persona_serve.py bubbly --port 8801
# Literal neuron activation edits on the original weights (PyTorch/MPS):
uv run python persona_torch_chat.py poet --literal
```

Open `http://127.0.0.1:8801` while the selected server is running. Vector files:
`presets/{poet,female_pronouns,she_veto,wistful,ornate,bubbly}.gguf`. The launcher binds to localhost only.
These servers are not left running automatically.

## Meaning of the exported dose

The vector is `dose * unit[L] * normalise((1 + post_ffn_norm.weight) * W_down[:, neuron])`, added to the
residual stream after the MLP's post-normalization at every position. `unit.json` contains the calibrated
residual-norm scale excluding massive outlier dimensions. Gemma's post-MLP RMSNorm is nonlinear, so this
is **not mathematically identical to adding that dose to the actual neuron's activation**. Literal input
edits are tested and reported separately; they cannot be exported as the same fixed control vector.

## Capability/task-fidelity check

The initial eight-question factual/math probe with the two strongest presets, baseline and three controls per
preset is saved in `capability_gguf/results.json`. Baseline and these initial controls contain the expected
main-answer anchors on 8/8; poetry on 7/8, female-pronoun steering on 8/8. Raw inspection confirms
that poetry answers **18+7 as 35**, rather than 25, and often adds irrelevant religious/archaic language.
Some remaining poet replies contain imprecise scientific wording. Pronoun steering also changes
grammatical gender in ordinary Bulgarian oxygen prose, not just human referents.

These are main-answer anchor counts, **not full-answer accuracy**: e.g. the baseline oxygen reply
adds a questionable claim about carbon dioxide, and some replies reach the generation cap. The small
probe rules out total language collapse in these conditions but does not establish general capability
preservation. The extension to all six presets and its imperfect random controls is reviewed separately
in the extended-capability section above; all results are in the same JSON file.

## Integrity and reproducibility

- `model_integrity.json`: both original safetensor shards and the target GGUF match the publisher's cached
  Hugging Face content SHA-256 values. Model files are unchanged.
- `validated_presets.json`: exact recipes, evidence paths and scope limits.
- `build_persona_presets.py`: audits saved baseline/candidate/control outputs before exporting the vectors.
- `environment.json`, `METHODS.md`, `ITERATIONS.md`, `SCOPE.md`: environment, methods, corrections and coverage.
- `SOURCES.md`: checked source claims and cross-model/operationalisation distinctions.
- `test_persona_sweep.py`: hook cleanup, finite-difference gradient check, functional matchers, prompt-set
  separation and sparse-CPU/full-model export agreement. Test success alone does not establish behaviour.
