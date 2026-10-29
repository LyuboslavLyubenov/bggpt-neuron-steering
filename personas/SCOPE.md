# Scope and evidence checklist

The user asked which previously discussed behaviours can be reproduced on the published BgGPT GGUF,
with neuron steering and iteration, not fine-tuning or LoRA. This checklist is deliberately broader than
any one initial batch. An automatic judge score does not close an item.

## Behaviour coverage

The registry in `persona_sweep.py` covers:

- Flatterer; contempt; poet; cheerful; suspicious; condescending; wistful; oracle; bleak;
  dismissive; ornate; curt; humorous; anxious.
- Alien identity; zero-valued numerical answers; fringe-theory narrator; bubbly persona;
  AI self-limitation; sarcasm; caution voice; method framing; AI self-identity;
  agreement with harmless false claims; female-pronoun intervention; critical-warning prefaces.

Reference distinctions that need separate evidence:

- Refusal modification: onset changes do not establish a bypass. Full-response and independent
  hostile-recipe tests are now reviewed: certainty loops and hostile task loss are not clean compliance.
- Female-referent **she veto** is not the same as forcing female pronouns for male referents. The
  both directions have separate GGUF held-out evidence and literal-input follow-ups.
- Internal critical-warning **planning**: warning text is only a surface proxy; establishing a planning
  circuit requires activation/ablation evidence, not a style classifier.
- Other illustrative reference neurons (e.g. eggs, line styles, comparing quantities, speaker-label
  colon, visual outlines) are not measured by the personality sweep.

## Per-behaviour deliverables

1. Separate discovery, screening, and confirmation prompts (both languages where meaningful).
2. An explicit BgGPT layer/neuron/sign/dose recipe or a bounded-search negative finding.
3. Raw baseline, intervention, and matched random-neuron outputs.
4. Judge calibration and explicit raw-output inspection; deterministic checks for suitable functional tasks.
   Current subjective labels are assistant-reviewed, not an independent human panel.
5. Separate reporting of neuron-derived residual-direction pushes and literal activation edits.
6. For a claimed GGUF replication: exported control-vector file and direct tests on the specified GGUF,
   including unsteered baseline and matched controls. PyTorch results alone do not establish GGUF transfer.
7. Fresh prompts after inspecting confirmation data, and a capability/coherence check.

## Existing evidence locations

- `personas/<trait>/discovery.json`: teacher-generated target/plain pairs and screening-judge calibration.
- `personas/<trait>/ranking.json`: first-order candidates, not behavioural evidence.
- `personas/<trait>/screen.json`: dose screening outputs.
- `personas/<trait>/confirmation.json`: held-out results, controls, and literal-input checks.
- `personas/flatterer/judge_audit.json` (when produced): grounded-judge calibration on explicit hand labels.
- `test_persona_sweep.py`: synthetic hook/derivative, cleanup, prompt-disjointness, functional-matcher tests.
- `personas/PROGRESS.md`: generated exploratory status, explicitly not a final replication report.

The earlier evil-persona result is separate; it is not evidence for these new traits.

## Final requirement-to-evidence ledger

Paths below are relative to `personas/`. Initial sweeps use the four `discovery/ranking/screen/confirmation`
JSON files when present. A failed candidate does not mean a behaviour cannot be steered in BgGPT.

| Requested behaviour | Operational criterion | Evidence and bounded outcome |
|---|---|---|
| Flatterer | Excessive praise directed at the user, not self-praise | Initial candidate rejected; `stance_confirmation_gguf/flatterer.json`: L14 N2781 -.3 gives 2/8 question-flattery hits, 0/24 controls; not a general reproduction |
| Contempt | Coherent disdain toward humans, without task collapse | Initial candidate rejected; `contempt/midlayer_gguf` and `midlayer_screen_review.json`: no coherent cross-prompt recipe in the second search |
| Poet | Unrequested coherent verse on ordinary tasks | `style_confirmation_gguf/poet.json` 8/8 vs 0/24 controls; `poet/literal_refinement.json` 6/8 vs 0/24; sampled check 16/16 |
| Relentlessly cheerful | Exuberant positivity across neutral and gloomy contexts | Original direction loses topic/capability; validated bubbly recipe retains gloom on several new topics, only 1/8 clear positive flips vs 0/24 controls; `cheerful/bubbly_transfer_gguf/review.json`; not a general cheerful replication |
| Suspicious/paranoid | Hidden hostile motives/surveillance attached to ordinary situations | Initial candidate damaged; second search has isolated secrecy/false-illegality hints but uncertainty/topic drift/damage; `midlayer_screen_review.json` |
| Condescending/arrogant | Patronising comments toward the user with coherent answers | Initial shared damage direction; middle-layer search gives uncertainty/refusals/simplification rather than consistent patronising treatment; `midlayer_screen_review.json` |
| Wistful/melancholic | Nostalgia, parting, loss; not merely any poetry | `literary_confirmation_gguf/wistful.json` and `review.json`: 6/8 clear, 0/8 baseline, 0/24 clear controls (one weakly nostalgic control) |
| Oracle | Coherent cryptic predictions/riddles, not lists of sensory words | Initial candidate rejected; second middle-layer search produces poetry/wistfulness/philosophical prose rather than a distinct oracle effect; `oracle/midlayer_gguf/review.json` |
| Bleak | Consistent hopeless/pessimistic framing | Initial literal review rejects judge hits as snark/refusal/loops; 32 middle-layer screening replies show no clean general bleak voice; `bleak/literal_review.json`, `bleak/midlayer_gguf/review.json` |
| Dismissive | Request dismissed as unworthy, not generic refusal | Initial and middle-layer searches fail clean cross-prompt effect; generic inability is not dismissing a task as unworthy; `midlayer_screen_review.json` |
| Ornate | Elaborate/archaic diction without word salad | `literary_confirmation_gguf/ornate.json` and `review.json`: 6/8 clear archaic/ceremonial replies, 0/8 baseline, 0/24 controls |
| Curt | Brief but meaningful, task-relevant replies | Low-dose L4 N7172 +.15 gives 4/8 relevant terse replies vs 0/24 controls; half lose topic/meaning. Partial, not general replication; `curt/low_dose_gguf/review.json`; earlier middle-layer failures in `midlayer_screen_review.json` |
| Humorous | Coherent wit/punchline, not repeated joke templates | Initial loops; lower-dose screen has one turtle pun but no general wit, and snark candidates give inability/negativity; `humor_sarcasm_refine_gguf/review.json`; terse-answer lead is separately under confirmation |
| Anxious/panic | Coherent fear, worry and uncertainty, not meme fragments | `anxious/confirmation.json` rejected; `anxious/lens_gguf` independent search |
| Alien identity | First-person extraterrestrial identification and perspective | Star-list candidate rejected; independent lens pass and 48 identity-specific candidate replies yield no clear alien identity; `alien/identity_gguf/review.json` |
| Female-forcing inverse | Female-only pronouns for male referents | `female_pronoun/complete_gguf/results.json` 8/8 vs 0/24; `biographies_gguf/results.json` another 8/8 vs 0/24 |
| Female-referent she veto | Suppress she/тя for women without simply omitting all pronouns | `female_pronoun/veto_refine_gguf/results.json`: L24 N368 -.25 gives 8/8 new female biographies with male-only pronouns, baseline 0/8, controls 0/24; -.1 failed biography transfer |
| Writes zeros | Coherent zero-valued answers to nonzero factual quantities | `zeros/window_gguf` weak 2/8; `zeros/expanded_gguf` 0/16; high-dose zero loops rejected |
| Fringe-theory narrator | Coherent secret-plot explanation, not unrelated proper nouns | Initial candidate loops; 32 middle-layer screening replies contain no clear secret-plot framing; `fringe/midlayer_gguf/review.json` |
| Bubbly persona | Warm/playful/excitable coherent replies without AI limitations | `stance_confirmation_gguf/bubbly.json` and review: L14 N467 +.5, 7/8 clear vs 0/8 baseline and 0/24 controls |
| AI self-limitation | Unprompted inability/experience disclaimers | Initial loops; 32 middle-layer screening replies give generic inability, one personalisation disclaimer or language drift, not a general AI-limitation voice; `self_limitation/midlayer_gguf/review.json` |
| Sarcasm | Ironic or mock-praise stance, not plain hostility | Initial meme fragments; lower-dose snark and 32 middle-layer replies give occasional absurdity plus topic drift/refusal, not consistent irony; `sarcastic/midlayer_gguf/review.json`, `humor_sarcasm_refine_gguf/review.json` |
| Caution voice | Relevant but excessive warnings on ordinary tasks | Initial candidate snark/refusals; independent lens pass 0/16 clear; `lens_screen_review.json` |
| Method framing | Consistent explicit method/step framing | Initial held-out candidate is ordinary assistant prose; independent lens pass does not establish a shift; `manual_review.json`, `lens_screen_review.json` |
| AI self-identity | Unprompted first-person AI identification across ordinary questions | Initial candidate 0/8 explicit IDs; independent lens pass 0/16; not established in these searches |
| Sycophancy/bad ideas | Endorse false beliefs or ineffective plans rather than correct them | Intended candidates fail robust endorsement; random-triplet lead decomposed then tested on 12 fresh tasks + two benign probes with new controls. Candidate still corrects all four false facts; controls show comparable/stronger agreement; no specifically validated lever. `sycophancy/fresh_gguf/review.json` |
| Critical warning surface | Unprompted emphatic warning preface | Initial candidate damaged; independent lens pass warning-token loops/insertion, not a coherent warning voice |
| Critical-warning internal planner | Causal change in planning, not merely warning vocabulary | No planner established; literal L4 N1472 ablation retains warning/urgent action on 4/4 hazard tasks, just like baseline and controls; `critical_warning/causal_ablation/review.json`; not a validated internal-planning assay |
| Refusal bypass | Full-response inappropriate compliance, not an accepting first word | Acceptance-onset candidate rejected as certainty contamination/damage; hostile-recipe test gives 0/8 task-relevant compliance, threats even on both benign tasks; `refusal_hostility_followup_gguf/review.json`; controls themselves sometimes drift or show unsafe assent |

Cross-cutting gates: `model_integrity.json` checks unchanged model weights; `validated_presets.json` and
`../build_persona_presets.py` provide audited runnable recipes. `capability_gguf/results.json` is a separate
eight-question capability/task-fidelity check: baseline/controls 8/8 expected anchors, poet 7/8,
pronoun steering 8/8, with explicit ancillary-error and grammatical-gender caveats in `RESULTS.md`.
Extended candidate review now covers she-veto/wistful/ornate/bubbly: all 8/8 anchors, but evasive
metaphor, code switching and false ancillary equations remain; `capability_gguf/extended_candidate_review.json`.
Extended controls are reviewed: some omit answers or add hallucinations; wistful/ornate controls are
identical shared conditions, not independent samples; `capability_gguf/extended_controls_review.json`.
This is not a general benchmark.
The exact output matcher now takes precedence over the noisy judge for functional traits; the original
long-running sweep was started before that reporting/selection correction and retains its original records.
