# Study audit — completed bounded investigation

## Requested end state

Evaluate every named hedwigai personality and lyra reference lever on the local published BgGPT
GGUF, using activation edits only; iterate candidates; compare held-out outputs with unsteered and
matched random-neuron controls; report reproductions, partial matches and bounded negatives without
counting damaged text or unreliable judge scores as success.

## Requirement → concrete evidence

| Requirement | Evidence inspected | Status / limitation |
|---|---|---|
| Local target model, no training/LoRA | `common.py:load` freezes every parameter and uses eval mode; runtime hooks edit activations; matching upstream SHA-256 for both safetensor shards and Q4_K_M GGUF | All three hashes recomputed at final audit on 2026-10-29: unchanged |
| All named personalities/reference behaviours | `SCOPE.md` maps every explicit behaviour, including separate reverse she-veto, bad-plan endorsement, full-response refusal and internal-planner caveat | Every named behaviour has a bounded outcome and evidence references; mechanistic claims not established are explicitly unresolved |
| Discovery and iteration | All 26 registry traits have discovery/ranking/screen/confirmation JSON; independent token-contrast and middle-layer screens, functional refinements and targeted failure-driven follow-ups | `coverage_audit.json` verifies structural coverage only |
| Held-out prompts and matched controls | Initial 26 traits each have eight candidate/baseline outputs, three same-layer/dose random controls, two literal-dose checks; new targeted sets have fresh prompts and newly drawn controls | 22 implementation/integrity tests pass; original confirmation JSON relies on registry prompt provenance rather than embedding prompts |
| Direct GGUF success evidence | Six exported presets have saved target-GGUF outputs, matched controls, explicit functional or quote-grounded style labels | `build_persona_presets.py` checks recipes, outputs, evidence and controls; subjective labels remain assistant-reviewed |
| Literal neuron vs residual direction | `common.py`, literal poetry and pronoun experiments, warning ablation, exact mathematical convention in `RESULTS.md` | Distinct interventions, never equated through Gemma RMSNorm |
| Reject damage and noisy judges | `manual_review.json` and per-experiment `review.json`; raw reports supersede initial judge totals | Repetition, task loss, warning-word insertion and unanchored worship excluded |
| Test semantic/task fidelity | Small factual probe plus raw topic/coherence review; extended candidate/control reviews document false equations, evasive answers and hallucinations despite anchor hits | Not a general capability benchmark; controls themselves sometimes degrade |
| Source fidelity | `SOURCES.md` separates actual source claims, cross-model references and internal-planner hypotheses | No assumed transfer of Qwen unit IDs; humor/anxiety were not clean source successes |
| Reproducibility | Saved recipes, raw outputs, model commits/hashes, control vectors, scripts; `uv run python persona_serve.py --list` lists six presets | No new preset for weak or damaged effects |

## Gates rerun

- `uv run python -m unittest -q test_persona_sweep`: **22 passed**.
- `uv run python audit_persona_study.py`: **26 initial traits with matched controls; six preset evidence audits; six targeted confirmation/control structure checks**.
- `uv run python persona_serve.py --list`: six expected reviewed recipes listed.
- `uv run python build_persona_presets.py`: passed with the stricter matched-layer/dose control audit;
  all six vectors rebuilt successfully from unchanged original weights.
- `uv run python persona_report.py`: regenerated the explicitly exploratory progress table with links
  to final conclusions and later follow-up dispositions.
- Python compile checks: passed.

## Completion disposition

The final fresh sycophancy study has all 70 outputs reviewed, including three newly drawn random
triplets; the candidate does not show specific reliable agreement over those controls. All other
targeted jobs and reviews are finished. Results, ledger, progress and exported recipes are reconciled.
No required experiment remains in flight for this bounded investigation.

A green structural audit is not the behavioural proof: positive claims additionally rely on the
saved raw outputs and explicit grounded labels; negatives retain their search limits. The objective
was to determine what reproduced in these local searches, not to prove absence of all possible neurons.

Completed since the first audit: all remaining middle-layer screens reviewed; low-dose curt direction
has only 4/8 task-relevant terse answers, not a clean general reproduction; extended capability
controls and full-response refusal controls inspected, with their own errors documented.
