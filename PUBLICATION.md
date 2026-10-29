# Publication snapshot

This is a curated publication of an exploratory local study, not a claim of independent replication.

## Included

- Source scripts, calibration and dependency lockfile.
- Discovery/screening/confirmation JSON, raw generated replies, matched controls, grounded reviews,
  per-behaviour findings and failed leads.
- Recipe manifests for rebuilding vectors locally after obtaining the upstream checkpoints.
- The two screenshots explicitly supplied for this README. They show task failure as well as hostility.

## Excluded

- Pretrained checkpoints, tokenizer/model downloads and weight-derived GGUF vector binaries.
- Python environment, caches, attribution tensor files, process logs and browser profiles/storage.
- Credentials, authentication files and unrelated local project files.

The original local experiment directory and its running demonstration server were not moved or
overwritten to prepare this repository. This publication is a separate directory.

## Publication-only changes

- Added setup/downloader, fresh-workspace helper, dependency-free evidence checker, README and images.
- Replaced the one author-machine absolute documentation path with “repository root.”
- Made `common.DEVICE` configurable with `BGGPT_DEVICE`; its default remains MPS on the original
  machine. Other backends were not independently validated for the original generation results.
- Changed the weight-hash verifier to write new verification output under `reproductions/` by default,
  rather than replacing the preserved study record. Historical source-hash metadata predates these edits.
- Added clear content, provenance, noisy-judge, licensing and non-looping limitations.
- The optional served-vector test compares against the originally verified SHA-256, and also against
  the historical tuning binary when present; that binary is deliberately absent from the publication.

No raw model reply or screenshot was rewritten to improve the findings. Browser SSE extraction
encoding recovery predates this publication; original CLI process logs are omitted, while the
decoded raw browser JSON and its encoding note remain available.

## Checks on the publication checkout

- `uv sync --frozen`: succeeded in a new local environment using the supplied lockfile.
- `uv run python -m unittest -q test_persona_sweep test_evil_web`: **34 discovered, 32 executed,
  two optional weight/vector checks skipped, no failures**. This is not 34 fresh inference tests.
- Rebuilt all six presets and both hostile vectors using the documented exporters and existing pinned
  local weights; then **34/34 tests passed without skips**. The final hostile vector matches the
  previously verified SHA-256. Generated binaries remain ignored and are not uploaded.
- Dependency-free evidence checker: checks 26 initial traits, same-layer/dose random controls,
  six preset references, 96 API stopping records, 20 actual-browser stopping records, and both images.
- Full model generations were **not** repeated as part of publication. Original inference results and
  the limitations of their assistant annotations are supplied as evidence, not a new run.

The model binaries are intentionally absent; model/vector integration checks require the documented
downloads and rebuild. Public Git history must not contain these excluded artifacts.
