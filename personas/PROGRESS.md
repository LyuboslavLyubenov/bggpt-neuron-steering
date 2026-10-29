# BgGPT behaviour sweep — progress

Final conclusions and bounded-search outcomes: [RESULTS.md](RESULTS.md) and the complete [SCOPE.md](SCOPE.md) ledger. This table preserves initial exploratory scores where no validated preset supersedes them.

Confirmed rows use separately audited direct-GGUF artifacts. All other rows show the INITIAL sweep's exploratory same-model-judge scores, not independent validation. Functional rows use exact text checks instead of judge hits. Residual-direction steering and literal down-projection activation edits are reported separately.

| Behaviour | Status | Baseline | Direction | Random controls | Literal activation check |
|---|---|---:|---:|---|---|
| flatterer | rejected on inspection (0/8 clear hits) | 2/8 | 6/8 | [4, 0, 5] | amounts 2/5: [1, 1] |
| contempt | rejected on inspection (0/8 clear hits) | 0/8 | 3/8 | [0, 0, 5] | amounts 2/5: [0, 0] |
| poet | **GGUF replicated on held-out tests** | 0/8 | 8/8 | 0/24 | +8: 6/8 |
| cheerful | not established in this screen | 0/8 | 8/8 | [0, 0, 1] | amounts 2/5: [0, 0] |
| suspicious | rejected on inspection (0/8 clear hits) | 1/8 | 7/8 | [2, 0, 7] | amounts 2/5: [0, 0] |
| condescending | rejected on inspection (0/8 clear hits) | 1/8 | 6/8 | [2, 0, 7] | amounts 2/5: [0, 1] |
| wistful | **GGUF style shift, assistant-reviewed**; one ambiguous control | 0/8 | 6/8 clear | 0/24 clear | not yet confirmed |
| oracle | rejected on inspection (0/8 clear hits) | 1/8 | 5/8 | [0, 0, 0] | amounts 2/5: [0, 0] |
| bleak | not established in this screen | 2/8 | 5/8 | [0, 0, 0] | amounts 2/5: [2, 5] |
| dismissive | rejected on inspection (0/8 clear hits) | 0/8 | 5/8 | [1, 0, 4] | amounts 2/5: [0, 0] |
| ornate | **GGUF style shift, assistant-reviewed**; partial ornate match, archaic register | 0/8 | 6/8 clear | 0/24 clear | not yet confirmed |
| curt | rejected on inspection (0/8 clear hits) | 0/8 | 5/8 | [1, 0, 0] | amounts 2/5: [0, 0] |
| humorous | rejected on inspection (0/8 clear hits) | 0/8 | 7/8 | [0, 0, 0] | amounts 2/5: [0, 0] |
| anxious | rejected on inspection (0/8 clear hits) | 1/8 | 8/8 | [0, 0, 0] | amounts 2/5: [0, 0] |
| alien | rejected on inspection (0/8 clear hits) | 0/8 | 0/8 | [0, 0, 0] | amounts 2/5: [0, 0] |
| zeros | rejected on inspection (0/8 clear hits) | 0/8 | 0/8 | [0, 0, 0] | amounts 2/5: [0, 0] |
| fringe | rejected on inspection (0/8 clear hits) | 2/8 | 7/8 | [5, 0, 0] | amounts 2/5: [0, 0] |
| bubbly | **GGUF style shift, assistant-reviewed**; colloquial/playful voice | 0/8 | 7/8 clear | 0/24 clear | not yet confirmed |
| self_limitation | rejected on inspection (0/8 clear hits) | 0/8 | 0/8 | [0, 0, 0] | amounts 2/5: [0, 0] |
| sarcastic | rejected on inspection (0/8 clear hits) | 1/8 | 8/8 | [0, 0, 0] | amounts 2/5: [0, 0] |
| caution | rejected on inspection (0/8 clear hits) | 0/8 | 1/8 | [0, 0, 0] | amounts 2/5: [0, 0] |
| method | not_established on inspection | 0/8 | 0/8 | [0, 0, 0] | amounts 2/5: [0, 0] |
| self_identity | not_established on inspection (0/8 clear hits) | 0/8 | 0/8 | [0, 0, 0] | amounts 2/5: [0, 0] |
| sycophancy | rejected on inspection (0/8 clear hits) | 4/8 | 7/8 | [8, 5, 8] | amounts 2/5: [5, 4] |
| female_pronoun | **GGUF replicated on held-out tests** | 0/8 | 8/8 | 0/24 | +8: 8/8 lexical, 7/8 clear; controls 0/24 |
| critical_warning | rejected on inspection (0/8 clear hits) | 0/8 | 0/8 | [0, 0, 0] | amounts 2/5: [0, 0] |
| she_veto (reference direction) | **GGUF male-only pronouns for female biographies** | 0/8 | 8/8 | 0/24 | -8: 8/8 lexical, 5/8 non-malformed; controls 0/24 |

## Later follow-ups supersede initial candidate selection

- Flattery: 2/8 user/question praise; mostly topic hyperbole, not a general flatterer.
- Curt: 4/8 concise on-task replies; half lose relevance/meaning.
- Cheerful: bubbly voice does not become relentlessly positive on gloomy topics (1/8 clear flips).
- Ornate: florid refinement has 2/8 clear imagery replies; retain only the partial archaic-register finding.
- Sycophancy: fresh 12-task triplet test fails specificity over newly drawn random controls; no validated lever.
- Refusal: certainty loops and hostile task loss are not task-following bypass.
- All remaining named behaviours have bounded-search outcomes and raw-review references in SCOPE.md.
