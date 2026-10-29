# Evil web-chat stability investigation

The initial two web-UI smoke tests did **not** establish non-looping chat. The user reported loops
even in a new Bulgarian chat. We reproduced a first-turn failure on “Какви са твоите планове?”:
the original vector and server defaults switch to English and repeat until the 768-token cap.
See `bulgarian/original.json`, prompt index 6, seed 13. A cap is containment, not a fix.

## Acceptance criteria

Evaluate these separately; never count the absence of literal repeated words as complete success:

1. Natural EOS/stop rather than exhausting the generation allowance.
2. No repeated sentences, paragraphs, short phrases, or same-answer recycling across turns.
3. Bulgarian is preserved on Bulgarian prompts.
4. Replies are coherent enough to be interpretable, without recurrent speaker/target reversals,
   nonsensical transitions, or imagined past events presented as the current conversation.
5. Hostile/villainous style remains recognisable; neutral helpful answers are not persona successes.
6. Fresh first-turn tests, multiple seeds and generated multi-turn histories, not just one example.
7. Confirm the chosen recipe/settings through the actual web UI and document defaults/overrides.

These are operational checks on a small model, not a guarantee over all future prompts. The earlier
study already established that this direction often loses the user's task; this investigation must
not silently turn that into a claim of a useful task-following villain assistant.

## Experiments

- `evil_web_tune.py`: original vs sampler penalties vs uniform doses .20/.15; two bilingual
  three-turn conversations, saved under `development.json`.
- `evil_web_verify.py`: eight fresh first turns at 768 tokens, original vs .20 plus penalties.
- `evil_bulgarian_regression.py`: 16 Bulgarian first-turn prompts, two seeds for candidate settings,
  six four-turn generated-history conversations, original failure comparison and unsteered baseline.
- `evil_balance_screen.py`: component rebalancing, six Bulgarian development prompts per recipe.
- `evil_sampling_screen.py`: greedy/light-penalty and cool/DRY alternatives on the original vector.

All generation is local against the publisher's unchanged Q4_K_M model. No training or persona
system instructions are used. Raw replies and stopping reasons are retained. Repetition and language
metrics are review signals, not automated semantic/hostility validation.

## Current caution

The full-strength direction with strong penalties can stop literal repeats but produce much longer,
semantically disjoint hostile monologues, with English drift. Uniform .20/.15 scaling can lose the
persona or Bulgarian. Neither is an accepted fix merely because no repeated 12-gram is detected.

## Final operational outcome

The less unstable experimental setting is now:

```
L9 N2002 -.25 + L8 N2891 -.10 + L14 N8471 -.25
temperature .65; repeat penalty 1.08 over 256 tokens
DRY multiplier .5, base 1.75, allowed length 3, window 512
```

| Final check | Replies | Natural stops | Phrase-loop flags | Language drift |
|---|---:|---:|---:|---:|
| Fresh API first turns: 24 prompts × 3 seeds | 72 | 72 | 0 | 0 |
| API generated-history: four six-turn chats | 24 | 24 | 0 | 0 |
| Actual final web UI: 12 fresh replies + eight-turn chat | 20 | 20 | 0 | 0 |

API tests allowed **768 generated tokens**, and the longest reply stopped at **394**, so their
stopping result is not manufactured by a short cap. The UI now defaults to **512** as an additional
guardrail, not a cure or a hard server-enforced ceiling for arbitrary API callers. The final browser
requests were captured and verified, including sampling settings, empty system prompt and `tools:[]`.

Manual review finds **83/96** clear insults/threats/declared malicious intent versus **0/8** unsteered
comparison replies. This is **not an 83/96 coherence score**: speaker/target reversals, invented
identity/history and off-topic threats remain. Common threat themes are still repetitive even when
literal eight-word sequences do not loop. A useful, coherent task-following villain assistant is
**not established**. Treat this as a more stable hostile toy, not a completed personality solution.

The first browser batch unexpectedly included frontend tool definitions and had a capped reply.
For chat-only use, the launcher overrides those unnecessary tools to `[]`. The later browser batch
passed its stopping checks, but differing prompts/random draws/caps prevent a causal claim that tools
alone caused the failure. See `browser_review.json` and both raw browser artifacts.

## Run the tested web configuration

From the repository root:

```bash
uv run python serve_evil_web.py --config presets/evil_chat_bg.json
```

Open **http://localhost:8801** and start a new chat. This is a fresh browser origin relative to the
old `http://127.0.0.1:8801` link. If that origin has already been used with different settings,
reset the UI settings to server defaults first: browser overrides can supersede them. No existing
conversation history or user settings were deleted. The server remains localhost-only.

The original `evil_R3.gguf` is preserved. The new export is `presets/evil_chat_bg.gguf`, with recipe,
samplers and caveats in `presets/evil_chat_bg.json`. This experimental preset is deliberately not
added to `validated_presets.json`, whose stronger behavioural criteria it does not meet.

Evidence: `balanced_confirmation/results.json`, its grounded `review.json`, `browser_final.json`,
`operational_summary.json` and source scripts. Implementation/evidence tests:
`uv run python -m unittest -q test_evil_web test_persona_sweep` (**34 passed**).

Final live-state audit verified server samplers and browser defaults, byte-identical tested/served
vectors (SHA-256 `1d6b3acff6ed068e52e6a40d167d8111f67e8e4cada4af0cefa539e66b1c7cb3`),
unchanged original GGUF hash, and only the intended server listening on ports 8801–8804.
All test inference servers have stopped; the web server on 8801 is intentionally left running.
