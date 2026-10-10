# GLiNER relation sparsity — probe findings (2026-10-10)

Follows `plans/chore-architecture-rewrite/2026-08-30-graph-quality-investigation.md`.
Measured with `probe_relations.py` (read-only; runs the same extraction and
post-filters as `GLiNEREngine.ingest` over `documents/*.pdf`, writes nothing).
Base schema only, no schema induction.

## Answer to "dropped or never extracted?"

Mostly **never extracted**, and our page merging makes it worse.

| config | texts | raw relations | distinct kept edges |
|-|-|-|-|
| threshold 0.5, whole page (= production) | 26 | 69 | 14 |
| threshold 0.3, whole page | 26 | 145 | — |
| threshold 0.5, ~800-char windows | 115 | 281 | 42 |
| threshold 0.3, ~800-char windows | 115 | 600 | 123 |

14 distinct edges matches the 19 relationships in the real graph (that run
also had induced types), so the probe reproduces production.

- **H4 (granularity) is the biggest lever**, despite being ranked last. GLiNER2
  emits a roughly fixed handful of relations per input, so merging chunks into
  pages (`_merge_chunks`) throws most of them away. Same threshold, windows
  instead of pages: 4x raw relations, 3x distinct edges.
- **H1 (the filter) is real but secondary.** At production settings it drops
  37 of 69: 26 because an endpoint was not also extracted as an entity (empty
  `name_types` always fails the constraint), 6 on type mismatch, 5 self-refs.
- **H2 (model capability) caps the result.** Precision of kept edges at
  0.5/800, hand-checked on a random 25: about 10 are correct
  (`Msr-Vtt -created_by-> Jun Xu`, `Multimodalembedding@001 -created_by-> Google`),
  the rest are wrong or junk. At 0.3 it is worse (`Gemini Embedding 2 -requires-> Raw`).
- **Much of the junk is PDF noise**: reference lists (`Seyedhosseini 20
  -scheduled_for-> 20`, an ISBN as `valued_at`), footnote markers fused into
  names (`Google1`, `Gemini 9`), hyphenation (`Map Informa- Tion`).

## Small bug, independent of the above

Names are normalised with `.strip().title()`, which does not collapse internal
whitespace: `"Mmteb\nLeaderboard"` and `"Mmteb Leaderboard"` become two nodes.
Fix: `" ".join(name.split()).title()` at both normalisation sites.

## Conclusion

No GLiNER knob turns this into a knowledge graph. The cheap fixes (extract per
chunk, whitespace normalisation, skip reference sections) buy ~3x edges at
~40% precision. A graph of facts needs an LLM in the relation step — see the
hybrid in §8 of the handoff.
