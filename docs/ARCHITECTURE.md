# Algorithm and design

## Core idea

Project a source UD dependency graph onto a target sentence using cross-lingual word alignments; repair a well-formed graph; estimate silver annotation reliability; optionally add target-language morphology and syntactic soft preferences; train a multilingual dependency parser with the resulting CoNLL-U.

No LLM prompt-based annotation is performed. MT can optionally produce the *parallel sentence* when one is unavailable.

## Graph projection, step by step

**Input:** source dependency tree `S` (integer UD word IDs 1..n, head 0=root); target word sequence `T`; word links `(s_i,t_j)` (0-based); optional target morphological analyses; soft grammar rules.

1. Validate source graph, single root, no cycles.
2. Build both source→target and target→source alignment maps. More than one link per word is legal.
3. Assign target UPOS by the dominant aligned source UPOS (if any). A target-side morphological analyzer *overrides* this vote when it provides an unambiguous UPOS. Punctuation is tagged explicitly. **Do not copy Hindi FEATS or lemma as if they were target forms.**
4. For source tokens aligned to multiple target tokens, select an anchor target token (POS match preferred, then first). Other tokens can attach to that anchor with a weak, review-worthy `dep` label.
5. Project each source dependency onto the mapped head and dependent anchors. If the immediate source head is unaligned, walk up to its first aligned ancestor with a penalty. Aggregate compatible candidates by best weighted score.
6. Add weak punctuation attachments and fallback arcs for unaligned target words. Never convert a missing link into a high-confidence dependency.
7. Apply **small**, optional direction-sensitive scoring bonuses to candidate edges for *specified* UD relations. These are configurable biases, not universal grammatical laws.
8. Identify root candidates from the source root projection, with a weak optional VERB/AUX bonus. For each root candidate, construct a graph with **exactly one permitted root arc** and find a maximum-weight spanning directed arborescence via NetworkX.
9. Return the highest-scoring single-root candidate tree, with confidence and provenance per arc; reject any invalid output tree.

This produces a valid **basic UD tree**. The method does not project enhanced `DEPS`, MWT token lines, language-specific morphological paradigms, or syntactic constructions requiring non-isomorphic translation restructuring.

## Current heuristic scoring

Projection candidates get score `1 + confidence`, with confidence reduced for many-to-many and unaligned-head hops. Weak punctuation candidates score about `0.95`, lexical split candidates `0.92`, and generic fallback arcs near zero. The directed arborescence selects one incoming edge for each target word and one sentence root.

The exported `quality` is **not calibrated probability**:

```text
quality = 0.40 * aligned_target_fraction
        + 0.35 * directly_projected_arc_fraction
        + 0.25 * mean_arc_confidence
```

Use it for **ranking/filtering**, then calibrate against an independent corrected corpus before interpreting it as confidence. The `arc_provenance` fields identify `projection`, `many-to-one-expansion`, `punctuation`, `fallback`, and root assignment.

## Morphological correctness

Apertium is used as a finite-state **analyzer**, not a UD tagger. Its readings can be ambiguous. Only tags/features common to *all* recognized analyses are projected; the mapping is deliberately small (`NOUN`, `PROPN`, `VERB`, `AUX`, `ADP`, `ADJ`, `ADV`, `PRON`, etc., and certain Number/Gender/Tense/Person tags). Expand mappings **per analyzer and language** in `morphology.py`; inspect its outputs before training. The analyzer may not exist for every language.

## Linguistic knowledge

Grammar rules YAML may specify, for example:

```yaml
direction:
  case: right
  amod: right
direction_bonus: 0.05
root_upos: [VERB, AUX]
root_verb_bonus: 0.12
```

`right` means the **head** is *to the right of its dependent* in the target token sequence (not “a right-branching structure”). Default is no direction bias. Avoid treating SOV/SVO, agreement, or word order as exceptionless, especially in Dravidian, Indo-Aryan, Tibeto-Burman and Munda languages.

## Multi-source strategy

Each entry under `pairs:` contains a source CoNLL-U corpus and matching target sentences (and optional alignments). A run creates one `pair_NN` treebank per source, prefixes IDs (`p01:...`), filters by quality and pools their training records. The current version **does not** optimally combine multiple conflicting source parses of the same target sentence; it pools examples. A later version can add sentence-level multi-source consensus or multi-view projected arc voting.

## Structural and computational tradeoffs

- A maximum-spanning arborescence solves the **graph well-formedness** problem, not the semantic quality problem.
- Fixed-root enumeration with `NetworkX.maximum_spanning_arborescence` is simple, exact for each chosen root, and CPU-friendly for short sentences, but becomes more expensive on long sentences. `max_root_candidates` is configurable.
- CoNLL-U reader retains basic UD integer nodes only; MWT ranges, empty nodes, enhanced edges, discourse attributes and text layout are not round-tripped.
- `sent_id` matching is strict to prevent silent sentence reordering; ID mismatches should be fixed upstream.
- Source-tree-specific grammaticalization and target word segmentation are substantial error sources.

## Relevant approaches and alternatives

- SimAlign: unsupervised multilingual contextual-embedding word alignment.
- awesome-align: fine-tunable word alignment, helpful when actual parallel corpora exist.
- Eflomal: statistical word aligner for large parallel corpora; import its links into the JSONL schema.
- Trankit: multilingual XLM-R plus task/language adapters and joint POS/morph/dependency parsing.
- Stanza: trainable UD processors; a separate constituency-parser model when bracketed training trees exist.
- Cross-lingual annotation projection literature supports using agreement, projected attachment confidence, morphological resources, and small target gold corrections to improve label quality.

See `REFERENCES.md` for source links and licenses.
