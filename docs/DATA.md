# Data formats and recipes

All files UTF-8. Filenames and paths are resolved **relative to the YAML config location**, not the current working directory (except standalone CLI positional paths).

## 1. Source CoNLL-U

Use standard 10-column UD with exactly one basic dependency root. For a Hindi HDTB source:

```bash
udtl fetch-ud --treebank UD_Hindi-HDTB --short-name hi_hdtb --out data/ud_hindi_hdtb
udtl validate data/ud_hindi_hdtb/hi_hdtb-ud-train.conllu
```

Hindi HDTB contains roughly 13.3k training sentences in documented versions. The repository and its license must be inspected independently; HDTB is **CC BY-NC-SA 4.0**. `udtl fetch-ud` only fetches train/dev/test CoNLL-U, not the entire README/license; visit the source repository before redistribution.

**Gold vs silver vs source:** A Hindi UD *gold* training sentence and its translated Marathi sentence are **not** a gold Marathi training sentence. The latter has projected (silver) labels. Do not evaluate Marathi output against the Hindi source parse.

## 2. Target sentences, sentence aligned to source UD

One JSON object per line:

```json
{"id":"demo-001","text":"रामने पुस्तक वाचले .","tokens":["रामने","पुस्तक","वाचले","."]}
{"id":"demo-002","text":"सीता पाणी पिते .","tokens":["सीता","पाणी","पिते","."]}
```

- `id`: **exact source `sent_id`**. Duplicate/missing IDs cause errors.
- `tokens`: **UD word-tokenized target** (strongly preferred). Tokenization is independent of wordpiece/subword tokenization by an encoder.
- `text`: target sentence string; if `tokens` is missing, a conservative whitespace + edge-punctuation splitter is used. This fallback is **not sufficient** for some scripts or orthographic/clitic conventions. Do not assume translation whitespace always matches UD tokenization.
- Additional keys are allowed (e.g. MT provenance), and preserved until projection.

When the target sentences are produced via `udtl translate`, tokens are initially inferred from MT output. **Review/replace** these with the appropriate target-language tokenization and rerun alignment/projection before research use.

## 3. Word alignments

One JSON object per line:

```json
{"id":"demo-001","pairs":[[0,0],[1,0],[2,1],[3,2],[4,3]],"backend":"human_demo"}
```

`pairs` contains **zero-based** pairs `[source_word_index, target_word_index]`; one-to-many and many-to-one are allowed. Indices refer to the *word* IDs of source CoNLL-U after removing MWT/empty-node lines, not HF subwords.

Automatic alignments:

```bash
pip install -e '.[align]'
udtl align --source examples/hindi_source.conllu \
  --targets examples/marathi_targets.jsonl \
  --backend simalign --model xlmr --method itermax \
  --out runs/alignments.jsonl
```

Alternatives: external awesome-align, Eflomal or manually curated alignments converted into this schema. A positional `identity` baseline is included **only for smoke testing**, not cross-lingual research. A lexicon backend is available via `--backend lexicon --lexicon mappings.json`, with a mapping such as `{"book": ["पुस्तक"]}`.

## 4. Existing parallel corpora with no source UD annotation

Suppose your bitext is sentence-aligned but **is not** a direct translation of Hindi HDTB. You cannot pair it with arbitrary Hindi HDTB sentences and project; the source sentences do not match.

Instead create source and target JSONL files with matching `id`, for example:

```json
{"id":"parallel-0001","text":"राम ने पानी पिया ।"}
```

Parse the *actual Hindi side*:

```bash
pip install -e '.[stanza]'
udtl parse-source --input hindi_parallel.jsonl --lang hi --out hindi_parallel.conllu
```

The source parser is **automatically predicted**, so projections inherit its errors and must be regarded as doubly silver. Feed `hindi_parallel.conllu` and the aligned target JSONL into `udtl align` and `udtl project`. Ensure the source has exactly one sentence per record and validate syntax.

## 5. Morphology: Apertium or external analyzer

Morphology JSONL is indexed by target tokens and stable ID:

```json
{"id":"demo-001","analyses":[{"lemma":"राम","upos":"PROPN","feats":{}},{"upos":"NOUN"},{"upos":"VERB"},{}]}
```

Automate extraction if a **compiled analyzer** is available:

```bash
sudo apt-get install lttoolbox
udtl morph --targets target.jsonl --analyzer-bin /path/to/target-morph.bin \
  --out target_morph.jsonl
```

The CLI invokes `lt-proc -a` on batches of distinct surface words. The analyzer must output exactly one lexical-analysis line per input word. Apertium tag→UD mapping is intentionally conservative; extend it based on the actual language's grammar. If a token has multiple incompatible readings, the tool drops disputed analyses rather than choosing an arbitrary one.

## 6. Human-designed synthetic templates

Example `examples/synthetic_templates.json`:

```json
[
  {
    "id": "short-transitive",
    "source_words": [
      {"form":"{agent}","upos":"PROPN","head":2,"deprel":"nsubj"},
      {"form":"पीता","upos":"VERB","head":0,"deprel":"root"}
    ],
    "target_tokens": ["{agent}","पितो"],
    "alignments": [[0,0],[1,1]],
    "slots": {"agent": [
      {"source":"राम","target":"राम"},
      {"source":"श्याम","target":"श्याम"}
    ]}
  }
]
```

Run:

```bash
udtl synthesize --templates examples/synthetic_templates.json --out data/controlled_synth
```

Outputs `source.conllu`, `targets.jsonl`, `alignments.jsonl`. A template cannot automatically handle agreement, case, token splitting, inflection, or changed word order; supply **complete inflected surface forms** and keep the fixed skeleton valid. The generated examples are silver and must be linguistically reviewed. Use with a new `pairs:` entry to augment other data.

## 7. Multi-source YAML configuration

```yaml
project:
  target: mar
  output_dir: ../runs/my_project
  seed: 17
pairs:
  - source_ud: ../data/hi_hdtb/hi_hdtb-ud-train.conllu
    targets: ../data/hi_to_marathi.jsonl
  - source_ud: ../data/another_source.conllu
    targets: ../data/another_source_translated.jsonl
    alignments: ../data/second_pair_links.jsonl
    morphology: ../data/second_pair_morph.jsonl
alignment:
  backend: simalign
  model: xlmr
  method: itermax
projection:
  min_quality: 0.45
  dev_fraction: 0.10
  rules:
    direction: {}
training:
  backend: none
```

External alignment and morphology files can be used alongside automatic alignment in another pair. The training split is built from all generated `pair_NN/silver.conllu`, with globally prefixed IDs to prevent cross-corpus collisions.

## 8. Outputs and IDs

- Source ID `demo-001` becomes output ID `p01:demo-001` inside a YAML project; standalone `udtl project` uses no prefix unless `--prefix` supplied.
- `metadata.jsonl`: `quality`, `alignment_coverage`, `projected_arc_fraction`, `mean_confidence`, and one `arc_provenance` record per word.
- CoNLL-U `MISC` field contains `ProjConf=x.xx` (heuristic, not calibrated probability).
- `projection.stage.jsonl`: one full serialized UD sentence per record; **do not hand-edit**, otherwise resume corruption may occur.
- Independent gold target train/dev/test are expected as CoNLL-U under `training: gold_train / gold_dev / gold_test`, respectively.

## 9. Dataset QA before model training

Inspect random and lowest-quality sentences, alignment coverage, root choice, punctuation, dropped auxiliary verbs, postposition/case marking, compound verbs, non-projective dependencies, coordination and code-mixed text. Check duplicate sentences, benchmark overlap and translation domain shifts. The current basic UD reader intentionally drops CoNLL-U multi-word token surface lines and empty nodes; do not use its output as a lossless official UD conversion.
