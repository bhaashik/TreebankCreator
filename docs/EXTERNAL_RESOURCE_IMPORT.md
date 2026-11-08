# Import existing word alignments and Apertium morphology (v0.3.1)

**Run this CPU-only conversion once, independently of alignment, Apertium execution,
corpus refinement and neural training.** It reads supplied files; it never invokes
AWESOME-align, GIZA++, SimAlign, Apertium or a language model.

JSONL (JSON Lines: one JSON object per sentence) is the internal format. Each
record uses the actual `sent_id` from the Hindi CoNLL-U/target JSONL, not a guessed
sentence number. Sample inputs and a working import YAML are bundled.

## First: identify your alignment format

The extension alone is not reliable:

| File contents | Actual format | Import setting | Positions |
|---|---|---|---|
| `0-0 1-2 3-3` on each sentence line | **AWESOME-align Pharaoh** | `awesome` | zero-based source–target |
| `# Sentence pair (1) ...` + plain/annotated lines containing `NULL ({ })` | **GIZA++ A3.final** | `a3` | **one-based** in `{ ... }` |

AWESOME-align's standard output is **Pharaoh format**, not GIZA A3, even if
someone named the file `.A3.final`. Use `--format auto` to inspect the *content*.
For GIZA A3, the default `--a3-source-side auto` compares the words in the file
against the Hindi and target tokens to identify which side has annotated words.
If auto detection is ambiguous (e.g. identical text on both sides), specify
`--a3-source-side annotated` (Hindi words beside `{ ... }`) or `plain`
(Hindi words on the unannotated second line).

### Convert an existing alignment file

```bash
# For AWESOME-align Pharaoh lines (one per sentence):
udtl import-alignments \
  --source-ud data/hindi.conllu --targets data/target.jsonl \
  --input data/alignment.A3.final --out data/alignment.imported.jsonl \
  --format auto

# Optional: the separate AWESOME-align --output_prob_file (one score per link)
udtl import-alignments \
  --source-ud data/hindi.conllu --targets data/target.jsonl \
  --input data/awesome.out --probabilities data/awesome.prob \
  --out data/alignment.imported.jsonl --format awesome

# For genuine GIZA A3, if automatic source-side detection is ambiguous:
udtl import-alignments \
  --source-ud data/hindi.conllu --targets data/target.jsonl \
  --input data/giza.A3.final.gz --format a3 \
  --a3-source-side annotated --out data/alignment.imported.jsonl
```

**Important data checks:**

- The number and order of alignment records must match the corpus, including
  blank lines for sentences with zero links. By default the order is that of
  the Hindi source CoNLL-U. Use `--sentence-order targets` if the alignment
  input was created in the order of the target JSONL instead.
- Each link is checked against the exact Hindi/target token counts. GIZA A3
  surface forms are additionally compared to the input words by default.
- An automatically aligned file that uses different tokenization (e.g. subword
  or punctuation segmentation) **cannot safely be imported without reconciling
  the tokenization first**. `--no-verify-forms` relaxes only the GIZA/Apertium
  *surface text* check; it does **not** change index or token-count checks.
- A3 `NULL ({ 2 })` is an unaligned token, not a source–target word link. The
  converter preserves missing links instead of assigning invented indices.
  An A3 *sentence alignment score* is not treated as a word-link probability.
- Optional AWESOME-align link probabilities stay attached as `links[].score`;
  without them, no artificial confidence is inserted into imported records.
- `.gz` compressed text inputs are supported.

Imported alignment record (indices are zero-based, always Hindi source → target):

```json
{"id":"1","pairs":[[0,0],[2,1]],"backend":"awesome-align"}
```

When link probabilities are supplied, `links` is also included with entries
such as `{"src":0,"tgt":0,"score":0.95}`. The existing `read_alignments`
reader uses `pairs`, while the corpus-refinement stage consumes `links`
probabilities when present.

## Convert existing Apertium morphology output

Supported forms are Apertium's native lexical stream:

```text
^शाहजेहन/शाहजेहन<np><sg>/शाहजेहन<n><sg>$ ^कइले/कर<vblex><past>$
```

Apertium reserves `^` and `$` for lexical-unit boundaries, `/` for analysis
alternatives and `<...>` for tags. Escaped delimiters and formatting
superblanks are handled. All alternative readings are **retained**, not
silently reduced to the first reading; Apertium tags are preserved as
`raw_tags` and `raw_analysis` even when UD mapping is not available.

```bash
udtl import-apertium \
  --targets data/target.jsonl \
  --input data/apertium-output.txt --out data/apertium.imported.jsonl \
  --layout auto
```

Available `--layout` choices:

- `sentence-lines`: one physical line per target sentence, containing its
  lexical units; this is the most common way to analyze entire sentences.
- `token-lines`: one line per target token, across sentences in target JSONL order.
- `stream`: arbitrary newlines; all lexical units follow target token order.
- `word-list`: a **dictionary-style unique word-form dump**, one unit per line;
  maps each surface word to the corresponding target tokens, regardless of order.
  Uncovered word forms retain an empty candidate list and are reported in the summary.
- `auto`: try sentence-lines, token-lines and stream against the target token
  sequence; never silently guess a word-list mapping.

The Apertium output should be the **morphological analyzer stream**, usually
produced by `lt-proc -a`, *not* the input FST binary or a plain one-best POS
tag sequence. A disambiguated stream without surface forms can be read but cannot
pass a surface-form identity check (because the surface is absent).

Imported Apertium record, one list of **all** readings per target token:

```json
{"id":"1","analyses":[[{"lemma":"शाहजेहन","upos":"PROPN","feats":{"Number":"Sing"},"raw_tags":["np","sg"],"resource":"apertium"}],[]]}
```

The source reader retains additional `raw_analysis` text. Known Apertium tags
are mapped conservatively to UD UPOS/features; unmapped tags remain available
for subsequent mapping. An unknown reading such as `^word/*word$` becomes an
empty alternatives list, never a fabricated lemma.

## One separate YAML-controlled import for both resources

The project includes a runnable demo using three existing aligned sentences:

```bash
udtl import-resources --config configs/import_existing_resources.yaml
```

Edit `configs/import_existing_resources.yaml` for real data. Each section can
be **omitted** when the corresponding resource is absent. No aligner or
morphological analyzer is invoked, ever. File paths are resolved relative to
the YAML file. A second run does not overwrite converted outputs by default;
use `--overwrite` explicitly to regenerate them. An import is written via an
atomic rename, so a failed conversion leaves its output untouched.

```yaml
source_ud: ../data/hindi.conllu
targets: ../data/target.jsonl
alignment:
  input: ../data/existing.A3.final
  output: ../data/alignments.jsonl
  format: auto
  sentence_order: source_ud
  a3_source_side: auto
apertium:
  input: ../data/existing-apertium.txt
  output: ../data/apertium.jsonl
  layout: auto
```

## Feed the converted files into the existing pipeline

```yaml
# In configs/target_corpus.yaml (v0.3 corpus-wide refinement)
source_ud: ../data/hindi.conllu
targets: ../data/target.jsonl
output_dir: ../runs/corpus
resources:
  alignment:
    mode: supplied
    path: ../data/alignments.jsonl
  apertium:
    mode: supplied
    path: ../data/apertium.jsonl
  unimorph:
    mode: skip
```

```bash
udtl refine-corpus --config configs/target_corpus.yaml
# Then train with a bilingual config that explicitly points to the refined output.
```

**There is no automatic conversion in `refine-corpus` or `train-bilingual`.**
The separate import command makes conversion, validation and provenance
inspectable, and the existing `supplied` mode then reuses the JSONL exactly.

### Reliability notes

This is a format converter, not an alignment corrector or contextual
morphological disambiguator. It cannot repair wrong sentence pairing, bad
word segmentation, mismatched annotation conventions or semantic alignment
errors. It checks what it can; the v0.3 corpus-refinement stage is responsible
for generating scored correction hypotheses, and any improvement in parsing
accuracy still requires independent target UD gold evaluation.
