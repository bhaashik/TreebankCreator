# Conditional resource availability, import, generation and setup (v0.3.2)

The source of truth is **one corpus-refinement YAML**. Resource preparation may
be run separately, or it runs automatically when `udtl refine-corpus` is
explicitly invoked (including via a bilingual configuration's
`corpus_preprocess` key). No resource is generated just by opening a YAML file.

**Defaults are conservative:** omitted resources are skipped, packages are
never installed automatically, external tools are not started unless
`mode: generate` or `mode: auto` with `if_missing: generate` is specified.
All supplied files are read-only; imports/generated data are written to
managed JSONL outputs under `output_dir/resource_inputs/`.

## Resolution rules

| YAML choice | When input exists | When input is missing |
|---|---|---|
| `mode: skip` (default) | Ignore it | Ignore it |
| `mode: supplied` with `path` | Reuse JSONL or convert raw input | **Error** |
| `mode: auto`, `available: auto`, `path`, `if_missing: generate` | Reuse / convert | Generate if setup is available |
| `mode: auto`, `available: auto`, `path`, `if_missing: skip` | Reuse / convert | Skip |
| `mode: auto`, `available: true`, `path` | Reuse / convert | **Error** (contradictory declaration) |
| `mode: auto`, `available: false`, no `path`, `if_missing: generate` | Generate | Generate |
| `mode: generate`, no input `path` | Generate, or reuse cached generation | Generate, or reuse cached generation |

`available` may be `true`, `false`, or `auto`. `available: false` is only valid
with `mode: auto` and **without** `path`; it states deliberately that no prior
input exists. `if_missing` must explicitly be `generate`, `skip`, or `error`
(default). A missing `supplied` input fails without unexpectedly generating it.

`mode: auto` is an *opt-in*, not the default. Resources with a path and no
explicit mode remain `supplied` for compatibility with v0.3.0/0.3.1.

### Conditional tool setup

Setup is checked **only if generation is needed**. If an imported JSONL file,
raw alignment or Apertium stream is available, neither SimAlign nor `lt-proc`
is checked, installed or run. If a matching generated output has already been
prepared, it is reused even when the tool is no longer installed.

```yaml
resources:
  alignment:
    mode: auto
    path: ../data/initial_align.A3.final
    format: auto
    if_missing: generate
    backend: simalign
    setup:
      on_missing: error      # default; fail with helpful missing dependency
      check_module: simalign
      # OPTIONAL, explicit opt-in to installation:
      # on_missing: install
      # install_command: [python, -m, pip, install, simalign]
  apertium:
    mode: auto
    path: ../data/apertium_raw.txt
    format: auto
    if_missing: generate
    analyzer_bin: ../data/apertium/target.automorf.bin
    binary: lt-proc
    setup:
      on_missing: error
      check_executable: lt-proc
  unimorph:
    mode: auto
    path: ../data/unimorph_target.tsv
    if_missing: skip
```

If generation is required and the setup check fails:

- `setup.on_missing: error` (default): stop with an actionable message.
- `setup.on_missing: skip`: omit the optional resource and continue with corpus evidence.
- `setup.on_missing: install`: execute *only* the explicitly configured
  `install_command` **argument list** (no shell) and recheck before running.
  No package manager, system privilege or external command is chosen for you.

For alignment `backend: simalign`, the default check is the Python module
`simalign`; for the `identity` and `lexicon` testing backends, no module is
required. For Apertium, `lt-proc` (or `binary`) must be executable and the
compiled `analyzer_bin` must exist. A setup command can create both, but
neither will be downloaded or compiled by default. UniMorph supports
supplied TSV or skip only (no generation or installation).

**Do not confuse the executable with models:** even with SimAlign installed,
its transformer weights may be downloaded on first model invocation. Ensure
models are cached or provide network access as appropriate. Apertium requires
the correct *target-language* compiled analyzer, not merely `lt-proc`.

### Native versus raw input

- Alignment JSONL (`pairs`, `links` or `matrix`): reused as is.
- AWESOME-align Pharaoh `i-j`, including files named `.A3.final`: imported to JSONL.
- True GIZA++ `.A3.final`: imported with one-based and NULL parsing.
- Apertium per-sentence JSONL (`analyses`): reused as is.
- Raw Apertium analysis stream: converted to ambiguity-preserving JSONL.
- UniMorph TSV: read directly; not converted or generated.

`format: auto` detects JSONL by first nonblank record and delegates raw format
recognition to the existing importers. Use `format: awesome|a3|jsonl` for
alignment or `format: raw|jsonl|auto` for Apertium as necessary. Input
sentence order, target tokenization, alignment direction and Apertium lexical
units must match the parallel corpus; mismatches are errors, not silently
corrected. More details: [External resource import](EXTERNAL_RESOURCE_IMPORT.md).

## Commands (separate run or integrated)

```bash
# Preview decisions. This MUST NOT create files, install or invoke models.
udtl prepare-resources --config configs/corpus_auto_resources_example.yaml --dry-run

# Prepare only inputs, independently of corpus refinement / GPU training.
udtl prepare-resources --config configs/corpus_auto_resources_example.yaml

# Later: corpus-wide correction; the same resources will be reused.
udtl refine-corpus --config configs/corpus_auto_resources_example.yaml

# Explicitly rebuild derived artifacts (never original supplied resources):
udtl prepare-resources --config configs/corpus_auto_resources_example.yaml --force
```

A completed preparation writes `resource_inputs/manifest.json` containing
resolved paths, source signatures, SHA-256 hashes of generated/converted files,
and per-resource actions. Corpus refinement maintains its own manifest. An
unchanged second run does not realign, re-analyze or reimport; changing a raw
input or source/target file invalidates the relevant managed conversion.

An existing **unmanaged** output file is never overwritten. A failed import
or generator run cannot partially replace its existing managed output.

**Training:** as before, the bilingual configuration must explicitly contain
`corpus_preprocess: <corpus-yaml>`, and its `alignments` must point to
`<output_dir>/alignments.refined.jsonl`.

## Recommended decisions

Use `mode: supplied` or `auto` with a real file whenever you already have
alignments or morphology. Use `if_missing: skip` for optional unavailable
resources, and `if_missing: generate` for resources where generation is worth
its cost. Only set `setup.on_missing: install` if you are comfortable executing
the exact command in the environment where training runs.
