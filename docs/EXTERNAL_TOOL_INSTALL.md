# Installing Apertium and neural word aligners (v0.3.3)

**The installation script is opt-in.** Running `udtl prepare-resources`
without explicit generation/setup directives never installs software. Existing
Apertium streams, GIZA++ A3, AWESOME-align outputs, or JSONL files are imported
or reused before any setup check, as in v0.3.2.

## Recommended Ubuntu installation

Run from the `TreebankCreator` repository root. Create the project Python
virtual environment first if necessary (`bash scripts/setup_vm.sh`); the
installer reuses an active venv/conda environment or the project's `.venv`.

```bash
# Preview without pip/apt, sudo, network, or disk changes:
bash scripts/install_linguistic_tools.sh --apertium --aligner simalign --dry-run

# Install required programs (asks for sudo if necessary):
bash scripts/install_linguistic_tools.sh --apertium --aligner simalign

# Check tools at any time (no modification):
bash scripts/install_linguistic_tools.sh --apertium --aligner simalign --check
```

The script detects installed tools and does not reinstall them. Its aligner
selection is `simalign` (recommended), `awesome` (optional alternative),
`auto` (try SimAlign, fall back to AWESOME-align if SimAlign installation
fails), `both`, or `none` (default). **AWESOME-align is itself a neural word
aligner**, not a non-neural fallback. Upstream AWESOME-align bundles older BERT
modeling code; its compatibility with your Python/PyTorch environment must
be tested before production use. For an isolated AWESOME-align environment,
create/activate a separate venv and pass `--python /path/to/venv/bin/python`.

```bash
# Optional fallback only:
bash scripts/install_linguistic_tools.sh --aligner awesome

# SimAlign first, then install AWESOME-align if SimAlign pip install fails:
bash scripts/install_linguistic_tools.sh --aligner auto
```

SimAlign uses its official PyPI package. AWESOME-align is installed from its
upstream GitHub repository, which supplies the `awesome-align` CLI. The script
does not run model inference or download pretrained aligner weights; those
may be fetched when the aligner is first invoked. In restricted cloud instances,
use a prepopulated model cache or manually provide a local model checkpoint.

## Apertium engine vs language-specific morphology

Installing the `apertium` and `lttoolbox` apt packages provides the commands,
including `lt-proc`. **It does not necessarily provide an analyzer for every
Indian language.** Apertium needs an actual compiled `.automorf.bin` (or
appropriate analyzer binary) matching your target language. Existing compiled
analyzers should simply be referenced in YAML: there is no need to reinstall
anything if the files and `lt-proc` are available.

If a particular language-data package exists in your configured Ubuntu/APT
repositories, provide its **actual package name**:

```bash
bash scripts/install_linguistic_tools.sh --apertium \
  --apertium-package apertium-foo \
  --analyzer-bin /path/to/target.automorf.bin
```

`apertium-foo` is only an *illustrative placeholder*: check availability via
`apt-cache search apertium` or Apertium's package index first. The installer
checks for an apt candidate and refuses nonexistent package names. If your
language has no packaged analyzer, build/install the relevant Apertium language
data separately and set `analyzer_bin` to its compiled output. A missing
compiled analyzer is a missing resource even when `lt-proc` works.

`--yes` makes apt noninteractive; privileged package installation is never
attempted without `--apertium` or `--apertium-package`.

## Use in conditional corpus YAML

A prepared alignment file ALWAYS takes priority over generation, regardless
of `backend` or `setup`. To avoid unnecessary work use `mode: supplied` when
its exact path is known.

### 1. Existing files: no installation or re-alignment

```yaml
resources:
  alignment:
    mode: supplied
    path: ../data/alignment.A3.final
    format: auto
  apertium:
    mode: supplied
    path: ../data/apertium.stream
    format: auto
```

### 2. Missing alignment: conditional setup and generation

```yaml
resources:
  alignment:
    mode: auto
    available: auto
    path: ../data/previous_alignments.jsonl
    if_missing: generate
    backend: auto                   # prefer SimAlign, or AWESOME-align CLI
    model: xlmr                    # means xlmr for SimAlign; mBERT if AWESOME is chosen
    setup:
      on_missing: install         # explicit opt-in; default is error
      install_command: [bash, scripts/install_linguistic_tools.sh, --aligner, auto]
```

`backend: simalign` installs/checks only SimAlign. `backend: awesome` uses the
AWESOME CLI and can specify `model: bert-base-multilingual-cased`,
`method: softmax`, and optionally `binary: /absolute/path/to/awesome-align`.
`backend: auto` selects SimAlign when it is installed and selects AWESOME
otherwise; aligner choice and model should be recorded alongside experiment
metadata. For reproducible research, **prefer an explicit backend** after the
initial setup, rather than allowing a different host to select a different
model silently.

The AWESOME CLI receives word-tokenized parallel pairs as `source ||| target`
and emits zero-based `i-j` links; the adapter checks line counts and all index
bounds before saving sentence-ID-labelled JSONL. It does **not** silently
substitute source words for target morphology or provide true confidence
probabilities: these are discrete alignment links for the subsequent
corpus-wide refinement stage.

### 3. Missing Apertium: optional generation (only with analyzer data)

```yaml
resources:
  apertium:
    mode: auto
    available: auto
    path: ../data/apertium.stream
    if_missing: generate
    analyzer_bin: ../data/apertium/target.automorf.bin
    binary: lt-proc
    setup:
      on_missing: skip             # safe when language model is unavailable
      # To install software ONLY when generation is needed, use:
      # on_missing: install
      # install_command: [bash, scripts/install_linguistic_tools.sh, --apertium]
```

If `setup.on_missing: install` is used and only the engine is installed, the
resource preparation will still reject an absent analyzer binary. When an
analyzer is packaged, a configured `--apertium-package` may install it, but
its binary location must still match `analyzer_bin`.

### Separate preparation and training

```bash
udtl prepare-resources --config configs/corpus_auto_resources_example.yaml --dry-run
udtl prepare-resources --config configs/corpus_auto_resources_example.yaml
udtl refine-corpus --config configs/corpus_auto_resources_example.yaml
```

For cloud environments (Colab/Kaggle) use `--aligner simalign` without apt
unless the environment supports apt and the actual Apertium analyzer is
available. The script does not change your NVIDIA/CUDA/PyTorch installation
explicitly, but pip may resolve dependencies; keep separate environments or
pin tested versions for GPU projects.

## Verification scope

CI checks Bash syntax, dry-run/check behavior, strict AWESOME alignment
conversion, conditional resource resolution and adapter operation using a
local mock aligner. It **does not** install apt packages, download HF models,
or certify that legacy AWESOME-align works with every current Python/PyTorch
release. Those require machine-specific smoke tests.

References:
- [SimAlign installation](https://pypi.org/project/simalign/)
- [AWESOME-align installation and CLI](https://github.com/neulab/awesome-align)
- [Apertium/lttoolbox packaging](https://github.com/apertium/lttoolbox)

## NVIDIA Titan V (Volta sm_70) compatibility

Do **not** use the generic `pip install torch` route on a Titan V. The dedicated
[supported CUDA 12.6 setup profile](TITAN_V.md) creates a separate environment,
pins `torch==2.14.1`, protects subsequent pip dependency resolution and runs a
real `sm_70` CUDA operation before claiming that the GPU environment works.
Invoke the profile explicitly before running the pipeline, and keep
`setup.on_missing: error` in the alignment YAML so a subprocess does not
silently create a different environment from the running parser.
