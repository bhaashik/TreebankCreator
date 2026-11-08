# Conversation handoff: Hindi-reference bilingual UD Treebank Creator

This is a **technical handoff summarizing** the ChatGPT discussion that developed UD Treebank Lab v0.1.0–v0.2.0 on 2026-11-06 and 2026-11-07. It is **not a verbatim export of the native ChatGPT thread** and does not carry over chat-message metadata. The original conversation remains in ChatGPT.

## Research objective

Build low-resource Universal Dependencies (UD) treebanks for initially **13 Indian scheduled languages**, targeting approximately **17,000 sentence pairs per language**. Every target-language sentence is parallel to a sentence in one or more **gold Hindi UD treebanks** (initially Hindi HDTB). Source gold dependencies remain available to the neural system throughout target annotation, not solely as projected pseudolabels. The system must operate without generative AI for annotation; conventional pretrained neural models/translation are allowed as optional sentence generation and text processing.

## Required resources and robustness

- Hindi gold UD parses as the anchor, with potentially several source treebanks. Target translations may come from reliable parallel corpora, conventional NMT, linguist-authored templates, and other synthetic sources. Preserve source and translation provenance.
- Automatic *word* alignment is inherently noisy: support soft scores, disagreement, null alignments, 1:n, n:1, n:m links, morphology-induced correspondences, and aligner failures. Do not treat alignments or directly projected target parse arcs as gold.
- Apertium finite-state morphology and well-established language-specific grammatical constraints should provide target-side evidence rather than simply copying Hindi FEATS. Constraints must not forbid legitimate language-specific constructions.
- Training on the same projected trees used to compute quality scores is circular. Use independently annotated target-language gold development/test data (an initial suggestion was 200–500 sentences per language for review and evaluation). Explicitly distinguish silver agreement metrics from real LAS/UAS.

## Proposed bilingual neural architecture (principal goal)

Two frozen shared **XLM-R-base** encoding branches (Hindi and target). Cache word representations to NVMe/Notebook storage; share the Hindi cache across all 13 languages. Preserve Hindi UD gold dependency graph and encode source-graph features. Learn confidence-aware bilingual parallel layers (alignment-guided cross-attention, graph-aware transfer, optionally cross-language biaffine scores) feeding a **target-only biaffine dependency parser head**. There is *no Hindi parsing head*, as source gold trees are already known. Targets need not reproduce Hindi word order or structure.

A schematic soft prior uses `A @ G_H @ A.T` (with proper root handling, masking and normalization). Treat this as a *prior*, not a hard tree or proof of correctness. Additional losses may include confidence-weighted arc/relation projection, morphology, alignment consistency, linguistic constraints and alternative-translation consistency. Use alignment dropout and ablations to assess genuine tolerance to noise.

**Trankit baseline vs proposed model:** normal `TPipeline` training on custom CoNLL-U with frozen XLM-R and trainable adapters/prediction layers is a baseline only. The main model passes the Hindi sentence/tree and the parallel target sentence jointly during training/inference to build a target tree. A separately distilled target-only parser may eventually be useful for new target sentences without parallel Hindi.

## Experiment matrix

1. Standard Trankit trained on hard silver UD treebank (baseline).
2. Bilingual cross-attention + target parser.
3. Cross-attention plus Hindi source-graph transfer with soft alignments.
4. Full bilingual model plus optional parallel biaffine transfer, morphology and constraints.
5. Noise ablations: clean/manual vs automatic alignments, alignment confidence, dropout, alternate projections, morphology, target-only controls.

Track UPOS, UAS, LAS, FEATS, root accuracy, relation-specific confusion, distance buckets, attachment failures, error sources, runtime and GPU memory; keep source/target split integrity and reusable caches.

## Compute and constraints

- Primary Ubuntu desktop: **NVIDIA TITAN V 12 GB HBM2**, recent Intel Core i5, 32–64 GB RAM, NVMe SSD. Pin compatible PyTorch/CUDA wheels; verify compute capability 7.0 at installation rather than assuming newest CUDA wheels work.
- Kaggle T4×2 free notebook: **two distinct 16 GB GPUs**. Run **two independent languages simultaneously**, one process per GPU via `CUDA_VISIBLE_DEVICES`; preserve separate run directories and checkpoints, share precomputed Hindi embeddings; the GPUs do *not* combine VRAM. Note notebook session limits, quotas, CPU/RAM contention and the need to back up ephemeral results.
- Colab Free: usable on a single T4, but runtime/quota availability varies and local files can disappear.
- Lightning AI Pro setup previously requested, including Google Drive/Dropbox rclone pulls/copies, start/sleep/resume and local model caches. The service was reported unavailable at the time.
- Prior speculative training estimates (NOT measured): ~2–10 hours per target on Titan V and ~3–12 hours on T4 for 17k pairs and 5–8 epochs, frozen/cached encoder; other stages excluded. Real benchmarking is required before promising throughput.

## Implemented as of v0.2.0

- v0.1.0: UD projection and repair, CoNLL-U data, confidence metadata, optional SimAlign/translation/morphology, language catalog, parser baselines, evaluation/HTML and offline demo.
- v0.1.1: Lightning setup guide, rclone Google Drive/Dropbox data/model/checkpoint workflows, persistence guidance and helper scripts.
- v0.2.0: new experimental bilingual neural trainer, XLM-R word-embedding caching, source-graph and alignment-aware cross-attention modules, confidence-weighted projected arc/relation supervision and target-only biaffine head, resumable shards/checkpoints, Kaggle two-GPU process launcher and notebook.
- Offline tests passed in development (reported 32 automated tests); actual pretrained XLM-R cache generation, GPU training, multilingual 17k runs and independent gold accuracy were *not benchmarked* at release time. Do not describe this as production-ready Trankit or a published gold treebank.

## Next tasks for Codex

1. Run unit tests and offline demonstration from the checked-out v0.2.0 source.
2. Verify real pretrained model cache generation and inference on Ubuntu Titan V or Kaggle T4×2. Profile GPU RAM and 2k-sentence throughput before full 17k experiments.
3. Audit source→target supervision and soft alignment calibration for cases with missing/incorrect links, and prevent leakage from Hindi/source dev/test into target train.
4. Add full target morphology/Apertium integration into bilingual loss (presently incomplete) with language-specific linguistic rules and ablations.
5. Compare target-gold LAS/UAS against standard Trankit, monolingual HF parser and projection-only baselines.
6. Add independent target-only parser/distillation if needed.
7. Automate persistent checkpoint backups, dual-worker GPU resource monitoring and deterministic resumption for interrupted Kaggle sessions.
8. Keep source corpus license attribution (Hindi HDTB and derivatives may be CC BY-NC-SA) separate from project's code license.

## Important user constraints

Use a conventional Python repository layout (`pyproject.toml`, `src/ud_treebank_lab`, `tests`, `docs`, `scripts`, `AGENTS.md`), easy setup and configuration, GPU days not weeks per language where possible, full checkpoint/resume, no generative AI annotation, and thorough scientific evaluation and reproducibility.

## Update on 2026-11-08: corpus-wide correspondence induction (v0.3.0)

User stressed that recurrent Hindi words, names, inflections, and function-word
realizations over the full 17k sentence pairs offer vital independent clues.
Example: an incorrect token link `ने → शाहजेहन` can be tested against the much
stronger recurring `शाहजहाँ → शाहजेहन` correspondence; the Hindi postposition
may be zero/bound/structural, not an overt target token. Also, initial alignments
and Apertium output may **already exist**. No unwanted recomputation should be
started by default; each resource's mode and path must be in YAML.

Implemented CPU-side `refine-corpus` and optional `corpus_preprocess` in bilingual
config. See `docs/CORPUS_REFINEMENT.md`. The 0.3 code estimates corpus-wide
lexical and sentence cooccurrence, preserves uncertain morph candidate lattices,
records word/NULL/suffix realization hypotheses, saves complete correction audit
metadata, and leaves expensive tool runs opt-in. Future work: statistically
calibrated lexical/morpheme association, phrase and multi-morpheme latent
alignment, differentiable morphological-generation and contextual analysis,
measured gold alignment/LAS gains across all 13 languages.
