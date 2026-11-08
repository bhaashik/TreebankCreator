# Bhaashik TreebankCreator

An open-source research prototype for constructing low-resource **Universal Dependencies (UD) treebanks** using parallel sentences, Hindi gold-standard dependency trees, bilingual alignment evidence and resource-adaptive morphological analysis.

## Current release: v0.3.0

- **v0.1.0:** UD conversion and projection, parser training and evaluation.
- **v0.1.1:** Lightning AI setup, cloud resources and resumable workflows.
- **v0.2.0:** Frozen-XLM-R bilingual Hindi/target parser with graph-aware cross-attention and target-only dependency head; multi-GPU launcher.
- **v0.3.0:** Corpus-wide lexical and grammatical consistency, confidence-aware alignment repair, optional existing Apertium and alignment inputs.

All four versions are preserved as tagged Git history on the [repository](https://github.com/bhaashik/TreebankCreator).

## Documentation

Start with [installation](INSTALL.md), then read [data preparation](DATA.md) and [corpus-wide alignment refinement](CORPUS_REFINEMENT.md). The [bilingual model guide](BILINGUAL_TRAINING.md) explains the model and its training limitations. For compute environments, see [Kaggle T4x2](KAGGLE_DUAL_T4.md) or [Lightning AI](LIGHTNING_AI_PRO.md).

## Research limitations

The system is an **experimental research implementation**. Automatically projected CoNLL-U data is not equivalent to independently verified gold UD data. Alignment and morphology repair are estimates, and production accuracy and GPU timing require empirical evaluation on each target language.