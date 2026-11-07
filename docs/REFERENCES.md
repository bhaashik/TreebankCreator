# Technical references and model resources

These are source/repository links consulted during project design, **not** a claim that every third-party integration was executed here. Verify licenses, model compatibility and availability before building a new dataset. Accessed November 2026.

1. Universal Dependencies, [UD Hindi-HDTB](https://github.com/UniversalDependencies/UD_Hindi-HDTB). Original HDTB source is **CC BY-NC-SA 4.0**. Also see [UD portal](https://universaldependencies.org/).
2. CoNLL 2018 Shared Task, [evaluation metrics and scorer](https://universaldependencies.org/conll18/evaluation.html). The official scorer computes word/token segmentation, tagging, UAS/LAS, MLAS and BLEX; tool code MPL-2.0.
3. Sabet et al., [SimAlign: High Quality Word Alignments Without Parallel Training Data](https://github.com/cisnlp/simalign) (EMNLP 2020), MIT license.
4. Dou and Neubig, [awesome-align: Word Alignment by Fine-tuning Embeddings](https://github.com/neulab/awesome-align) (ACL 2021).
5. Nguyen et al., [Trankit](https://github.com/nlp-uoregon/trankit) and [training a customized pipeline](https://trankit.readthedocs.io/en/latest/training.html) (joint POS/morph/dependency, multilingual adapters).
6. Qi et al., [Stanza](https://stanfordnlp.github.io/stanza/) and [training/evaluation](https://stanfordnlp.github.io/stanza/training_and_evaluation.html); [constituency models](https://stanfordnlp.github.io/stanza/constituency.html) and [adding constituency models](https://stanfordnlp.github.io/stanza/new_language_constituency.html).
7. Gala et al., [IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) (2023), translation of all 22 Eighth Schedule languages. [Indic-Indic distilled 320M model](https://huggingface.co/ai4bharat/indictrans2-indic-indic-dist-320M). Model access/license vary by checkpoint; check the card.
8. [IndicTransToolkit](https://pypi.org/project/indictranstoolkit/) (runtime for IndicTrans2), including model-specific input normalization, tokenization and postprocessing.
9. [Apertium lttoolbox](https://github.com/apertium/lttoolbox), FST morphology (`lt-proc`). Apertium analyzer data/license varies by language.
10. [Udapi](https://github.com/udapi/udapi-python), Python tools for UD processing; useful for production treebank conversion/visualization.
11. [UDPipe 2](https://ufal.mff.cuni.cz/udpipe/2/models), additional pretrained UD tools and baselines.
12. [XLM-RoBERTa](https://huggingface.co/FacebookAI/xlm-roberta-base) multilingual transformer backbone for HF training and alignment alternatives.

The project itself uses **NetworkX** for exact directed arborescence decoding; this is a graph-validity repair device, not a linguistic parser architecture claim.
