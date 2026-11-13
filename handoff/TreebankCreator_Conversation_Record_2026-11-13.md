# TreebankCreator — Conversation Record and Technical Handoff

Compiled: 2026-11-13 (Asia/Kolkata).
Repository: https://github.com/bhaashik/TreebankCreator

**Completeness disclosure:** This is a reconstructed chronological handoff, NOT a verbatim ChatGPT export. Accessible user requests are quoted when available; assistant outputs are summarized. Earlier portions of the chat were only available through technical handoff notes. Tool call logs and metadata are not included. Obtain the official ChatGPT conversation export if a complete verbatim record is required.

The conversation file catalog contained 116 generated-file records and no records tagged as user uploads. The downloadable full ZIP captures 75 recoverable working-directory files, with a SHA-256 manifest. A separate selected ZIP contains 18 key files. Git tags preserve v0.1.0 through v0.3.4 source ancestry.

## Chronological transcript reconstruction

### 1. Research foundation

**User request / discussed issue**

Research project: 13 low-resource Indic languages, ~17,000 Hindi–target sentence pairs per language, Hindi gold UD, frozen XLM-R embeddings and bilingual source graph/cross-attention with target-only parser.

**Assistant outcome — summary, not verbatim**

Initial v0.1.0, v0.1.1, v0.2.0 project iterations. Earlier dialogue is available only through a prior handoff summary.


### 2. Bhojpuri failure

**User request / discussed issue**

एकर निर्माण नवाब शाहजेहन कइले रहलें । Example included incorrect token correspondence शाहजेहन ↔ ने, wrong copied lemmas/UPOS, while dependency tree passed structural validation.

**Assistant outcome — summary, not verbatim**

v0.2.0 could not reliably repair systematic shifted word alignments, null Hindi function-word realizations or target lemmas.


### 3. Morphological generation

**User request / discussed issue**

One way could to create a module for morphologican generation (inflection generation) as in CoNLL shared task on this problem. This module can then be integrated into the architecture so that alignment corrections is an almost first class part of it. For this, Apertium morph anaylyzer output could be used (to get potentially multiple analyses), and Unimorph data (where available, what amount) could be used to disambiguate analyses. If neither is available, there may still be enough information in the data (source and target) to achieve reasonably good generation.

**Assistant outcome — summary, not verbatim**

Proposed Apertium analysis, UniMorph paradigms, contextual disambiguation and character-level morphological generation; full neural inflection generator not yet implemented.


### 4. Global consistency

**User request / discussed issue**

There is another very useful information at the global dataset level which can make the task much easier. We are not doing everything from scratch for the generation part. In the case of शाहजेहन ↔ ने, ने will occur very often in the source and in majority of places it is likely to be correct. This likely-correct information, based on frequency kind of things, can be used to check consistency. It won't solve the problem completely, but the gains might be very significant.

Even शाहजेहन might have the correct name lemma (शाहजहाँ) is many places if it occurs many times in the data. This will also give a clue that two Hindi words align with one target word. Of course, we can't apply the whole process on the data along with training, but it could be done prior to training proper.

The main point is that function word to to/from affix mapping (if Apertium and Unimorph data are not available) is likely to be a big clue in a large number of cases.

**Assistant outcome — summary, not verbatim**

Corpus-wide bilingual lexicon, name anchors, NULL links, function-word-to-affix hypotheses, co-occurrence statistics and safeguards against self-reinforcing bad alignments.


### 5. Preprocessing implementation

**User request / discussed issue**

Go ahead. In the pipeline, take into account the fact that a few things like initial word alignment and  Apertium output might be available, so that these are not run by default. The config file can specify this, and where the data is.

**Assistant outcome — summary, not verbatim**

v0.3.0: opt-in corpus-wide repair, supplied/generate/skip resources, retained confidence/uncertainty; 43 offline tests reported.


### 6. History and Pages

**User request / discussed issue**

Can you push with history after each version. The repo is now public, so Pages->Actions-Deploy-with-actions is available. The packages can also be uploaded, although that is not crucial for now.

**Assistant outcome — summary, not verbatim**

Original version history merged without rewriting existing remote commits; tags v0.1.0–v0.3.0 added and MkDocs Pages Actions configured.


### 7. Enable Pages

**User request / discussed issue**

Deploy from GH allowed.

**Assistant outcome — summary, not verbatim**

Pages deployment retried and later succeeded.


### 8. Import existing resources

**User request / discussed issue**

Add a steps to convert word alignment and morphology files from .A3.final (awesome-align format) and Apertium output to JSOL for cases when these are already available, by a separate run.

**Assistant outcome — summary, not verbatim**

v0.3.1: import-alignments/import-apertium/import-resources; supports GIZA A3 and AWESOME i-j, alternative Apertium analyses; 58 tests.


### 9. Conditional generation/setup

**User request / discussed issue**

Is this subject to config specs saying it is already available? If not, please do that. If not available, they can be run, again provided the setup for them is available. The setup part can also be conditional on whether the setup is present or not. This will increase flexibility and also avoid confusion.

**Assistant outcome — summary, not verbatim**

v0.3.2: mode:auto, available:auto, if_missing, setup.on_missing, explicit install_command, prepare-resources --dry-run; 66 tests.


### 10. Installer

**User request / discussed issue**

It would be a good idea to add a script for installing Apertium and neural word alignment (if not feasible, awesome-align). Or is it already present?

**Assistant outcome — summary, not verbatim**

v0.3.3: install_linguistic_tools.sh, Apertium engine and language analyzers, SimAlign/AWESOME-align and checks/dry-run; 75 tests.


### 11. Comparing aligners

**User request / discussed issue**

For Indic languages, is SimAlign likely to be better or awesome-align?

**Assistant outcome — summary, not verbatim**

SimAlign+XLM-R+IterMax recommended baseline; compare with fine-tuned AWESOME-align; language-specific gold alignment sample essential.


### 12. HP Pavilion CPU

**User request / discussed issue**

OK. Is it feasible to run it on HP Pavillion x360 cd14xxxx Ubuntu laptop?

**Assistant outcome — summary, not verbatim**

SimAlign and corpus refinement feasible on laptop with 32 GB RAM; GPU preferred for fine-tuning.


### 13. Titan V

**User request / discussed issue**

And on Titan V GPU destop?

**Assistant outcome — summary, not verbatim**

Titan V sm_70/12 GB supports GPU alignment/fine-tuning with compatible pinned CUDA/PyTorch environment.


### 14. Titan V profile

**User request / discussed issue**

Go ahead. Also update README. Roughly how much time Titav V per language?

**Assistant outcome — summary, not verbatim**

v0.3.4 added Titan V setup/check/activate scripts and docs/TITAN_V.md. Rough estimates per 17K pairs: SimAlign 15–90 min, refinement 5–30 min, parser 2–8 h; not measured. 82 tests reported.


### 15. Current handoff

**User request / discussed issue**

Prepare a complete record of this conversation to be push to GH repo: 1. MD file of this entire conversation. 2. Complete handoff zip of all the file uploaded and generated. 3. Shorter handoff zip for the most relevant files, uploaded and generated, based on 2. Make these available for downloadable here and also push them to separate branches (if not already there).

**Assistant outcome — summary, not verbatim**

Handoff performed on 2026-11-13. The authoritative raw ChatGPT export and file attachments absent from the accessible file catalog cannot be fabricated.

## Release history

| Version | Main contribution |
|---|---|
| v0.1.0 | Initial UD projection |
| v0.1.1 | Lightning AI and cloud guides |
| v0.2.0 | Bilingual source-guided parser / dual GPU |
| v0.3.0 | Corpus-wide alignment refinement |
| v0.3.1 | Existing resource format converters |
| v0.3.2 | Conditional resource/setup orchestration |
| v0.3.3 | Neural aligner and Apertium installers |
| v0.3.4 | Titan V CUDA profile |

Official technical handoffs: docs/CONVERSATION_HANDOFF.md, docs/RELEASE_PROVENANCE.md, docs/TITAN_V.md in the primary repository.

**Important:** The versioned neural parser does not yet implement full trainable inflection generation. Alignment correctness and LAS must be evaluated against independent target-language gold data. Runtime estimates were not measured on Titan V.
