# 22 Eighth Schedule Indian languages: initial coverage

All 22 are present in `udtl languages`. The IndicTrans2 language tags reflect **translation model routing**, not availability of a high-quality UD dataset, Apertium dictionary, Stanza constituency model or successful model fine-tuning.

| Language | Code (project) | IndicTrans2 tag | Initial note |
|---|---|---|---|
| Assamese | asm | asm_Beng | Indic script; assess lexicon and tokenizer |
| Bengali | ben | ben_Beng | Evaluate punctuation/clitics and target-specific morphology |
| Bodo | brx | brx_Deva | Low-resource; high-risk weak word alignment |
| Dogri | doi | doi_Deva | Devanagari; possible script sharing, not syntactic identity |
| Gujarati | guj | guj_Gujr | Indo-Aryan; establish gold dev |
| Hindi | hin | hin_Deva | Initial gold UD source, also target baseline |
| Kannada | kan | kan_Knda | Dravidian morphology and case/agreement need special care |
| Kashmiri | kas | kas_Arab | Alternative IndicTrans2 `kas_Deva` also available |
| Konkani | gom | gom_Deva | Devanagari route; other scripts require explicit handling |
| Maithili | mai | mai_Deva | Close to Hindi but alignment/grammar differences significant |
| Malayalam | mal | mal_Mlym | Rich morphology and tokenization; recommend explicit analyzer |
| Manipuri | mni | mni_Mtei | Alternative `mni_Beng` available; verify model and tokenizer |
| Marathi | mar | mar_Deva | Suggested **first pilot** given Hindi lexical/script overlap |
| Nepali | npi | npi_Deva | Close-script pilot after Marathi |
| Odia | ory | ory_Orya | Script-specific alignment/tokenization testing |
| Punjabi | pan | pan_Guru | Case, markers and script-specific treatment |
| Sanskrit | san | san_Deva | Sandhi/compound tokenization demands special work |
| Santali | sat | sat_Olck | Low-resource Munda; script and alignment need inspection |
| Sindhi | snd | snd_Arab | Alternative `snd_Deva` available |
| Tamil | tam | tam_Taml | Rich morphology; Hindi projections need careful restructuring |
| Telugu | tel | tel_Telu | Dravidian; rich case and verbal morphology |
| Urdu | urd | urd_Arab | Hindi/Urdu linguistic proximity but script and spelling differ |

**Important:** Model tag names are **not ISO 639-1**, and some have script suffixes. Where multiple writing systems are supported, the current language catalog contains one default route per language; set an alternate IndicTrans2 tag manually in YAML if appropriate. Do not transliterate whole treebanks indiscriminately: orthography affects segmentation and morphology. Prefer consistent native scripts throughout target text, analyzer and gold annotation.

## Suggested research order

**Phase I:** Marathi and Nepali, then Gujarati, Bengali, Punjabi and Urdu. This order is an experiment design heuristic, not a guaranteed quality ordering; parallel-data availability and UD gold supervision matter more than perceived closeness to Hindi.

**Phase II:** Kannada, Telugu, Malayalam, Tamil and Odia, explicitly adding target morphological analyzers/tokenizers and a larger independent gold set.

**Phase III:** Assamese, Maithili, Konkani, Dogri, Bodo, Kashmiri, Sindhi, Manipuri, Santali and Sanskrit, prioritizing language-specific segmentation, named entities, auxiliaries, case marking, borrowing, compounds and scarce target resources.

For each language: (1) identify the conventional target UD orthography and word segmentation; (2) assess parallel data/translation quality; (3) calibrate SimAlign against ~100 checked word-alignment examples; (4) project 1k–5k UD sentences; (5) review 100–300 gold target sentences; (6) only then scale. Additional source treebanks in `pairs` may reduce transfer bias.

## When IndicTrans2 is inadequate

The project can accept **any** target-language parallel sentences as JSONL; translation is not required. IndicTrans2 model cards now mention an emerging **Indic-Translate** successor. This package's implemented MT adapter is for the tested/documented **IndicTrans2 inference interface**, not a claim that IndicTrans2 remains the best model in every target language as of 2026. If using newer NMT/parallel corpora, export the same `targets.jsonl` schema; the downstream annotation pipeline stays unchanged.

## Source licenses

Source HDTB is CC BY-NC-SA 4.0. Other UD treebanks have **different** licenses (some CC BY, some restricted). Verify each source's README/LICENSE, plus any translations/parallel corpora and model licenses, before publishing a derived corpus, scripts or commercial model.
