"""Command-line interface for UD Treebank Lab."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.request import urlopen

import yaml

from . import __version__


def _report(data) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="udtl", description="UD treebank annotation projection and training")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="Run a YAML-configured, resumable multi-source project")
    r.add_argument("config")
    r.add_argument("--train", choices=["none", "trankit", "hf"], default=None)
    r = sub.add_parser("fetch-ud", help="Download source UD CoNLL-U files; respect source license")
    r.add_argument("--treebank", default="UD_Hindi-HDTB")
    r.add_argument("--short-name", default="hi_hdtb")
    r.add_argument("--out", default="data/ud_hindi_hdtb")
    r.add_argument("--splits", nargs="+", choices=["train", "dev", "test"], default=["train", "dev", "test"])
    r = sub.add_parser("languages", help="Show all 22 scheduled Indian languages and IndicTrans codes")
    r = sub.add_parser("synthesize", help="Expand linguist-authored bilingual UD templates")
    r.add_argument("--templates", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--max-examples", type=int, default=10000)
    r = sub.add_parser("validate", help="Validate basic UD tree structure")
    r.add_argument("conllu")
    r = sub.add_parser("align", help="Align source CoNLL-U and target JSONL")
    r.add_argument("--source", required=True)
    r.add_argument("--targets", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--backend", choices=["simalign", "lexicon", "identity"], default="simalign")
    r.add_argument("--model", default="xlmr")
    r.add_argument("--method", choices=["itermax", "inter", "mwmf"], default="itermax")
    r.add_argument("--device", default="auto")
    r.add_argument("--lexicon", help="JSON file mapping source forms to target forms (lexicon backend)")
    r = sub.add_parser("project", help="Project single aligned source-target pair to CoNLL-U")
    r.add_argument("--source", required=True)
    r.add_argument("--targets", required=True)
    r.add_argument("--alignments", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--metadata", required=True)
    r.add_argument("--morph")
    r.add_argument("--rules", help="YAML language-specific soft constraints")
    r.add_argument("--prefix", default="")
    r = sub.add_parser("translate", help="Translate source UD with IndicTrans2 (optional)")
    r.add_argument("--source", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--src-lang", default="hin_Deva")
    r.add_argument("--tgt-lang", required=True)
    r.add_argument("--model", default="ai4bharat/indictrans2-indic-indic-dist-320M")
    r.add_argument("--batch-size", type=int, default=8)
    r.add_argument("--device", default="auto")
    r = sub.add_parser("morph", help="Get conservative Apertium morphological analyses")
    r.add_argument("--targets", required=True)
    r.add_argument("--analyzer-bin", required=True)
    r.add_argument("--out", required=True)
    r = sub.add_parser("prepare", help="Filter and split silver training treebanks")
    r.add_argument("--silver", nargs="+", required=True)
    r.add_argument("--metadata", nargs="+", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--min-quality", type=float, default=.45)
    r.add_argument("--dev-fraction", type=float, default=.10)
    r.add_argument("--seed", type=int, default=17)
    r.add_argument("--gold-train")
    r.add_argument("--gold-dev")
    r.add_argument("--gold-test")
    r.add_argument("--gold-repeat", type=int, default=3)
    r = sub.add_parser("train-trankit", help="Train Trankit customized POS/dependency model")
    r.add_argument("--train", required=True)
    r.add_argument("--dev", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--task", default="posdep", choices=["posdep"])
    r.add_argument("--category", default="customized")
    r = sub.add_parser("train-hf", help="Train checkpointable HF encoder POS/dependency parser")
    r.add_argument("--train", required=True)
    r.add_argument("--dev", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--pretrained", default="xlm-roberta-base")
    r.add_argument("--epochs", type=int, default=5)
    r.add_argument("--batch-size", type=int, default=4)
    r.add_argument("--accumulate", type=int, default=2)
    r.add_argument("--lr", type=float, default=.0003)
    r.add_argument("--unfreeze-encoder", action="store_true")
    r.add_argument("--max-words", type=int, default=70)
    r.add_argument("--max-subtokens", type=int, default=256)
    r.add_argument("--save-steps", type=int, default=200)
    r.add_argument("--seed", type=int, default=17)
    r.add_argument("--no-resume", action="store_true")
    r.add_argument("--device", default="auto")
    r = sub.add_parser("predict-hf", help="Predict UD tags/trees with trained HF model")
    r.add_argument("--model-dir", required=True)
    r.add_argument("--input", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--checkpoint", default="best.pt")
    r.add_argument("--device", default="auto")
    # Bilingual source-guided training. This is NOT Trankit.TPipeline training;
    # it uses frozen cached XLM-R embeddings plus new bilingual components.
    r = sub.add_parser("make-bilingual-configs", help="Generate configs for a 13-language manifest")
    r.add_argument("--manifest", required=True)
    r.add_argument("--source", required=True)
    r.add_argument("--config-dir", required=True)
    r.add_argument("--cache-root", required=True)
    r.add_argument("--runs-root", required=True)
    r.add_argument("--model", default="xlm-roberta-base")
    r.add_argument("--epochs", type=int, default=6)
    r.add_argument("--batch-size", type=int, default=4)
    r.add_argument("--max-pairs", type=int, default=None)
    r = sub.add_parser("import-alignments", help="Convert existing AWESOME Pharaoh or GIZA A3 to JSONL")
    r.add_argument("--source-ud", required=True)
    r.add_argument("--targets", required=True)
    r.add_argument("--input", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--format", choices=["auto", "awesome", "a3"], default="auto")
    r.add_argument("--probabilities", help="Optional AWESOME-align per-link probability file")
    r.add_argument("--sentence-order", choices=["source_ud", "targets"], default="source_ud")
    r.add_argument("--a3-source-side", choices=["auto", "annotated", "plain"], default="auto")
    r.add_argument("--no-verify-forms", action="store_true")
    r.add_argument("--overwrite", action="store_true")
    r = sub.add_parser("import-apertium", help="Convert existing Apertium stream to JSONL lattices")
    r.add_argument("--targets", required=True)
    r.add_argument("--input", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--layout", choices=["auto", "sentence-lines", "token-lines", "stream", "word-list"],
                   default="auto")
    r.add_argument("--no-verify-forms", action="store_true")
    r.add_argument("--overwrite", action="store_true")
    r = sub.add_parser("import-resources", help="Separate YAML-configured import of supplied resources")
    r.add_argument("--config", required=True)
    r.add_argument("--overwrite", action="store_true")
    r = sub.add_parser("prepare-resources", help="Conditionally reuse/import/generate corpus inputs")
    r.add_argument("--config", required=True, help="Corpus refinement YAML")
    r.add_argument("--dry-run", action="store_true", help="Report actions without writing or installing")
    r.add_argument("--force", action="store_true", help="Regenerate only managed derived resources")
    r = sub.add_parser("refine-corpus", help="CPU-only corpus-wide alignment/morphology induction")
    r.add_argument("--config", required=True, help="Corpus preprocessing YAML with explicit resource modes")
    r.add_argument("--force", action="store_true", help="Recompute derived outputs, never supplied inputs")
    r = sub.add_parser("cache-bilingual", help="Cache Hindi/target frozen XLM-R word vectors")
    r.add_argument("--config", required=True)
    r.add_argument("--side", choices=["both", "source", "target"], default="both")
    r.add_argument("--device", default=None)
    r = sub.add_parser("train-bilingual", help="Train source-guided target-only bilingual dependency parser")
    r.add_argument("--config", required=True)
    r.add_argument("--device", default=None)
    r.add_argument("--max-steps", type=int, help="Pause after N optimizer steps (testing/resume)")
    r.add_argument("--no-auto-cache", action="store_true")
    r = sub.add_parser("predict-bilingual", help="Export source-guided silver target CoNLL-U")
    r.add_argument("--config", required=True)
    r.add_argument("--out", default=None)
    r.add_argument("--checkpoint", default="best.pt")
    r.add_argument("--device", default="auto")
    r = sub.add_parser("launch-bilingual", help="Run languages independently on distinct Kaggle/desktop GPUs")
    r.add_argument("--configs", nargs="+", required=True)
    r.add_argument("--gpus", default="0,1", help="Comma-separated physical GPU IDs e.g. 0,1")
    r.add_argument("--no-precache-source", action="store_true")
    r.add_argument("--dry-run", action="store_true")
    r = sub.add_parser("export-stanza", help="Export Stanza-compatible train/dev file layout")
    r.add_argument("--train", required=True)
    r.add_argument("--dev", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--language", required=True, help="UD short language code, e.g. mr")
    r.add_argument("--treebank", default="Silver")
    r = sub.add_parser("parse-source", help="Create UD source side from aligned corpus via Stanza")
    r.add_argument("--input", required=True, help="JSONL {id,text}")
    r.add_argument("--out", required=True)
    r.add_argument("--lang", default="hi")
    r.add_argument("--package", default="default")
    r.add_argument("--cpu", action="store_true")
    r = sub.add_parser("constituency", help="Stanza .mrg inference for supported conparse languages")
    r.add_argument("--input", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--lang", required=True)
    r.add_argument("--package", default="default")
    r.add_argument("--cpu", action="store_true")
    r = sub.add_parser("evaluate", help="Compute strict gold-tokenized metrics and errors")
    r.add_argument("--gold", required=True)
    r.add_argument("--pred", required=True)
    r.add_argument("--out", required=True)
    r = sub.add_parser("official-eval", help="Run an installed official CoNLL18 evaluator script")
    r.add_argument("--script", required=True, help="Path to conll18_ud_eval.py")
    r.add_argument("--gold", required=True)
    r.add_argument("--pred", required=True)
    r.add_argument("--out", required=True)
    r = sub.add_parser("visualize", help="Create standalone HTML with dependency SVG")
    r.add_argument("--input", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--metrics")
    r.add_argument("--metadata")
    r.add_argument("--limit", type=int, default=40)
    return parser


def fetch_ud(treebank: str, short: str, out: str, splits: list[str]) -> dict:
    if not re.fullmatch(r"UD_[A-Za-z0-9_-]+", treebank) or not re.fullmatch(r"[a-z0-9_]+", short):
        raise ValueError("Invalid treebank or short name")
    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for split in splits:
        name = f"{short}-ud-{split}.conllu"
        uri = f"https://raw.githubusercontent.com/UniversalDependencies/{treebank}/master/{name}"
        try:
            with urlopen(uri, timeout=120) as f:
                contents = f.read()
        except Exception as exc:
            raise RuntimeError(f"Download failed: {uri} ({exc})") from exc
        (directory / name).write_bytes(contents)
        files[split] = str(directory / name)
    return {"files": files, "source": f"https://github.com/UniversalDependencies/{treebank}",
            "license_warning": "Read individual treebank LICENSE/README; Hindi HDTB is CC BY-NC-SA 4.0"}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.cmd == "languages":
            from .languages import SCHEDULED_LANGUAGES
            _report({code: {"name": name, "indictrans2": tag}
                     for code, (name, tag) in SCHEDULED_LANGUAGES.items()})
        elif args.cmd == "fetch-ud":
            _report(fetch_ud(args.treebank, args.short_name, args.out, args.splits))
        elif args.cmd == "synthesize":
            from .synthetic import create_from_templates
            _report(create_from_templates(args.templates, args.out, args.max_examples))
        elif args.cmd == "validate":
            from .conllu import read_conllu, validate_tree
            errors = []
            n = 0
            for sentence in read_conllu(args.conllu):
                n += 1
                bad = validate_tree(sentence)
                if bad: errors.append({"id": sentence.sent_id, "errors": bad})
            _report({"sentences": n, "invalid": len(errors), "first_errors": errors[:20]})
            return 2 if errors else 0
        elif args.cmd == "run":
            from .workflow import run_workflow
            _report(run_workflow(args.config, args.train))
        elif args.cmd == "align":
            from .alignment import run_alignment
            lexicon = json.loads(Path(args.lexicon).read_text(encoding="utf-8")) if args.lexicon else None
            _report(run_alignment(args.source, args.targets, args.out, args.backend, args.model,
                                  args.method, args.device, lexicon))
        elif args.cmd == "project":
            from .projection import run_projection
            rules = yaml.safe_load(Path(args.rules).read_text(encoding="utf-8")) if args.rules else None
            _report(run_projection(args.source, args.targets, args.alignments, args.out,
                                   args.metadata, args.morph, rules, prefix=args.prefix))
        elif args.cmd == "translate":
            from .translate import translate_ud
            _report(translate_ud(args.source, args.out, args.src_lang, args.tgt_lang, args.model,
                                 args.batch_size, device=args.device))
        elif args.cmd == "morph":
            from .morphology import run_apertium
            _report(run_apertium(args.targets, args.out, args.analyzer_bin))
        elif args.cmd == "prepare":
            from .training import prepare_data
            _report(prepare_data(args.silver, args.metadata, args.out, args.min_quality,
                                 args.dev_fraction, args.seed, args.gold_train, args.gold_dev,
                                 args.gold_test, args.gold_repeat))
        elif args.cmd == "train-trankit":
            from .training import train_trankit
            _report(train_trankit(args.train, args.dev, args.out, args.task, args.category))
        elif args.cmd == "train-hf":
            from .hf_parser import train_hf
            _report(train_hf(args.train, args.dev, args.out, args.pretrained, args.epochs,
                             args.batch_size, args.accumulate, args.lr, not args.unfreeze_encoder,
                             args.max_words, args.max_subtokens, args.seed, args.save_steps,
                             not args.no_resume, args.device))
        elif args.cmd == "predict-hf":
            from .hf_parser import predict_hf
            _report(predict_hf(args.model_dir, args.input, args.out, args.checkpoint, args.device))
        elif args.cmd == "make-bilingual-configs":
            from .bilingual_manifest import make_configs
            _report(make_configs(args.manifest, args.source, args.config_dir,
                                 args.cache_root, args.runs_root, model=args.model,
                                 epochs=args.epochs, batch_size=args.batch_size,
                                 max_pairs=args.max_pairs))
        elif args.cmd == "import-alignments":
            from .resource_import import import_alignments
            _report(import_alignments(source_ud=args.source_ud, targets=args.targets,
                                      input_path=args.input, output=args.out,
                                      format=args.format, probabilities=args.probabilities,
                                      sentence_order=args.sentence_order,
                                      a3_source_side=args.a3_source_side,
                                      verify_forms=not args.no_verify_forms,
                                      overwrite=args.overwrite))
        elif args.cmd == "import-apertium":
            from .resource_import import import_apertium
            _report(import_apertium(targets=args.targets, input_path=args.input,
                                    output=args.out, layout=args.layout,
                                    verify_forms=not args.no_verify_forms,
                                    overwrite=args.overwrite))
        elif args.cmd == "import-resources":
            from .resource_import import import_resources
            _report(import_resources(args.config, overwrite=args.overwrite))
        elif args.cmd == "prepare-resources":
            from .corpus_resources import prepare_corpus_resources
            _report(prepare_corpus_resources(args.config, force=args.force, dry_run=args.dry_run))
        elif args.cmd == "refine-corpus":
            from .corpus_refinement import run_corpus_refinement
            _report(run_corpus_refinement(args.config, force=args.force))
        elif args.cmd == "cache-bilingual":
            from .bilingual_training import cache_for_config
            _report(cache_for_config(args.config, args.side, args.device))
        elif args.cmd == "train-bilingual":
            from .bilingual_training import train_bilingual
            _report(train_bilingual(args.config, device=args.device,
                                    max_steps=args.max_steps, auto_cache=not args.no_auto_cache))
        elif args.cmd == "predict-bilingual":
            from .bilingual_training import predict_bilingual
            _report(predict_bilingual(args.config, out_file=args.out,
                                      checkpoint=args.checkpoint, device=args.device))
        elif args.cmd == "launch-bilingual":
            from .bilingual_launcher import launch
            result = launch(args.configs, args.gpus.split(","),
                            precache_source=not args.no_precache_source,
                            dry_run=args.dry_run)
            _report(result)
            return 1 if result.get("failed") else 0
        elif args.cmd == "export-stanza":
            from .training import export_stanza
            _report(export_stanza(args.train, args.dev, args.out, args.language, args.treebank))
        elif args.cmd == "parse-source":
            from .stanza_adapter import parse_parallel_source
            _report(parse_parallel_source(args.input, args.out, args.lang, args.package, not args.cpu))
        elif args.cmd == "constituency":
            from .stanza_adapter import constituency_infer
            _report(constituency_infer(args.input, args.out, args.lang, args.package, not args.cpu))
        elif args.cmd == "evaluate":
            from .evaluation import evaluate
            _report(evaluate(args.gold, args.pred, args.out))
        elif args.cmd == "official-eval":
            import subprocess
            proc = subprocess.run([sys.executable, args.script, "-v", args.gold, args.pred],
                                  capture_output=True, text=True, check=True)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(proc.stdout, encoding="utf-8")
            _report({"report": args.out, "output": proc.stdout})
        elif args.cmd == "visualize":
            from .visualize import render_html
            _report(render_html(args.input, args.out, args.metrics, args.metadata, args.limit))
        else:
            raise AssertionError(args.cmd)
        return 0
    except (ValueError, KeyError, OSError, RuntimeError, __import__("subprocess").CalledProcessError) as exc:
        print(f"udtl: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
