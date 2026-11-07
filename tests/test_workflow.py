from pathlib import Path
import json

import pytest

from ud_treebank_lab.cli import main
from ud_treebank_lab.conllu import read_conllu
from ud_treebank_lab.evaluation import evaluate
from ud_treebank_lab.synthetic import create_from_templates
from ud_treebank_lab.training import prepare_data
from ud_treebank_lab.utils import jsonl
from ud_treebank_lab.workflow import run_workflow

EX = Path(__file__).resolve().parent.parent / "examples"


def test_pipeline_and_resume(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("project:\n  target: mar\n  output_dir: ./run\n"
                   "pairs:\n  - source_ud: " + str(EX / "hindi_source.conllu") + "\n"
                   "    targets: " + str(EX / "marathi_targets.jsonl") + "\n"
                   "    alignments: " + str(EX / "manual_alignments.jsonl") + "\n"
                   "projection:\n  min_quality: 0.1\ntraining:\n  backend: none\n", encoding="utf-8")
    result = run_workflow(cfg)
    repeated = run_workflow(cfg)
    assert result["pairs"][0]["silver"]["processed_this_call"] == 3
    assert repeated["pairs"][0]["silver"]["processed_this_call"] == 0
    assert (tmp_path / "run/run_manifest.json").is_file()
    assert len(list(read_conllu(tmp_path / "run/training_data/train.conllu"))) >= 1


def test_gold_eval_and_leakage(tmp_path):
    output = tmp_path / "data"
    from ud_treebank_lab.projection import run_projection
    silver = output / "s.conllu"
    meta = output / "m.jsonl"
    run_projection(EX / "hindi_source.conllu", EX / "marathi_targets.jsonl",
                   EX / "manual_alignments.jsonl", silver, meta, prefix="p01:")
    results = evaluate(EX / "marathi_gold_sample.conllu", silver)
    assert results["scores"]["LAS"] == 1.0  # tiny handcrafted fixture, not a benchmark
    assert results["scores"]["UPOS_accuracy"] == 1.0
    split = prepare_data([silver], [meta], output / "training", min_quality=0.0,
                         gold_test=str(EX / "marathi_gold_sample.conllu"),
                         gold_train=str(EX / "marathi_gold_sample.conllu"))
    assert split["rejected_gold_overlap"] == 3
    assert split["gold_train_sentences"] == 3
    assert split["test_gold_sentences"] == 3


def test_eval_mismatched_tokenization(tmp_path):
    from ud_treebank_lab.projection import run_projection
    out = tmp_path / "out.conllu"
    run_projection(EX / "hindi_source.conllu", EX / "marathi_targets.jsonl",
                   EX / "manual_alignments.jsonl", out, tmp_path / "m.jsonl")
    with pytest.raises(ValueError, match="Tokenization differs"):
        evaluate(EX / "hindi_source.conllu", out)


def test_synthetic(tmp_path):
    results = create_from_templates(str(EX / "synthetic_templates.json"), str(tmp_path))
    assert results["generated"] == 2
    assert len(list(read_conllu(tmp_path / "source.conllu"))) == 2
    assert len(list(jsonl(tmp_path / "alignments.jsonl"))) == 2


def test_cli_languages(capsys):
    assert main(["languages"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert len(output) == 22
    assert output["mar"]["indictrans2"] == "mar_Deva"


def test_visualize(tmp_path):
    from ud_treebank_lab.visualize import render_html
    result = render_html(EX / "hindi_source.conllu", tmp_path / "render.html")
    assert result["rendered_trees"] == 3
    contents = (tmp_path / "render.html").read_text(encoding="utf-8")
    assert "<svg" in contents and "राम" in contents


def test_model_reference_resolution(tmp_path):
    from ud_treebank_lab.workflow import _model_ref
    base = tmp_path / "configs"
    assert _model_ref(base, "xlm-roberta-base") == "xlm-roberta-base"
    assert (_model_ref(base, "ai4bharat/indictrans2-indic-indic-dist-320M")
            == "ai4bharat/indictrans2-indic-indic-dist-320M")
    assert _model_ref(base, "../assets/models/xlmr") == str(tmp_path / "assets/models/xlmr")
    assert _model_ref(base, "/tmp/model") == "/tmp/model"
