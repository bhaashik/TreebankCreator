"""Offline simulated AWESOME CLI; no network, model download, or GPU needed."""
from __future__ import annotations

import json
from pathlib import Path
import stat

import pytest

from ud_treebank_lab.alignment import read_alignments, run_alignment
from ud_treebank_lab.awesome_adapter import parse_awesome_links
from ud_treebank_lab.corpus_config import read_corpus_config
from ud_treebank_lab.corpus_resources import prepare_corpus_resources


def sample_data(tmp_path):
    source = tmp_path / "source.conllu"
    source.write_text("""# sent_id = 1
# text = राम ने पानी पिया ।
1\tराम\tराम\tPROPN\t_\t_\t4\tnsubj\t_\t_
2\tने\tने\tADP\t_\t_\t1\tcase\t_\t_
3\tपानी\tपानी\tNOUN\t_\t_\t4\tobj\t_\t_
4\tपिया\tपीना\tVERB\t_\t_\t0\troot\t_\t_
5\t।\t।\tPUNCT\t_\t_\t4\tpunct\t_\t_

# sent_id = 2
# text = वह आया ।
1\tवह\tवह\tPRON\t_\t_\t2\tnsubj\t_\t_
2\tआया\tआना\tVERB\t_\t_\t0\troot\t_\t_
3\t।\t।\tPUNCT\t_\t_\t2\tpunct\t_\t_

""", encoding="utf-8")
    targets = tmp_path / "targets.jsonl"
    targets.write_text(json.dumps({"id": "1", "tokens": ["राम", "पानी", "पिलें", "।"]}, ensure_ascii=False) + "\n" +
                       json.dumps({"id": "2", "tokens": ["ऊ", "अइल", "।"]}, ensure_ascii=False) + "\n", encoding="utf-8")
    return source, targets


def fake_aligner(tmp_path, content: str = "0-0 2-1 3-2 4-3\n0-0 1-1 2-2\n"):
    tool = tmp_path / "awesome-align"
    tool.write_text("#!/usr/bin/env python3\n"
                    "import sys\n"
                    "from pathlib import Path\n"
                    "args=sys.argv\n"
                    "input_path=Path(args[args.index('--data_file')+1])\n"
                    "output_path=Path(args[args.index('--output_file')+1])\n"
                    f"text={content!r}\n"
                    "assert len(input_path.read_text().splitlines()) == 2\n"
                    "output_path.write_text(text,encoding='utf-8')\n")
    tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
    return tool


def test_awesome_link_validation():
    assert parse_awesome_links("1-0 0-0 1-0", source_count=3, target_count=2) == [[0, 0], [1, 0]]
    assert parse_awesome_links("", source_count=3, target_count=2) == []
    with pytest.raises(ValueError, match="Malformed"):
        parse_awesome_links("1-foo", source_count=3, target_count=2)
    with pytest.raises(ValueError, match="outside word bounds"):
        parse_awesome_links("3-0", source_count=3, target_count=2)


def test_awesome_run_round_trip_resume_and_null(tmp_path):
    src, tgt = sample_data(tmp_path)
    tool = fake_aligner(tmp_path)
    dest = tmp_path / "alignment.jsonl"
    first = run_alignment(src, tgt, dest, backend="awesome", model="xlmr",
                          method="itermax", device="cpu", binary=str(tool))
    assert first["processed_this_call"] == 2
    assert read_alignments(dest)["1"] == [(0, 0), (2, 1), (3, 2), (4, 3)]
    assert run_alignment(src, tgt, dest, backend="awesome", binary=str(tool))["processed_this_call"] == 0


def test_awesome_rejects_shifted_or_missing_lines_without_partial_output(tmp_path):
    src, tgt = sample_data(tmp_path)
    out = tmp_path / "result.jsonl"
    tool = fake_aligner(tmp_path, "0-0\n")
    with pytest.raises(ValueError, match="returned 1 lines for 2"):
        run_alignment(src, tgt, out, backend="awesome", binary=str(tool))
    assert not out.exists()
    tool = fake_aligner(tmp_path, "0-0\n50-0\n")
    with pytest.raises(ValueError, match="outside word bounds"):
        run_alignment(src, tgt, out, backend="awesome", binary=str(tool))
    assert not out.exists()


def test_pipeline_can_generate_with_configured_awesome_binary(tmp_path):
    src, tgt = sample_data(tmp_path)
    tool = fake_aligner(tmp_path)
    cfg = tmp_path / "corpus.yaml"
    import yaml
    cfg.write_text(yaml.safe_dump({
        "source_ud": str(src), "targets": str(tgt), "output_dir": str(tmp_path / "output"),
        "resources": {"alignment": {"mode": "generate", "backend": "awesome",
                                   "binary": str(tool), "model": "bert-base-multilingual-cased",
                                   "method": "softmax", "setup": {"on_missing": "error"}}}
    }), encoding="utf-8")
    preview = prepare_corpus_resources(cfg, dry_run=True)
    assert preview["stages"]["alignment"]["setup"]["status"] == "available"
    result = prepare_corpus_resources(cfg)
    assert result["stages"]["alignment"]["status"] == "generated"
    assert len(read_alignments(result["selected_inputs"]["alignment"])) == 2
    second = prepare_corpus_resources(cfg)
    assert second["stages"]["alignment"]["status"] == "reused-generation"
