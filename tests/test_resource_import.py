"""No-network regression tests for existing-resource import and downstream readers."""
import json
from pathlib import Path

import pytest
import yaml

from ud_treebank_lab.alignment import read_alignments
from ud_treebank_lab.corpus_morph import read_apertium_candidates
from ud_treebank_lab.corpus_refinement import run_corpus_refinement
from ud_treebank_lab.resource_import import import_alignments, import_apertium, import_resources
from ud_treebank_lab.utils import jsonl
from ud_treebank_lab.cli import main


def sample(tmp):
    (tmp / "source.conllu").write_text(
        "# sent_id = one\n1\tनवाब\tनवाब\tNOUN\t_\t_\t3\tnsubj\t_\t_\n"
        "2\tने\tने\tADP\t_\t_\t1\tcase\t_\t_\n"
        "3\tबनाया\tबनाना\tVERB\t_\t_\t0\troot\t_\t_\n\n"
        "# sent_id = two\n1\tराम\tराम\tPROPN\t_\t_\t2\tnsubj\t_\t_\n"
        "2\tगया\tजाना\tVERB\t_\t_\t0\troot\t_\t_\n\n", encoding="utf-8"
    )
    (tmp / "targets.jsonl").write_text(
        '{"id":"one","tokens":["नवाब","बनवलें"]}\n'
        '{"id":"two","tokens":["राम","गइलें"]}\n', encoding="utf-8"
    )
    return {"source_ud": str(tmp / "source.conllu"), "targets": str(tmp / "targets.jsonl")}


def test_awesome_pharaoh_blank_line_probabilities_and_schema(tmp_path):
    kw = sample(tmp_path)
    (tmp_path / "links.A3.final").write_text("0-0 2-1 1-0\n\n")
    (tmp_path / "probs.txt").write_text("0.95 0.80 0.13\n\n")
    dst = tmp_path / "out.jsonl"
    result = import_alignments(**kw, input_path=str(tmp_path / "links.A3.final"),
                               probabilities=str(tmp_path / "probs.txt"), output=str(dst))
    assert result["format"] == "awesome"  # extension is NOT a format detector
    rows = list(jsonl(dst))
    assert [r["id"] for r in rows] == ["one", "two"]
    assert rows[0]["pairs"] == [[0, 0], [1, 0], [2, 1]]
    assert rows[0]["links"][1]["score"] == .13
    assert rows[1]["pairs"] == []
    assert read_alignments(dst)["one"] == [(0, 0), (1, 0), (2, 1)]
    from ud_treebank_lab.corpus_refinement import _initial_matrix
    assert _initial_matrix(rows[0], 3, 2)[0][1] == .13


def test_giza_a3_annotated_source_null_links_and_wrapping(tmp_path):
    kw = sample(tmp_path)
    (tmp_path / "true.A3.final").write_text(
        "# Sentence pair (1) source length 3 target length 2 alignment score : 0.000001\n"
        "नवाब बनवलें\nNULL ({ }) नवाब ({ 1 }) ने ({ })\n बनाया ({ 2 })\n"
        "# Sentence pair (2) source length 2 target length 2 alignment score : 0.72\n"
        "राम गइलें\nNULL ({ }) राम ({ 1 }) गया ({ 2 })\n", encoding="utf-8"
    )
    dst = tmp_path / "out.jsonl"
    result = import_alignments(**kw, input_path=str(tmp_path / "true.A3.final"), output=str(dst))
    assert result["format"] == "a3"
    rows = list(jsonl(dst))
    assert rows[0]["pairs"] == [[0, 0], [2, 1]]
    assert "links" not in rows[0]  # sentence score is not link probability
    assert rows[0]["a3_sentence_score"] == 1e-6


def test_giza_a3_plain_source_inverse_alignment(tmp_path):
    kw = sample(tmp_path)
    (tmp_path / "reverse.A3.final").write_text(
        "# Sentence pair (1) source length 2 target length 3 alignment score : 0.1\n"
        "नवाब ने बनाया\nNULL ({ 2 }) नवाब ({ 1 }) बनवलें ({ 3 })\n"
        "# Sentence pair (2) source length 2 target length 2 alignment score : 0.9\n"
        "राम गया\nNULL ({ }) राम ({ 1 }) गइलें ({ 2 })\n", encoding="utf-8"
    )
    dst = tmp_path / "out.jsonl"
    import_alignments(**kw, input_path=str(tmp_path / "reverse.A3.final"), output=str(dst))
    converted = next(jsonl(dst))
    assert converted["pairs"] == [[0, 0], [2, 1]]
    assert converted["null_source"] == [1]  # Hindi ने had no overt target word


@pytest.mark.parametrize("text,expected", [
    ("0-0\n", "expected 2"),
    ("0-2\n\n", "outside"),
    ("-1-0\n\n", "invalid"),
    ("0-0\n0-0\n", None),
])
def test_alignment_checks_and_atomic_write(tmp_path, text, expected):
    kw = sample(tmp_path)
    (tmp_path / "links.txt").write_text(text)
    out = tmp_path / "out.jsonl"
    if expected:
        with pytest.raises(ValueError, match=expected):
            import_alignments(**kw, input_path=str(tmp_path / "links.txt"), output=str(out))
        assert not out.exists()
    else:
        import_alignments(**kw, input_path=str(tmp_path / "links.txt"), output=str(out))
        assert out.exists()
        original = out.read_bytes()
        with pytest.raises(FileExistsError):
            import_alignments(**kw, input_path=str(tmp_path / "links.txt"), output=str(out))
        assert out.read_bytes() == original


def test_giza_text_mismatch_rejected_and_override(tmp_path):
    kw = sample(tmp_path)
    (tmp_path / "bad.A3.final").write_text(
        "# Sentence pair (1) source length 3 target length 2 alignment score : 0.4\n"
        "गलत बनवलें\nNULL ({ }) नवाब ({ 1 }) ने ({ }) बनाया ({ 2 })\n"
        "# Sentence pair (2) source length 2 target length 2 alignment score : 0.1\n"
        "राम गइलें\nNULL ({ }) राम ({ 1 }) गया ({ 2 })\n", encoding="utf-8"
    )
    out = tmp_path / "out.jsonl"
    with pytest.raises(ValueError, match="cannot determine Hindi source side"):
        import_alignments(**kw, input_path=str(tmp_path / "bad.A3.final"), output=str(out))
    with pytest.raises(ValueError, match="text mismatch"):
        import_alignments(**kw, input_path=str(tmp_path / "bad.A3.final"), output=str(out),
                          a3_source_side="annotated")
    import_alignments(**kw, input_path=str(tmp_path / "bad.A3.final"), output=str(out),
                      a3_source_side="annotated", verify_forms=False)
    assert out.exists()


def test_apertium_sentence_lines_ambiguity_and_unknown(tmp_path):
    kw = sample(tmp_path)
    raw = ("^नवाब/नवाब<n><sg>/नवाब<np><sg>$ ^बनवलें/बनावल<vblex><past>$\n"
           "^राम/राम<np><sg>$ ^गइलें/*गइलें$\n")
    (tmp_path / "apertium.txt").write_text(raw, encoding="utf-8")
    dst = tmp_path / "morph.jsonl"
    result = import_apertium(targets=kw["targets"], input_path=str(tmp_path / "apertium.txt"),
                             output=str(dst))
    assert result["layout"] == "sentence-lines"
    rows = list(jsonl(dst))
    assert [len(a) for a in rows[0]["analyses"]] == [2, 1]
    assert rows[0]["analyses"][0][1]["upos"] == "PROPN"
    assert rows[0]["analyses"][0][1]["raw_tags"] == ["np", "sg"]
    assert rows[1]["analyses"][1] == []
    from ud_treebank_lab.alignment import read_targets
    loaded = read_apertium_candidates(str(dst), read_targets(kw["targets"]))
    assert len(loaded["one"][0]) == 2


def test_apertium_token_lines_gzip_escaped_and_superblank(tmp_path):
    kw = sample(tmp_path)
    raw = ("[<html>]^नवाब/नवाब<np><sg>$\n"
           "^बनवलें/बना\\/वल<vblex><past>$\n"
           "^राम/राम<np><sg>$\n"
           "^गइलें/जा<vblex><past>$\n")
    dest = tmp_path / "apertium.txt.gz"
    import gzip
    with gzip.open(dest, "wt", encoding="utf-8") as f:
        f.write(raw)
    out = tmp_path / "out.jsonl"
    r = import_apertium(targets=kw["targets"], input_path=str(dest),
                        layout="auto", output=str(out))
    assert r["layout"] == "token-lines"
    assert next(jsonl(out))["analyses"][1][0]["lemma"] == "बना/वल"


def test_apertium_word_list_and_missing_tokens(tmp_path):
    kw = sample(tmp_path)
    (tmp_path / "wordlist.txt").write_text(
        "^राम/राम<np>$\n^नवाब/नवाब<n>$\n^बनवलें/बनावल<vblex>$\n", encoding="utf-8")
    dst = tmp_path / "out.jsonl"
    result = import_apertium(targets=kw["targets"], input_path=str(tmp_path / "wordlist.txt"),
                             output=str(dst), layout="word-list")
    assert result["missing_forms"] == 1
    rows = list(jsonl(dst))
    assert rows[1]["analyses"] == [[{
        "lemma": "राम", "upos": "PROPN", "feats": {}, "raw_tags": ["np"],
        "raw_analysis": "राम<np>", "resource": "apertium"}], []]


def test_apertium_surface_mismatch_keeps_previous_output(tmp_path):
    kw = sample(tmp_path)
    (tmp_path / "bad.txt").write_text(
        "^नवाब/नवाब<n>$ ^X/जा<vblex>$\n^राम/राम<np>$ ^गइलें/जा<vblex>$\n", encoding="utf-8")
    out = tmp_path / "out.jsonl"
    with pytest.raises(ValueError, match="could not be reconciled"):
        import_apertium(targets=kw["targets"], input_path=str(tmp_path / "bad.txt"),
                        output=str(out))
    assert not out.exists()


def test_independent_config_cli_and_refinement(tmp_path):
    kw = sample(tmp_path)
    (tmp_path / "links.A3.final").write_text("0-0 2-1\n0-0 1-1\n")
    (tmp_path / "stream.txt").write_text(
        "^नवाब/नवाब<n>$ ^बनवलें/बनावल<vblex>$\n"
        "^राम/राम<np>$ ^गइलें/जाना<vblex>$\n", encoding="utf-8")
    cfg = {"source_ud": "source.conllu", "targets": "targets.jsonl",
           "alignment": {"input": "links.A3.final", "output": "align.jsonl"},
           "apertium": {"input": "stream.txt", "output": "morph.jsonl"}}
    path = tmp_path / "import.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True))
    result = import_resources(path)
    assert result["alignment"]["sentences"] == 2
    assert result["apertium"]["sentences"] == 2
    assert main(["import-resources", "--config", str(path)]) == 1  # no implicit overwrite
    corpus = {"source_ud": "source.conllu", "targets": "targets.jsonl",
              "output_dir": "corpus", "resources": {
                  "alignment": {"mode": "supplied", "path": "align.jsonl"},
                  "apertium": {"mode": "supplied", "path": "morph.jsonl"}}}
    (tmp_path / "corpus.yaml").write_text(yaml.safe_dump(corpus))
    out = run_corpus_refinement(tmp_path / "corpus.yaml")
    assert out["status"] == "complete"
    assert (tmp_path / "corpus" / "alignments.refined.jsonl").is_file()
    assert list(jsonl(tmp_path / "morph.jsonl"))[0]["analyses"][0]


def test_config_only_apertium_and_no_source_file(tmp_path):
    kw = sample(tmp_path)
    (tmp_path / "ap.txt").write_text(
        "^नवाब/नवाब<n>$ ^बनवलें/बनावल<vblex>$\n"
        "^राम/राम<np>$ ^गइलें/जाना<vblex>$\n", encoding="utf-8")
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "targets": "targets.jsonl", "apertium": {"input": "ap.txt", "output": "morph.jsonl"}}))
    assert "apertium" in import_resources(tmp_path / "config.yaml")
    assert "alignment" not in import_resources(tmp_path / "config.yaml", overwrite=True)


def test_importer_refuses_to_overwrite_original_input(tmp_path):
    kw = sample(tmp_path)
    f = tmp_path / "links.txt"
    f.write_text("0-0\n0-0\n")
    with pytest.raises(ValueError, match="input file"):
        import_alignments(**kw, input_path=str(f), output=str(f), overwrite=True)
