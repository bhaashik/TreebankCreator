from pathlib import Path

import pytest

from ud_treebank_lab.alignment import Aligner, read_targets, tokenize
from ud_treebank_lab.conllu import Sentence, Word, read_conllu, validate_tree
from ud_treebank_lab.morphology import parse_apertium
from ud_treebank_lab.projection import project_sentence, run_projection

EX = Path(__file__).resolve().parent.parent / "examples"


def test_known_alignments_valid_and_accuracy():
    from ud_treebank_lab.alignment import read_alignments
    sources = list(read_conllu(EX / "hindi_source.conllu"))
    targets = read_targets(EX / "marathi_targets.jsonl")
    alignments = read_alignments(EX / "manual_alignments.jsonl")
    for source in sources:
        result, meta = project_sentence(source, targets[source.sent_id]["tokens"],
                                        alignments[source.sent_id])
        assert not validate_tree(result)
        assert sum(w.head == 0 for w in result.words) == 1
        assert 0 <= meta["quality"] <= 1
        assert all(0 <= arc["confidence"] <= 1 for arc in meta["arc_provenance"])


def test_zero_and_many_to_many_alignments():
    source = next(read_conllu(EX / "hindi_source.conllu"))
    for pairs in ([], [(0, 0), (0, 1), (2, 0), (3, 2)]):
        result, metrics = project_sentence(source, ["नया", "राम", "किया", "."], pairs)
        assert validate_tree(result) == []
        assert 0 <= metrics["alignment_coverage"] <= 1
    assert metrics["quality"] > 0


def test_morph_override_without_transferring_hindi_inflections():
    source = next(read_conllu(EX / "hindi_source.conllu"))
    tgt = ["राम", "खाता"]
    result, _ = project_sentence(source, tgt, [(0, 0), (3, 1)],
                                  morph=[{"lemma": "राम", "upos": "NOUN", "feats": {"Number": "Sing"}}, {}])
    assert result.words[0].upos == "NOUN"
    assert result.words[0].feats == "Number=Sing"
    assert result.words[1].feats == "_"


def test_out_of_range_raises():
    src = next(read_conllu(EX / "hindi_source.conllu"))
    with pytest.raises(ValueError, match="Out-of-range"):
        project_sentence(src, ["x"], [(99, 0)])


def test_projection_resume(tmp_path):
    output = tmp_path / "silver.conllu"
    meta = tmp_path / "meta.jsonl"
    values1 = run_projection(EX / "hindi_source.conllu", EX / "marathi_targets.jsonl",
                             EX / "manual_alignments.jsonl", output, meta)
    values2 = run_projection(EX / "hindi_source.conllu", EX / "marathi_targets.jsonl",
                             EX / "manual_alignments.jsonl", output, meta)
    assert values1["processed_this_call"] == 3
    assert values2["processed_this_call"] == 0
    assert len(list(read_conllu(output))) == 3
    with output.with_name(output.name + ".stage.jsonl").open("a", encoding="utf-8") as f:
        f.write('{"broken":')
    values3 = run_projection(EX / "hindi_source.conllu", EX / "marathi_targets.jsonl",
                             EX / "manual_alignments.jsonl", output, meta)
    assert values3["total"] == 3


def test_simple_aligners_and_tokenizer():
    assert tokenize("राम।  Hello, world!") == ["राम", "।", "Hello", ",", "world", "!"]
    assert Aligner("identity").align(["A", "B"], ["X", "Y"]) == [(0, 0), (1, 1)]
    assert Aligner("lexicon", lexicon={"book": ["पुस्तक"]}).align(["book"], ["पुस्तक"]) == [(0, 0)]


def test_apertium_analysis_ambiguity():
    assert parse_apertium("^cats/cat<n><pl>$")["upos"] == "NOUN"
    assert parse_apertium("^cats/cat<n><pl>$")["feats"] == {"Number": "Plur"}
    assert parse_apertium("^x/a<n>/b<adj>$").get("upos") is None
    assert parse_apertium("^x/*x$") == {}
