from pathlib import Path
import pytest

from ud_treebank_lab.conllu import Sentence, Word, read_conllu, validate_tree, parse_conllu_block

EX = Path(__file__).resolve().parent.parent / "examples"


def test_roundtrip_and_tree():
    sents = list(read_conllu(EX / "hindi_source.conllu"))
    assert len(sents) == 3
    assert all(validate_tree(s) == [] for s in sents)
    parsed = parse_conllu_block(sents[0].to_conllu())
    assert parsed.sent_id == sents[0].sent_id
    assert [x.to_line() for x in parsed.words] == [x.to_line() for x in sents[0].words]


def test_bad_tree_cycles_and_roots():
    s = Sentence([Word(1, "a", head=2, deprel="nsubj"),
                  Word(2, "b", head=1, deprel="obj")], "bad")
    assert any("cycle" in e for e in validate_tree(s))
    assert any("roots" in e for e in validate_tree(s))


def test_mwt_and_empty_nodes_ignored(tmp_path):
    f = tmp_path / "in.conllu"
    f.write_text("# sent_id = x\n1-2\tcan't\t_\t_\t_\t_\t_\t_\t_\t_\n"
                 "1\tca\t_\tAUX\t_\t_\t2\taux\t_\t_\n"
                 "2\tn't\t_\tVERB\t_\t_\t0\troot\t_\t_\n"
                 "2.1\tghost\t_\tX\t_\t_\t_\t_\t_\t_\n\n", encoding="utf-8")
    words = next(read_conllu(f)).words
    assert [w.form for w in words] == ["ca", "n't"]


def test_non_contiguous_ids_rejected(tmp_path):
    f = tmp_path / "bad.conllu"
    f.write_text("1\ta\t_\tX\t_\t_\t0\troot\t_\t_\n"
                 "3\tb\t_\tX\t_\t_\t1\tdep\t_\t_\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Non-contiguous"):
        list(read_conllu(f))
