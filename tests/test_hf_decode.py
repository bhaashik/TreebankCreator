"""Graph-decoding tests do not load model checkpoints or transformers."""
import numpy as np
from ud_treebank_lab.conllu import Sentence, Word, validate_tree
from ud_treebank_lab.hf_parser import _decode


def test_decoder_has_one_root_and_no_cycles():
    # Columns are root (0) and word positions (1..N). Even cyclic local
    # predictions must decode to a single, valid UD dependency tree.
    logits = np.array([
        [0.1, -10.0, 9.0, 0.1, 0.1],
        [0.2, 9.0, -10.0, 0.1, 0.1],
        [5.0, 0.1, 0.1, -10.0, 0.1],
        [0.5, 0.2, 0.2, 9.0, -10.0],
    ])
    heads = _decode(logits)
    assert len(heads) == 4
    assert heads.count(0) == 1
    root = heads.index(0) + 1
    sent = Sentence([
        Word(id=i+1, form=f"t{i+1}", head=h, deprel="root" if h == 0 else "dep")
        for i, h in enumerate(heads)
    ], "test")
    assert not validate_tree(sent)


def test_one_word_decoder():
    assert _decode(np.array([[2.0, -10.0]])) == [0]
