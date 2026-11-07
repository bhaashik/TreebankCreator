"""No external model downloads: verify the complete bilingual train/resume/predict path."""
import json
from pathlib import Path

import pytest
import torch
import yaml

from ud_treebank_lab.bilingual_cache import cache_signature, read_embeddings
from ud_treebank_lab.bilingual_data import soft_alignment, soft_structural_targets, Pair
from ud_treebank_lab.bilingual_model import ParallelParser, weak_loss
from ud_treebank_lab.bilingual_training import train_bilingual, predict_bilingual, _pairs, _source_rows
from ud_treebank_lab.bilingual_launcher import launch
from ud_treebank_lab.conllu import Word, Sentence, read_conllu, validate_tree
from ud_treebank_lab.utils import write_json


EXAMPLES = Path(__file__).parents[1] / "examples"


def test_soft_alignment_uncertainty():
    a, rel = soft_alignment({"pairs": [[0, 0], [1, 0], [2, 2]]}, 3, 4)
    assert a.shape == (4, 3)
    assert a[0, 0] == pytest.approx(0.5)
    assert 0 < rel[0] < rel[2] < 1
    assert rel[1] == rel[3] == 0
    with pytest.raises(ValueError):
        soft_alignment({"pairs": [[3, 0]]}, 3, 4)
    weighted, confidence = soft_alignment({"links": [{"src": 0, "tgt": 1, "score": .9},
                                                      {"src": 1, "tgt": 1, "score": .1}]}, 2, 2)
    assert weighted[1, 0] == pytest.approx(.9)
    assert confidence[0] == 0


def test_soft_structural_target_with_missing_links():
    sent = Sentence([Word(1, "x", head=2, deprel="nsubj"),
                     Word(2, "y", head=0, deprel="root")], "a")
    align, conf = soft_alignment({"pairs": [[0, 0], [1, 1]]}, 2, 3)
    p = Pair("a", sent, ["p", "q", "missing"], align, conf)
    q, arcs, weights = soft_structural_targets(p, {"dep": 0, "nsubj": 1, "root": 2})
    assert q.shape == (3, 4, 3)
    assert arcs[0, 2] == pytest.approx(1)
    assert arcs[1, 0] == pytest.approx(1)
    assert weights[0] > 0
    assert weights[2] == 0
    assert q[2].sum() == 0


def test_parallel_model_gradients_and_masks():
    torch.manual_seed(1)
    model = ParallelParser(12, 3, hidden=16, layers=2, alignment_dropout=0)
    src = torch.randn(2, 3, 12)
    tgt = torch.randn(2, 4, 12)
    sh = torch.tensor([[2, 0, 2], [0, 1, 0]])
    sm = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    tm = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    a = torch.full((2, 4, 3), 1/3)
    r = torch.full((2, 4), .55)
    arcs, rels, attn = model(src, tgt, sh, sm, tm, a, r)
    assert arcs.shape == (2, 4, 5)
    assert rels.shape == (2, 4, 5, 3)
    assert attn.shape == (2, 4, 3)
    assert arcs[0, 0, 1] < -9999
    assert arcs[1, 0, 3] < -9999  # masked target head
    q = torch.zeros(2, 4, 5, 3)
    q[0, 0, 2, 1] = 1
    q[1, 1, 0, 2] = 1
    loss, _ = weak_loss(arcs, rels, q, torch.tensor([[.55, 0, 0, 0], [0, .5, 0, 0]]))
    assert torch.isfinite(loss)
    loss.backward()
    assert model.arc_U.grad.abs().sum() > 0
    assert model.label_U.grad.abs().sum() > 0
    assert model.graph_parent.weight.grad is not None
    assert model.x_queries[0].weight.grad is not None


def _fake_cache(directory, rows, model, subtokens=64, shard_size=2, width=12):
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "manifest.json", {
        "signature": cache_signature(rows, model, subtokens, shard_size),
        "model": model, "max_subtokens": subtokens, "shard_size": shard_size,
        "num_sentences": len(rows), "pooling": "first-subword-v1"})
    for start in range(0, len(rows), shard_size):
        block = {}
        for sid, tokens in rows[start:start+shard_size]:
            gen = torch.Generator().manual_seed(sum(map(ord, sid)) + len(tokens))
            block[sid] = torch.randn(len(tokens), width, generator=gen).half()
        torch.save(block, directory / f"part_{start//shard_size:05d}.pt")


def _demo_cfg(tmp_path, name="mr"):
    cfg = {"language": name, "source_ud": str(EXAMPLES / "hindi_source.conllu"),
           "targets": str(EXAMPLES / "marathi_targets.jsonl"),
           "alignments": str(EXAMPLES / "manual_alignments.jsonl"),
           "source_cache": str(tmp_path / "hindi_cache"),
           "target_cache": str(tmp_path / (name + "_cache")),
           "output_dir": str(tmp_path / (name + "_run")),
           "model": "offline-fake", "epochs": 2, "batch_size": 1,
           "accumulate": 1, "save_steps": 1, "hidden": 16, "layers": 1,
           "shard_size": 2, "max_subtokens": 64, "cache_batch_size": 2,
           "dropout": 0.0, "alignment_dropout": 0.0, "dev_fraction": .34,
           "device": "cpu", "max_words": 70}
    file = tmp_path / (name + ".yaml")
    file.write_text(yaml.safe_dump(cfg), encoding="utf8")
    from ud_treebank_lab.bilingual_config import read_config
    resolved = read_config(file)
    _fake_cache(Path(cfg["source_cache"]), _source_rows(resolved), "offline-fake")
    pairs = _pairs(resolved)
    _fake_cache(Path(cfg["target_cache"]), [(p.sid, p.tokens) for p in pairs], "offline-fake")
    return file


def test_cached_trainer_resume_and_export(tmp_path):
    cfg = _demo_cfg(tmp_path)
    first = train_bilingual(str(cfg), device="cpu", max_steps=1, auto_cache=False)
    assert first["status"] == "paused"
    assert (tmp_path / "mr_run" / "last.pt").exists()
    result = train_bilingual(str(cfg), device="cpu", auto_cache=False)
    assert result["status"] == "complete"
    assert result["model_selection"] == "weak_projection_dev_loss"
    assert train_bilingual(str(cfg), device="cpu", auto_cache=False)["status"] == "already-complete"
    file = tmp_path / "pred.conllu"
    inferred = predict_bilingual(str(cfg), out_file=str(file), device="cpu")
    assert inferred["sentences"] == 3
    assert all(not validate_tree(s) for s in read_conllu(file))


def test_launcher_gpu_assignments_and_duplicate_outputs(tmp_path):
    a = _demo_cfg(tmp_path, "mr")
    b = _demo_cfg(tmp_path, "bn")
    plan = launch([str(a), str(b)], ["0", "1"], dry_run=True)
    assert plan["status"] == "dry-run"
    assert len(plan["source_preparation"]) == 1  # Hindi cached once
    assert len(plan["jobs"]) == 2
    assert len(plan["gpu_slots"]) == 2
    with pytest.raises(ValueError):
        launch([str(a), str(a)], ["0", "1"], dry_run=True)


def test_launcher_two_cpu_processes(tmp_path):
    # Same multi-process scheduler as Kaggle, independently verified without GPUs.
    a = _demo_cfg(tmp_path, "mr")
    b = _demo_cfg(tmp_path, "bn")
    result = launch([str(a), str(b)], ["cpu", "cpu"], precache_source=True, poll_seconds=.05)
    assert result["failed"] == 0, result
    assert len(result["results"]) == 2
    assert (tmp_path / "mr_run" / "done.json").exists()
    assert (tmp_path / "bn_run" / "done.json").exists()


def test_manifest_generation_and_subset(tmp_path):
    from ud_treebank_lab.bilingual_manifest import make_configs
    from ud_treebank_lab.bilingual_data import load_pairs
    manifest = tmp_path / "languages.jsonl"
    manifest.write_text("\n".join([
        json.dumps({"language": "mr", "targets": str(EXAMPLES / "marathi_targets.jsonl"),
                    "alignments": str(EXAMPLES / "manual_alignments.jsonl")}),
        json.dumps({"language": "bn", "targets": str(EXAMPLES / "marathi_targets.jsonl"),
                    "alignments": str(EXAMPLES / "manual_alignments.jsonl")}),
    ]) + "\n")
    result = make_configs(str(manifest), str(EXAMPLES / "hindi_source.conllu"),
                          str(tmp_path / "configs"), str(tmp_path / "caches"),
                          str(tmp_path / "runs"), max_pairs=2)
    assert result["languages"] == 2
    import yaml
    cfgs = [yaml.safe_load(Path(x).read_text()) for x in result["configs"]]
    assert cfgs[0]["source_cache"] == cfgs[1]["source_cache"]
    assert cfgs[0]["target_cache"] != cfgs[1]["target_cache"]
    pairs = load_pairs(EXAMPLES / "hindi_source.conllu", EXAMPLES / "marathi_targets.jsonl",
                       EXAMPLES / "manual_alignments.jsonl", limit=2)
    assert len(pairs) == 2


def test_fast_decoder_greedy_and_mst_fallback():
    import numpy as np
    from ud_treebank_lab.bilingual_training import _decode_tree
    simple = np.array([[0., -100., 3., 1.], [4., 0., -100., 2.], [1., 1., 3., -100.]])
    # 1->2, 2->root, 3->2: already a valid tree.
    assert _decode_tree(simple) == [2, 0, 2]
    # Two roots require single-root MST repair.
    multi = np.array([[3., -100., 1.], [3., 1., -100.]])
    heads = _decode_tree(multi)
    assert heads.count(0) == 1
    assert all(head != i+1 for i, head in enumerate(heads))


def test_real_cache_code_with_fake_encoder_and_resume(tmp_path, monkeypatch):
    """Exercise sharding/token selection/recovery without HF internet access."""
    import sys
    from types import SimpleNamespace
    from ud_treebank_lab.bilingual_cache import cache_embeddings, read_embeddings

    class FakeTokenized(dict):
        def __init__(self, rows):
            maxlen = max(len(row) for row in rows) + 2
            ids = torch.zeros((len(rows), maxlen), dtype=torch.long)
            mask = torch.zeros_like(ids)
            self.words = []
            for i, tokens in enumerate(rows):
                ids[i, :len(tokens) + 2] = torch.arange(len(tokens) + 2)
                mask[i, :len(tokens) + 2] = 1
                self.words.append([None, *range(len(tokens)), None] + [None] * (maxlen - len(tokens) - 2))
            super().__init__(input_ids=ids, attention_mask=mask)

        def word_ids(self, batch_index):
            return self.words[batch_index]

    class FakeTokenizer:
        is_fast = True
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()
        def __call__(self, words, **kwargs):
            return FakeTokenized(words)

    class FakeEncoder:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()
        def to(self, *args):
            return self
        def eval(self):
            return self
        def __call__(self, input_ids, attention_mask):
            return SimpleNamespace(last_hidden_state=input_ids.float().unsqueeze(-1).expand(-1,-1,8))

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=FakeTokenizer, AutoModel=FakeEncoder))
    rows = [(f"i{i}", ["a", "b"]) for i in range(5)]
    dest = tmp_path / "feature_cache"
    first = cache_embeddings(rows, dest, model="fake", device="cpu", batch_size=2,
                             shard_size=2, max_subtokens=20)
    assert first["new_shards"] == 3
    assert cache_embeddings(rows, dest, model="fake", device="cpu", batch_size=2,
                            shard_size=2, max_subtokens=20)["new_shards"] == 0
    (dest / "part_00001.pt").unlink()
    assert cache_embeddings(rows, dest, model="fake", device="cpu", batch_size=2,
                            shard_size=2, max_subtokens=20)["new_shards"] == 1
    cache = read_embeddings(rows, dest, model="fake", shard_size=2, max_subtokens=20)
    assert len(cache) == 5
    assert cache["i0"].shape == (2, 8)
    with pytest.raises(ValueError):
        cache_embeddings(rows, dest, model="changed", device="cpu", shard_size=2,
                         max_subtokens=20)
