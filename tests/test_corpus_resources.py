"""No-network lifecycle tests for supplied/generated/installed resource decisions."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import yaml

from ud_treebank_lab.cli import main
from ud_treebank_lab.corpus_config import read_corpus_config
from ud_treebank_lab.corpus_refinement import run_corpus_refinement
from ud_treebank_lab.corpus_resources import prepare_corpus_resources
from ud_treebank_lab.utils import jsonl


def _input(tmp: Path) -> Path:
    (tmp / "hi.conllu").write_text(
        "# sent_id = x\n1\tराम\tराम\tPROPN\t_\t_\t2\tnsubj\t_\t_\n"
        "2\tगया\tजाना\tVERB\t_\t_\t0\troot\t_\t_\n\n",
        encoding="utf-8")
    (tmp / "target.jsonl").write_text(
        '{"id":"x","tokens":["राम","गइलें"]}\n', encoding="utf-8")
    (tmp / "original.A3.final").write_text("0-0 1-1\n", encoding="utf-8")
    (tmp / "raw.apertium").write_text(
        "^राम/राम<np><sg>$ ^गइलें/जा<vblex><past>$\n", encoding="utf-8")
    return tmp / "corpus.yaml"


def _cfg(file: Path, resources: dict) -> Path:
    file.write_text(yaml.safe_dump({"source_ud": "hi.conllu", "targets": "target.jsonl",
                                   "output_dir": "work", "resources": resources,
                                   "refinement": {"iterations": 1}}, allow_unicode=True))
    return file


def test_supplied_raw_files_auto_convert_and_cache(tmp_path, monkeypatch):
    f = _input(tmp_path)
    _cfg(f, {"alignment": {"mode": "auto", "available": True,
                           "path": "original.A3.final", "format": "auto",
                           "if_missing": "generate"},
             "apertium": {"mode": "auto", "path": "raw.apertium", "format": "raw",
                          "if_missing": "generate"}})
    from ud_treebank_lab import corpus_resources as rr
    def unexpected(*args, **kwargs):
        raise AssertionError("No external tool may be run for supplied inputs")
    monkeypatch.setattr(rr, "run_alignment", unexpected)
    monkeypatch.setattr(rr, "generate_apertium_lattice", unexpected)
    originals = {k: (tmp_path / k).read_bytes() for k in ("original.A3.final", "raw.apertium")}
    dry = prepare_corpus_resources(f, dry_run=True)
    assert dry["stages"]["alignment"]["status"] == "would-import"
    assert dry["stages"]["apertium"]["status"] == "would-import"
    assert not (tmp_path / "work").exists()
    done = prepare_corpus_resources(f)
    assert done["stages"]["alignment"]["status"] == "imported"
    assert done["stages"]["apertium"]["status"] == "imported"
    assert list(jsonl(done["selected_inputs"]["alignment"]))[0]["pairs"] == [[0, 0], [1, 1]]
    assert list(jsonl(done["selected_inputs"]["apertium"]))[0]["analyses"][0][0]["upos"] == "PROPN"
    assert all((tmp_path / k).read_bytes() == b for k, b in originals.items())
    reused = prepare_corpus_resources(f)
    assert reused["stages"]["alignment"]["status"] == "reused-conversion"
    assert reused["stages"]["apertium"]["status"] == "reused-conversion"
    result = run_corpus_refinement(f)
    assert result["status"] == "complete"
    assert run_corpus_refinement(f)["status"] == "already-complete"


def test_jsonl_supplied_is_not_reconverted_or_replaced(tmp_path, monkeypatch):
    f = _input(tmp_path)
    existing = tmp_path / "existing.jsonl"
    existing.write_text('{"id":"x","pairs":[[0,0],[1,1]]}\n')
    _cfg(f, {"alignment": {"mode": "supplied", "path": "existing.jsonl"}})
    from ud_treebank_lab import corpus_resources as rr
    monkeypatch.setattr(rr, "import_alignments", lambda **kw: (_ for _ in ()).throw(AssertionError()))
    before = existing.read_bytes()
    result = prepare_corpus_resources(f)
    assert result["selected_inputs"]["alignment"] == str(existing)
    assert result["stages"]["alignment"]["status"] == "supplied-jsonl"
    assert existing.read_bytes() == before


def test_conditional_generate_uses_only_missing_resources(tmp_path, monkeypatch):
    f = _input(tmp_path)
    _cfg(f, {"alignment": {"mode": "auto", "path": "missing.alignment",
                           "if_missing": "generate", "backend": "identity"},
             "apertium": {"mode": "auto", "path": "raw.apertium",
                          "if_missing": "generate"}})
    from ud_treebank_lab import corpus_resources as rr
    def bad(*args, **kw):
        raise AssertionError("Apertium should be reused rather than generated")
    monkeypatch.setattr(rr, "generate_apertium_lattice", bad)
    result = prepare_corpus_resources(f)
    assert result["stages"]["alignment"]["status"] == "generated"
    assert result["stages"]["alignment"]["setup"] == "available"
    assert result["stages"]["apertium"]["status"] == "imported"
    assert list(jsonl(result["selected_inputs"]["alignment"]))[0]["pairs"] == [[0,0],[1,1]]
    again = prepare_corpus_resources(f)
    assert again["stages"]["alignment"]["status"] == "reused-generation"


def test_setup_check_no_implicit_install_and_missing_policy(tmp_path, monkeypatch):
    f = _input(tmp_path)
    _cfg(f, {"apertium": {"mode": "auto", "if_missing": "generate",
                          "analyzer_bin": "missing.analyzer", "binary": "missing-lt-proc"}})
    before = prepare_corpus_resources(f, dry_run=True)
    assert before["stages"]["apertium"]["setup"]["status"] == "missing"
    assert not (tmp_path / "work").exists()
    with pytest.raises(RuntimeError, match="Cannot generate apertium"):
        prepare_corpus_resources(f)
    assert not (tmp_path / "work/resource_inputs/apertium.generated.jsonl").exists()
    cfg = yaml.safe_load(f.read_text())
    cfg["resources"]["apertium"]["setup"] = {"on_missing": "skip"}
    f.write_text(yaml.safe_dump(cfg))
    skipped = prepare_corpus_resources(f)
    assert skipped["selected_inputs"]["apertium"] is None
    assert skipped["stages"]["apertium"]["status"] == "skipped-no-setup"
    assert run_corpus_refinement(f)["status"] == "complete"


def test_explicit_install_command_only_when_missing(tmp_path):
    f = _input(tmp_path)
    ltproc = tmp_path / "bin" / "lt-proc"
    analyzer = tmp_path / "made.analyzer"
    # Python creates an executable mock analyzer; no pip, apt or network access.
    creator = ("from pathlib import Path; import sys; "
               "p=Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True); "
               "p.write_text('#!'+sys.executable+'\\nimport sys\\nfor line in sys.stdin:\\n "
               " w=line.strip(); print(\"^\"+w+\"/\"+w+\"<n>$\") if w else print()\\n'); "
               "p.chmod(0o755); Path(sys.argv[2]).write_text('analyzer')")
    cfg = {"apertium": {"mode": "auto", "if_missing": "generate",
                        "analyzer_bin": "made.analyzer", "binary": str(ltproc),
                        "setup": {"on_missing": "install",
                                  "install_command": [sys.executable, "-c", creator,
                                                      str(ltproc), str(analyzer)]}}}
    _cfg(f, cfg)
    dry = prepare_corpus_resources(f, dry_run=True)
    assert dry["stages"]["apertium"]["setup"]["status"] == "would-install"
    assert not ltproc.exists()
    done = prepare_corpus_resources(f)
    assert done["stages"]["apertium"]["setup"] == "installed"
    assert ltproc.exists() and analyzer.exists()
    assert next(jsonl(done["selected_inputs"]["apertium"]))["analyses"][0][0]["lemma"] == "राम"
    # Even if executable disappears after generation, cache is reusable.
    ltproc.unlink()
    again = prepare_corpus_resources(f)
    assert again["stages"]["apertium"]["status"] == "reused-generation"


def test_existing_setup_never_executes_install_command(tmp_path):
    f = _input(tmp_path)
    marker = tmp_path / "should-not-run"
    _cfg(f, {"alignment": {"mode": "generate", "backend": "identity",
                           "setup": {"on_missing": "install",
                                     "install_command": [sys.executable, "-c",
                                                         f"open({str(marker)!r},'w').write('BAD')"]}}})
    result = prepare_corpus_resources(f)
    assert result["stages"]["alignment"]["setup"] == "available"
    assert not marker.exists()


def test_invalid_modes_and_existing_output_protection(tmp_path):
    f = _input(tmp_path)
    _cfg(f, {"alignment": {"mode": "supplied", "path": "missing.A3.final"}})
    with pytest.raises(FileNotFoundError):
        read_corpus_config(f)
    _cfg(f, {"alignment": {"mode": "auto", "path": "missing.A3.final",
                           "available": True, "if_missing": "generate"}})
    with pytest.raises(FileNotFoundError, match="available=true"):
        prepare_corpus_resources(f)
    _cfg(f, {"alignment": {"mode": "auto", "path": "original.A3.final",
                           "output": "original.A3.final"}})
    with pytest.raises(ValueError, match="must differ"):
        read_corpus_config(f)
    _cfg(f, {"alignment": {"mode": "auto", "path": "original.A3.final",
                           "output": "work/resource_inputs/alignment.imported.jsonl"}})
    out = tmp_path / "work/resource_inputs/alignment.imported.jsonl"
    out.parent.mkdir(parents=True)
    out.write_text("user-owned")
    with pytest.raises(FileExistsError, match="unmanaged"):
        prepare_corpus_resources(f)
    assert out.read_text() == "user-owned"


def test_input_change_invalidates_managed_import_and_cli(tmp_path):
    f = _input(tmp_path)
    _cfg(f, {"alignment": {"path": "original.A3.final", "mode": "supplied"}})
    assert main(["prepare-resources", "--config", str(f), "--dry-run"]) == 0
    assert not (tmp_path / "work").exists()
    first = prepare_corpus_resources(f)
    target_path = Path(first["selected_inputs"]["alignment"])
    (tmp_path / "original.A3.final").write_text("0-0\n")
    again = prepare_corpus_resources(f)
    assert again["stages"]["alignment"]["status"] == "imported"
    assert next(jsonl(target_path))["pairs"] == [[0, 0]]
