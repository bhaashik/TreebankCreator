"""Availability-aware, non-destructive preparation of alignment/morphology inputs.

Side effects are opt-in. This stage can run independently of neural training or
as part of ``refine-corpus``. Conversion output is managed separately from any
supplied raw/JSONL file. Installation requires explicit setup.on_missing=install.
"""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from .alignment import run_alignment
from .corpus_config import read_corpus_config
from .corpus_morph import generate_apertium_lattice
from .resource_import import import_alignments, import_apertium
from .utils import fingerprint, write_json


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _native_jsonl(path: Path, fmt: str, name: str) -> bool:
    """Recognize per-sentence JSONL without inferring structure from filenames."""
    if fmt not in {"auto", "jsonl"}:
        return False
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as inp:
        for line in inp:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except (ValueError, TypeError):
                if fmt == "jsonl":
                    raise ValueError(f"{name}: declared JSONL but first record is not JSON")
                return False
            valid = isinstance(obj, dict) and "id" in obj and (
                any(k in obj for k in ("pairs", "links", "matrix"))
                if name == "alignment" else "analyses" in obj)
            if not valid and fmt == "jsonl":
                raise ValueError(f"{name}: declared JSONL but required sentence fields are missing")
            if valid and path.suffix.lower() == ".gz":
                raise ValueError(f"{name}: compressed JSONL must first be decompressed to .jsonl")
            return valid
    raise ValueError(f"Empty resource file: {path}")


def _missing_setup(name: str, spec: dict) -> list[str]:
    setup = spec.get("setup") or {}
    checks = []
    module = setup.get("check_module")
    if name == "alignment" and spec.get("backend") == "auto" and not setup.get("check_module") and not setup.get("check_executable"):
        simalign_found = importlib.util.find_spec("simalign") is not None
        from .awesome_adapter import _executable
        try:
            _executable(spec.get("binary", "awesome-align"))
            awesome_found = True
        except RuntimeError:
            awesome_found = False
        return [] if simalign_found or awesome_found else ["SimAlign module or AWESOME-align CLI"]
    if module is None and name == "alignment" and spec.get("backend", "simalign") == "simalign":
        module = "simalign"
    if module:
        try:
            present = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError, AttributeError):
            present = False
        if not present:
            checks.append(f"Python module {module!r}")
    binary = setup.get("check_executable")
    if binary is None and name == "alignment" and spec.get("backend", "simalign") == "awesome":
        binary = spec.get("binary", "awesome-align")
    if binary is None and name == "apertium":
        binary = spec.get("binary", "lt-proc")
    if binary and not (shutil.which(binary) or (Path(binary).is_file() and
                                                 os.access(binary, os.X_OK))):
        checks.append(f"executable {binary!r}")
    if name == "apertium" and not (spec.get("analyzer_bin") and
                                     Path(spec["analyzer_bin"]).is_file()):
        checks.append(f"Apertium analyzer file {spec.get('analyzer_bin')!r}")
    return checks


def _ensure_setup(name: str, spec: dict, *, dry_run: bool) -> dict:
    missing = _missing_setup(name, spec)
    if not missing:
        return {"status": "available", "missing": []}
    setup = spec.get("setup") or {}
    behavior = setup.get("on_missing", "error")
    if dry_run:
        return {"status": ("would-install" if behavior == "install" else
                           "would-skip" if behavior == "skip" else "missing"),
                "missing": missing}
    if behavior == "skip":
        return {"status": "skipped", "missing": missing}
    if behavior != "install":
        raise RuntimeError(f"Cannot generate {name}: missing {', '.join(missing)}; "
                           "install/configure the tool or set setup.on_missing=install "
                           "with an explicit install_command")
    # No shell, no automatic pip/apt. The operator supplies the *exact* command.
    subprocess.run(setup["install_command"], shell=False, check=True)
    remaining = _missing_setup(name, spec)
    if remaining:
        raise RuntimeError(f"{name} setup command completed but still missing: {remaining}")
    return {"status": "installed", "missing": []}


def _resource_action(name: str, spec: dict) -> tuple[str, Path | None]:
    mode = spec["mode"]
    if mode == "skip":
        return "skip", None
    path = Path(spec["path"]) if spec.get("path") else None
    exists = bool(path and path.is_file())
    if mode == "supplied":
        if not exists:
            raise FileNotFoundError(f"{name}: supplied resource not found: {path}")
        return "use", path
    if mode == "generate":
        return "generate", None
    if mode != "auto":
        raise ValueError(f"Unknown resource mode: {mode}")
    available = spec["available"]
    if available is True and not exists:
        raise FileNotFoundError(f"{name}.available=true but file is absent: {path}")
    if available is not False and exists:
        return "use", path
    policy = spec["if_missing"]
    if policy == "skip":
        return "skip", None
    if policy == "generate":
        return "generate", None
    raise FileNotFoundError(f"{name}: input is missing: {path}. Set if_missing: generate "
                            "or skip (or provide a valid input path)")


def prepare_corpus_resources(config: str | Path | dict, *, force: bool = False,
                             dry_run: bool = False) -> dict:
    """Resolve supplied JSONL/raw files or explicitly generate missing resources.

    Cached conversions/generations are reused even if their external tools are
    not installed anymore. Unmanaged output files are NEVER overwritten.
    """
    cfg = config if isinstance(config, dict) else read_corpus_config(config)
    dest = Path(cfg["output_dir"]) / "resource_inputs"
    manifest_path = dest / "manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    old = previous.get("resources", {})
    selected: dict[str, str | None] = {}
    stages = {}
    new_records = {}
    for name in ("alignment", "apertium", "unimorph"):
        spec = cfg["resources"][name]
        action, source = _resource_action(name, spec)
        if action == "skip":
            selected[name] = None
            stages[name] = {"status": "skipped", "reason": "mode/if_missing"}
            new_records[name] = {"action": "skip"}
            continue
        if name == "unimorph":
            if action == "generate":
                raise ValueError("UniMorph generation is not implemented")
            selected[name] = str(source)
            stages[name] = {"status": "supplied", "path": str(source)}
            new_records[name] = {"action": "supplied", "path": str(source)}
            continue
        native = action == "use" and _native_jsonl(source, spec["format"], name)
        if action == "use" and native:
            selected[name] = str(source)
            stages[name] = {"status": "supplied-jsonl", "path": str(source)}
            new_records[name] = {"action": "supplied", "path": str(source)}
            continue
        if action == "use":
            suffix = "imported"
        else:
            suffix = "generated"
        output = Path(spec.get("output") or dest / f"{name}.{suffix}.jsonl")
        protected = {Path(cfg["source_ud"]), Path(cfg["targets"])}
        for key in ("path", "lexicon", "analyzer_bin", "probabilities"):
            if spec.get(key):
                protected.add(Path(spec[key]))
        if output in protected:
            raise ValueError(f"Managed output for {name} cannot replace an input file: {output}")
        needed = [Path(cfg["source_ud"]), Path(cfg["targets"])]
        if source is not None:
            needed.append(source)
        for key in ("lexicon", "probabilities", "analyzer_bin"):
            if spec.get(key) and Path(spec[key]).is_file():
                needed.append(Path(spec[key]))
        sig = fingerprint({"resource-prep": 1, "name": name, "action": action, "spec": spec}, *needed)
        prev = old.get(name, {})
        reused = (output.is_file() and prev.get("signature") == sig and
                  prev.get("path") == str(output) and prev.get("sha256") == _sha(output))
        if reused and not force:
            status = "reused-conversion" if action == "use" else "reused-generation"
            stages[name] = {"status": status, "path": str(output)}
        elif dry_run:
            stages[name] = {"status": "would-import" if action == "use" else "would-generate",
                            "path": str(output)}
        else:
            if output.is_file() and not (prev.get("path") == str(output) and prev.get("signature")):
                raise FileExistsError(f"Refusing to overwrite unmanaged resource output: {output}")
            if action == "use":
                if name == "alignment":
                    import_alignments(source_ud=cfg["source_ud"], targets=cfg["targets"],
                                      input_path=str(source), output=str(output),
                                      format=spec["format"] if spec["format"] != "jsonl" else "auto",
                                      probabilities=spec.get("probabilities"),
                                      sentence_order=spec.get("sentence_order", "source_ud"),
                                      a3_source_side=spec.get("a3_source_side", "auto"),
                                      verify_forms=spec.get("verify_forms", True),
                                      overwrite=output.exists())
                else:
                    import_apertium(targets=cfg["targets"], input_path=str(source),
                                    output=str(output), layout=spec.get("layout", "auto"),
                                    verify_forms=spec.get("verify_forms", True),
                                    overwrite=output.exists())
                stages[name] = {"status": "imported", "path": str(output), "source": str(source)}
            else:
                setup = _ensure_setup(name, spec, dry_run=False)
                if setup["status"] == "skipped":
                    selected[name] = None
                    stages[name] = {"status": "skipped-no-setup", "missing": setup["missing"]}
                    new_records[name] = {"action": "skip-no-setup"}
                    continue
                output.parent.mkdir(parents=True, exist_ok=True)
                # Generated writers can write incrementally. Always stage in a
                # temporary path, then publish atomically on successful completion.
                fd, staging = tempfile.mkstemp(prefix=f".{name}-", suffix=".jsonl",
                                               dir=output.parent)
                os.close(fd)
                try:
                    if name == "alignment":
                        lex = json.loads(Path(spec["lexicon"]).read_text(encoding="utf-8")) if spec.get("lexicon") else None
                        run_alignment(cfg["source_ud"], cfg["targets"], staging,
                                      backend=spec.get("backend", "simalign"),
                                      model=spec.get("model", "xlmr"),
                                      method=spec.get("method", "itermax"),
                                      device=spec.get("device", "auto"), lexicon=lex,
                                      binary=spec.get("binary", "awesome-align"))
                    else:
                        generate_apertium_lattice(cfg["targets"], staging,
                                                 spec["analyzer_bin"],
                                                 binary=spec.get("binary", "lt-proc"),
                                                 batch_size=spec.get("batch_size", 1000))
                    os.replace(staging, output)
                finally:
                    Path(staging).unlink(missing_ok=True)
                # An opt-in setup command may create the Apertium analyzer.
                # Incorporate its post-install bytes into the cache signature.
                if spec.get("analyzer_bin") and Path(spec["analyzer_bin"]).is_file() and Path(spec["analyzer_bin"]) not in needed:
                    needed.append(Path(spec["analyzer_bin"]))
                    sig = fingerprint({"resource-prep": 1, "name": name,
                                       "action": action, "spec": spec}, *needed)
                stages[name] = {"status": "generated", "path": str(output),
                                "setup": setup["status"]}
        if dry_run and action == "generate" and not reused:
            stages[name]["setup"] = _ensure_setup(name, spec, dry_run=True)
        selected[name] = str(output)
        new_records[name] = {"action": action, "signature": sig,
                             "path": str(output),
                             "sha256": _sha(output) if output.is_file() and not dry_run else None}
    if not dry_run:
        write_json(manifest_path, {"resources": new_records, "selected_inputs": selected})
    return {"status": "dry-run" if dry_run else "complete", "selected_inputs": selected,
            "stages": stages, "manifest": str(manifest_path)}
