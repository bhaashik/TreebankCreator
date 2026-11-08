"""Optional AWESOME-align CLI adapter, returning the project's JSONL schema.

Uses zero-based Pharaoh links. Does not import the old AWESOME-align Python
package into the application's Python process; it can run from a separate
executable/environment when supplied by the operator.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import sysconfig
import tempfile

from .alignment import read_targets
from .conllu import read_conllu
from .utils import append_jsonl, done_ids

_LINK = re.compile(r"^(\d+)-(\d+)$")


def _executable(binary: str) -> str:
    """Find a CLI on PATH, or in the selected Python scripts directory."""
    if Path(binary).is_file():
        return str(Path(binary).resolve())
    found = shutil.which(binary)
    if found:
        return found
    sibling = Path(sysconfig.get_path("scripts")) / binary
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return str(sibling)
    raise RuntimeError("AWESOME-align executable not found. Install it explicitly with "
                       "scripts/install_linguistic_tools.sh --aligner awesome and "
                       "activate the corresponding Python environment (or use binary: /path/to/awesome-align).")


def parse_awesome_links(line: str, *, source_count: int, target_count: int) -> list[list[int]]:
    """Strictly validate 0-based `i-j` pairs and reject malformed/out-of-range links."""
    links: set[tuple[int, int]] = set()
    for item in line.split():
        m = _LINK.fullmatch(item)
        if not m:
            raise ValueError(f"Malformed AWESOME-align link: {item!r}")
        i, j = int(m[1]), int(m[2])
        if not (0 <= i < source_count and 0 <= j < target_count):
            raise ValueError(f"AWESOME-align link {item!r} outside word bounds "
                             f"{source_count}x{target_count}")
        links.add((i, j))
    return [list(p) for p in sorted(links)]


def run_awesome_alignment(source_ud: str | Path, target_jsonl: str | Path,
                          output: str | Path, *, model: str = "bert-base-multilingual-cased",
                          extraction: str = "softmax", device: str = "auto",
                          binary: str = "awesome-align") -> dict:
    """Use one CLI process for all missing sentences, preserving order/IDs.

    No files are modified until CLI output line counts and all links validate.
    Resource orchestration provides its own atomic managed-output staging.
    """
    if extraction not in {"softmax", "entmax15"}:
        raise ValueError("AWESOME-align extraction must be softmax or entmax15")
    if device not in {"auto", "cpu", "cuda"} and not re.fullmatch(r"cuda:\d+", device):
        raise ValueError("AWESOME-align device must be auto, cpu, cuda, or cuda:<index>")
    exe = _executable(binary)
    targets = read_targets(target_jsonl)
    completed = done_ids(output)
    jobs = []
    seen = set()
    for sentence in read_conllu(source_ud):
        sid = str(sentence.sent_id)
        if sid in seen:
            raise ValueError(f"Duplicate source sentence ID: {sid}")
        seen.add(sid)
        if sid not in targets:
            raise ValueError(f"No target sentence for source {sid}")
        if sid in completed:
            continue
        source_tokens = [word.form for word in sentence.words]
        target_tokens = targets[sid]["tokens"]
        if not source_tokens or not target_tokens:
            raise ValueError(f"Empty source/target word sequence: {sid}")
        if any(any(ch.isspace() for ch in word) for word in source_tokens + target_tokens):
            raise ValueError(f"AWESOME-align requires whitespace-free tokens: {sid}")
        jobs.append((sid, source_tokens, target_tokens))
    if not jobs:
        return {"processed_this_call": 0, "total": len(completed), "path": str(output)}

    # Legacy AWESOME-align implements a BERT-compatible model, so do not
    # automatically substitute SimAlign's default `xlmr` alias.
    model = "bert-base-multilingual-cased" if model == "xlmr" else model
    with tempfile.TemporaryDirectory(prefix="udtl-awesome-") as tmp:
        input_file, align_file = Path(tmp) / "pairs.txt", Path(tmp) / "alignments.txt"
        input_file.write_text("".join(
            f"{' '.join(src)} ||| {' '.join(tgt)}\n" for _, src, tgt in jobs),
            encoding="utf-8")
        cmd = [exe, "--data_file", str(input_file), "--output_file", str(align_file),
               "--model_name_or_path", model, "--extraction", extraction,
               "--batch_size", "8", "--num_workers", "0"]
        env = os.environ.copy()
        if device == "cpu":
            cmd.append("--no_cuda")
        elif device.startswith("cuda:"):
            env["CUDA_VISIBLE_DEVICES"] = device.partition(":")[2]
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
        if result.returncode:
            raise RuntimeError(f"AWESOME-align failed ({result.returncode}): "
                               f"{(result.stderr or result.stdout)[-2000:]}")
        if not align_file.exists():
            raise RuntimeError("AWESOME-align did not produce its requested output file")
        lines = align_file.read_text(encoding="utf-8").splitlines()
        if len(lines) != len(jobs):
            raise ValueError(f"AWESOME-align returned {len(lines)} lines for {len(jobs)} "
                             "sentence pairs; refusing to shift sentence IDs")
        checked = [(sid, parse_awesome_links(line, source_count=len(src),
                           target_count=len(tgt)))
                   for (sid, src, tgt), line in zip(jobs, lines)]
        for sid, pairs in checked:
            append_jsonl(output, {"id": sid, "pairs": pairs, "backend": "awesome-align"})
    return {"processed_this_call": len(checked), "total": len(done_ids(output)),
            "path": str(output)}
