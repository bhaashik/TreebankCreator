"""Allocate independent bilingual-language jobs to distinct visible GPUs.

In Kaggle T4x2, 0 and 1 are independent 16-GB devices. Running separate Python
processes avoids accidental model placement or memory sharing across GPUs.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .bilingual_config import read_config


def _device(gpu: str) -> str:
    if gpu == "cpu":
        return "cpu"
    if not gpu.isdecimal():
        raise ValueError("GPU IDs must be 0,1,... or cpu")
    return "cuda"


def launch(configs: list[str], gpus: list[str], *, precache_source: bool = True,
           dry_run: bool = False, poll_seconds: float = 0.25) -> dict:
    if not configs or not gpus:
        raise ValueError("Provide at least one --config and one --gpu")
    cuda_gpus = [g for g in gpus if g != "cpu"]
    if len(cuda_gpus) != len(set(cuda_gpus)):
        raise ValueError("Each CUDA GPU slot must have a distinct physical ID")
    for gpu in gpus:
        _device(gpu)
    entries = [(str(Path(name).expanduser().resolve()), read_config(name)) for name in configs]
    dirs = [cfg["output_dir"] for _, cfg in entries]
    if len(set(dirs)) != len(dirs):
        raise ValueError("Each language configuration requires its own output_dir")
    source_keys = {}
    for name, cfg in entries:
        key = (cfg["source_cache"], cfg["source_ud"], cfg["model"],
               cfg["max_words"], cfg["max_subtokens"], cfg["shard_size"])
        source_keys.setdefault(key, name)
    def cmd(name: str, gpu: str, cache: bool):
        base = [sys.executable, "-m", "ud_treebank_lab"]
        return (base + ["cache-bilingual", "--config", name, "--side", "source", "--device", _device(gpu)]
                if cache else base + ["train-bilingual", "--config", name,
                                   "--device", _device(gpu)])
    plan = {"gpu_slots": gpus, "source_preparation": [cmd(name, gpus[0], True)
            for name in source_keys.values()] if precache_source else [],
            "jobs": [{"config": name, "language": cfg["language"], "output_dir": cfg["output_dir"]}
                     for name, cfg in entries]}
    if dry_run:
        return {"status": "dry-run", **plan}
    if any(g != "cpu" for g in gpus):
        import torch
        available = torch.cuda.device_count()
        for gpu in gpus:
            if gpu != "cpu" and int(gpu) >= available:
                raise RuntimeError(f"Requested GPU {gpu}; only {available} CUDA devices visible")
    # Stage shared Hindi cache in its own process so GPU allocations are fully
    # released *before* workers start. If all shards exist, caching is a no-op.
    for command in plan["source_preparation"]:
        env = os.environ.copy()
        gpu = gpus[0]
        env["CUDA_VISIBLE_DEVICES"] = "" if gpu == "cpu" else gpu
        print("Preparing shared Hindi embedding cache:", " ".join(command), flush=True)
        subprocess.run(command, env=env, check=True)
    remaining = list(entries)
    free = gpus[:]
    running = []
    finished = []
    while remaining or running:
        while remaining and free:
            name, cfg = remaining.pop(0)
            gpu = free.pop(0)
            output = Path(cfg["output_dir"])
            output.mkdir(parents=True, exist_ok=True)
            log_path = output / "gpu_worker.log"
            log = log_path.open("a", encoding="utf-8")
            command = cmd(name, gpu, False)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "" if gpu == "cpu" else gpu
            # Threads must be limited: Kaggle dual T4 has shared CPU cores.
            env.setdefault("OMP_NUM_THREADS", "2")
            env.setdefault("TOKENIZERS_PARALLELISM", "false")
            print(f"Starting {cfg['language']} on GPU {gpu}; log: {log_path}", flush=True)
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            running.append((process, gpu, name, log_path, log))
        for item in running[:]:
            process, gpu, name, logfile, log = item
            result = process.poll()
            if result is None:
                continue
            log.close()
            free.append(gpu)
            running.remove(item)
            finished.append({"config": name, "gpu": gpu, "exit_code": result,
                             "log": str(logfile)})
            print(f"Finished {Path(name).name} on GPU {gpu}: return code {result}", flush=True)
        if running:
            time.sleep(poll_seconds)
    failures = [item for item in finished if item["exit_code"]]
    return {"status": "failed" if failures else "complete", "results": finished,
            "failed": len(failures), "jobs": len(finished)}
