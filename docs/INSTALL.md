# Installation and GPU VM operations

**Lightning AI Pro Studio users:** follow **[`LIGHTNING_AI_PRO.md`](LIGHTNING_AI_PRO.md)**
for the correct Studio-managed Python environment, machine creation/start/sleep,
Google Drive and Dropbox via rclone, remote model snapshots, and complete resume
procedure. The **virtual-environment commands below are for generic Ubuntu/Linux
VMs, not Lightning Studios**.

## Supported baseline

- **Linux x86-64**, Python 3.10 or 3.11 recommended, internet for the initial model downloads.
- CPU only: core projector + evaluation/visualization + external alignment import.
- A single NVIDIA CUDA GPU (8–24 GB, 12–16 GB preferable) for SimAlign / MT / training.
- The **GPU must have a PyTorch-compatible CUDA runtime**; a machine described as a "TensorFlow GPU VM" is not automatically a usable PyTorch GPU. Check `torch.cuda.is_available()`.
- On hosted GPUs with disposable local disks, mount/persist the **whole `runs/` directory** and the Hugging Face model cache, or copy checkpoints to durable storage before teardown. Do not share account tokens in configs.

## Minimal installation

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -e '.[test]'
python -m pytest
bash scripts/demo.sh
```

Install on Ubuntu using normal `python3-venv` if your base image lacks `venv`. The provided `scripts/setup_vm.sh` performs these steps automatically **on generic VMs**. In a Lightning AI Studio, run `bash scripts/setup_lightning.sh` instead; Lightning documents one existing managed conda environment and does not support creating additional virtual environments inside a Studio.

## GPU checks

Install the **appropriate GPU build of PyTorch** using the version-specific instructions at https://pytorch.org/get-started/locally/ (do not assume the pip default builds are compatible). Then:

```bash
python - <<'PY'
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
PY
```

Only add the dependencies required for the chosen run:

```bash
pip install -e '.[align]'        # SimAlign and HF multilingual embeddings
pip install -e '.[hf]'           # HF checkpointable UD parser
pip install -e '.[translate]'    # IndicTrans2/IndicTransToolkit adapter
pip install -e '.[trankit]'      # Training a custom Trankit POS/dep parser
pip install -e '.[stanza]'       # Automatic parsing of source aligned text and optional PSG inference
```

**Dependency compatibility:** Some Trankit versions constrain transformer/PyTorch dependencies. If pip rejects a combination, put Trankit in a **separate venv on a generic Ubuntu VM**, or **a separate Studio on Lightning AI** (do not make an extra venv inside a Studio). Run annotation projection in the primary environment; both backends communicate through plain CoNLL-U files. Do not force-install incompatible versions. IndicTransToolkit may require a comparatively recent torch/transformers build; isolate translation in another compatible VM/Studio if necessary.

## Hugging Face login and download caching

Translation model `ai4bharat/indictrans2-indic-indic-dist-320M` may require agreeing to the HF model conditions (check its model card), followed by `hf auth login` when necessary. The adapter uses `trust_remote_code=True`; only load verified publishers' revisions. You can override `model` in the YAML, but only with a model implementing the IndicTrans2/IndicTransToolkit interface.

```bash
export HF_HOME="$HOME/.cache/huggingface"
# HF_HOME should point to persistent storage on ephemeral GPU VMs.
```

Cloud accounts often disconnect interactive sessions. `udtl run` can be re-executed, but only if its output directory survives. For HF training, `last.pt` includes optimizer/model states and exact batch/epoch progress at the previous save step (default every 200 optimizer steps). For Trankit, the wrapper only recognizes a **completed** task; do not assume mid-epoch recovery.

## Fastest learning path

1. `bash scripts/demo.sh` (takes a moment on CPU).
2. `udtl fetch-ud --out data/ud_hindi_hdtb` (large UD source corpus; no GPU needed).
3. Assemble a **500–2,000 sentence** Hindi↔target parallel subset with reliable one-sentence-to-one-sentence pairing.
4. Use pretokenized target JSONL, SimAlign, projection and inspection.
5. Hand-annotate 100–300 target sentences for preliminary *independent* gold evaluation.
6. Train first on 1,000–5,000 highest-confidence projections, then scale if results justify it.

## Time/memory caveats

- The demo does not download neural models.
- SimAlign+XLM-R is GPU-accelerated, but alignment throughput depends on sentence length, batching support, and hardware. This adapter processes sentences sequentially.
- IndicTrans2 320M is not free computationally; 13k Hindi sentences may take minutes to hours or longer depending on hardware and model setup.
- The default HF encoder is frozen (lower VRAM but less representation adaptation); unfreezing requires more compute, slower training and larger checkpoints.
- Trankit has its own model defaults and memory demands. Expect to **pilot and benchmark** before scheduling a full 22-language run.
- A few days **per language** on a single good GPU is a feasible *research target*, **not a guarantee**. All 22 languages sequentially may take much longer.

## Recommended directory handling

```bash
mkdir -p data runs
# Leave generated *.stage.jsonl, *.state.json and model_hf/last.pt intact for resume.
# Use a different output_dir for different source data, parameter configurations, or experiments.
```
