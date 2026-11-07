# Lightning AI Pro: complete Studio, GPU and cloud-storage walkthrough

**Audience:** a Lightning AI **Pro** account holder who wants to run UD Treebank Lab on a
single GPU, keep datasets/pretrained models on **Google Drive and/or Dropbox**, stop compute
between experiments, and resume work without recreating the environment.

**Last checked:** 2026-11-06. Lightning's user interface, machine availability, credit and
storage limits may change. Check the live machine selection screen and your account's
**Billing / Usage** panel before enabling paid hardware. The procedure below uses
**AI Studios**, rather than assuming a generic virtual machine rented independently.

## 1. The essential design

| Item | Location / purpose | Survives Studio sleep? |
|---|---|---|
| Project code and YAML | `/teamspace/studios/this_studio/ud-treebank-lab/` | Yes, within the persistent Studio |
| Python packages | Studio's **existing** managed environment | Yes |
| Active source/target corpora | Project `data/` copied from cloud ahead of processing | Yes (but larger Studio images affect startup) |
| Downloaded pretrained model snapshots | Project `assets/models/` | Yes (but consider storage cost/size) |
| Active stages and optimizer checkpoints | Project `runs/` | Yes, if files were fully written before sleep |
| Durable off-platform backups | `drive:UDTreebankLab/...` and/or `dropbox:UDTreebankLab/...` | Independent of the Studio |
| Optional cross-Studio storage | Lightning Drive at `/teamspace/lightning_storage`, uploads at `/teamspace/uploads` | Separate from a single Studio |

**Important:** Lightning's documentation says the Studio environment supports **one managed
conda environment** and advises against creating additional virtual environments inside a
Studio. Therefore **use `scripts/setup_lightning.sh`, not `scripts/setup_vm.sh`** on Lightning.
If a Trankit/IndicTrans2 dependency version conflicts with another backend, create a
*separate Studio* for that backend and share CoNLL-U/JSONL using cloud storage; don't
force incompatible pip installations into a single Studio.

GPU **sleep/stop does not mean the Python process continues running**. Saving the Studio's
filesystem is different from a live training checkpoint. Re-run the same workflow after
restart; the pipeline resumes only from stages and model checkpoints already written.

## 2. Create a Studio (web interface)

1. Sign in to <https://lightning.ai/> with the Pro account and open the **Home / Teamspace**
   view. Choose the intended personal or shared **Teamspace** first: it determines access,
   storage, and billing.
2. Click **New Studio** (some interfaces begin at <https://studio.lightning.ai/>).
3. Give the Studio a descriptive name such as `ud-treebank-lab`. Choose a normal PyTorch/
   Python environment. Use **Lightning cloud** if the account offers that option.
4. **Start with a CPU Studio.** Upload the project, install packages, test commands, download
   data and check paths *before* paying for a GPU.
5. Open the built-in **VS Code** editor or **Terminal**. No SSH configuration is necessary
   for this browser-only path.
6. Check the current environment and free space:

   ```bash
   pwd
   python --version
   python -m pip --version
   df -h .
   ```

**Persistent location:** use `/teamspace/studios/this_studio/` for this walkthrough.
Lightning documents this directory and your home directory as persisted. Avoid writing
important results to an undocumented `/tmp` or an ephemeral scratch volume.

### Optional: create/start/switch/stop from your Ubuntu laptop

Install the **Lightning SDK on your laptop**, not merely inside the remote Studio:

```bash
python3 -m pip install --user -U lightning-sdk
lightning login

# The --teamspace value is of the form OWNER/TEAMSPACE.
lightning studio create --name ud-treebank-lab --teamspace OWNER/TEAMSPACE
lightning studio start --name ud-treebank-lab --teamspace OWNER/TEAMSPACE --machine T4
lightning studio stop --name ud-treebank-lab --teamspace OWNER/TEAMSPACE
```

`T4` is an example **machine identifier in Lightning's CLI documentation**, not a
promise that it will be offered on your Pro plan. To start on the lowest-cost CPU,
prefer the **web interface** or omit the explicit GPU machine option where supported.
Other documented commands are:

```bash
lightning studio switch --name ud-treebank-lab --teamspace OWNER/TEAMSPACE --machine T4
lightning studio ssh --name ud-treebank-lab
```

Run `lightning studio --help` / `lightning studio start --help` if your installed SDK
exposes a revised set of flags. Be careful not to run `create` again to resume an
existing Studio: use **Start**, not a new Studio.

## 3. Put this project's ZIP into the Studio

**Method A — web UI:** use VS Code's file explorer / upload feature (where shown) to
place `ud-treebank-lab-v0.1.1.zip` inside `/teamspace/studios/this_studio/`.

**Method B — Lightning CLI from your Ubuntu laptop:**

```bash
lightning cp ./ud-treebank-lab-v0.1.1.zip \
  lit:///studios/ud-treebank-lab/ud-treebank-lab-v0.1.1.zip
```

For a non-default Teamspace, the full URL is
`lit://OWNER/TEAMSPACE/studios/ud-treebank-lab/ud-treebank-lab-v0.1.1.zip`.

In the **Studio Terminal**:

```bash
cd /teamspace/studios/this_studio
unzip -o ud-treebank-lab-v0.1.1.zip
cd ud-treebank-lab
ls README.md pyproject.toml scripts/setup_lightning.sh
```

If the upload landed elsewhere, move the ZIP into the persistent Studio directory
first. Once unzipped, its extracted folder is the root of the project. Keep it at
this same path after checkpointed training begins.

## 4. Install the project *in the Studio's current environment*

Start on CPU with the core package and tests:

```bash
cd /teamspace/studios/this_studio/ud-treebank-lab
bash scripts/setup_lightning.sh
udtl --help
```

The script installs into the **active** Studio Python environment. It does not create
`.venv`, make a new conda environment, reinstall CUDA itself, or download large models.

When ready, install only the necessary ML extras (preferably still on CPU, if your
machine and package versions allow it):

```bash
bash scripts/setup_lightning.sh align,hf --skip-tests
# Optional MT for source-to-target translation; may need its own compatible Studio:
python -m pip install -e '.[translate]'
# Optional alternative backend:
# python -m pip install -e '.[trankit]'
```

**Do not blindly upgrade PyTorch or CUDA.** Lightning provides a preconfigured GPU
software stack; before replacing packages, check the CUDA build and the compatibility
of `torch`, `transformers`, SimAlign, Trankit and IndicTransToolkit.

The offline test should work without GPU/network models:

```bash
python -m pytest
bash scripts/demo.sh
# Open runs/marathi_demo/preview.html in VS Code / a browser.
```

## 5. Install rclone to connect Google Drive and/or Dropbox

For this project **Google Drive and Dropbox are rclone remotes**, not Lightning's
native S3/GCS data connections. A Dropbox connection in another app (including
ChatGPT) does not automatically authorize a Lightning Studio. `rclone copy` stages real filesystem files locally
for training. This is more reliable than training directly against a live FUSE mount
(which can disconnect or have poor small-file performance).

Check whether rclone is present:

```bash
rclone version
```

If not, on a Studio with sudo/apt access:

```bash
sudo apt-get update
sudo apt-get install -y rclone
rclone version
```

If the distribution version is too old for your provider's current OAuth behavior,
install the current verified release using rclone's official install instructions:
<https://rclone.org/install/>. Use only trusted installers.

### Method 1 — bring over your *existing* laptop rclone configuration (easiest)

If Google Drive / Dropbox already work with rclone on Ubuntu, you do **not** need to
re-authorize each service from scratch.

On your **Ubuntu laptop**:

```bash
rclone config file
rclone listremotes
# The first command prints the actual configuration path; do not assume its location.
# Example, when the printed file is ~/.config/rclone/rclone.conf:
lightning cp ~/.config/rclone/rclone.conf \
  lit:///studios/ud-treebank-lab/rclone.conf
```

Inside the **Studio**, move that uploaded credential-bearing file out of the project:

```bash
mkdir -p "$HOME/.config/rclone"
mv /teamspace/studios/this_studio/rclone.conf "$HOME/.config/rclone/rclone.conf"
chmod 700 "$HOME/.config/rclone"
chmod 600 "$HOME/.config/rclone/rclone.conf"
rclone config file
rclone listremotes
```

If your laptop uses a different config path, substitute the actual path. If you
use a non-default Studio/teamspace name, adjust the upload URL. **The rclone config
contains OAuth access/refresh tokens.** Never commit it, upload it into a public
repository, paste it into a support ticket, or put it inside the project ZIP.

### Method 2 — set up a new Google Drive remote

```bash
rclone config
```

- Choose **New remote**, name it `drive`, choose storage type **Google Drive**.
- **As of 2026, rclone says its shared Google Drive OAuth client ID is being retired.**
  Create your own **Desktop-app OAuth client** using the steps at
  <https://rclone.org/drive/#making-your-own-client-id>. That guide covers enabling
  Google Drive API, OAuth consent, scopes, creating a desktop app ID/secret, adding a
  test user, and the publication setting needed for persistent authorization.
- Enter that **client ID and secret** in the rclone prompts. For a simple private
  Drive backup, the full `drive` scope is convenient but broad; use a more restricted
  scope when it meets your needs.
- Lightning's terminal may not have a browser on localhost. At the prompt asking
  whether to open a browser, choose **No** and follow the printed
  **`rclone authorize`** steps on your *Ubuntu laptop* (which has a browser).
  Paste only the token requested by the trusted rclone CLI into the Studio terminal.
  Refer to <https://rclone.org/remote_setup/>. Keep the token private.
- Test: `rclone lsd drive:`. Only you can decide which folders to expose.

### Method 3 — set up a new Dropbox remote

```bash
rclone config
```

- Choose **New remote**, name it `dropbox`, choose storage type **Dropbox**.
- Follow its OAuth browser authorization. For a headless Studio, answer **No** to
  the auto-open-browser prompt and follow the local `rclone authorize` instructions;
  alternatively configure Dropbox on the laptop and use Method 1.
- Test: `rclone lsd dropbox:`.
- Backend reference: <https://rclone.org/dropbox/>.

You can configure **both** remotes in the same rclone config, or just one. The
example names `drive` and `dropbox` are arbitrary: replace them with names reported
by `rclone listremotes` if yours differ.

## 6. Choose a simple common cloud folder layout

Create/maintain something like this **inside Google Drive or Dropbox**:

```text
UDTreebankLab/
  data/
    ud_hindi_hdtb/
      hi_hdtb-ud-train.conllu
    parallel/
      marathi.jsonl
    alignments.jsonl
    gold/
      mr_dev.conllu
      mr_test.conllu
  models/
    xlm-roberta-base/
      config.json, tokenizer files, model weights ...
    indictrans2-320m/
      config.json, tokenizer files, model weights ...
  backups/
    runs_2026-11-06/          # example immutable experiment snapshot
```

The exact filenames must match the YAML config. **Never put independent gold test
annotations in a translation/alignment training input.** Download only data that
you have permission to use, and respect source-treebank licenses.

The bundled helper uses **non-deleting `rclone copy`**, not `sync`:

```bash
cd /teamspace/studios/this_studio/ud-treebank-lab
# Inspect remote:
bash scripts/cloud_io.sh list drive:UDTreebankLab

# Stage DATA from Google Drive:
bash scripts/cloud_io.sh pull-data drive:UDTreebankLab/data

# Alternatively, stage DATA from Dropbox:
# bash scripts/cloud_io.sh pull-data dropbox:UDTreebankLab/data

# Stage PRETRAINED MODEL SNAPSHOTS from either service:
bash scripts/cloud_io.sh pull-models drive:UDTreebankLab/models
# bash scripts/cloud_io.sh pull-models dropbox:UDTreebankLab/models

# Optional: inspect transfers without copying anything:
UDTL_DRY_RUN=1 bash scripts/cloud_io.sh pull-data drive:UDTreebankLab/data
```

This creates `data/` and `assets/models/` directly under the project. It does not
store data credentials in YAML. If source data aren't already in cloud storage:

```bash
udtl fetch-ud --treebank UD_Hindi-HDTB --short-name hi_hdtb --out data/ud_hindi_hdtb
bash scripts/cloud_io.sh push-data drive:UDTreebankLab/data
```

If the cloud contains old training outputs needed for **resume**:

```bash
bash scripts/cloud_io.sh pull-runs dropbox:UDTreebankLab/backups/runs_2026-11-06
```

Make sure the restoring project configuration, output paths and model settings are
unchanged. Check that restored `model_hf/last.pt`, per-sentence `.stage.jsonl`,
`.state.json` and data files are present before resuming.

### Alternative: use Lightning Drive (no rclone needed)

In your Lightning Teamspace, upload data with the **Drive** tab or from the laptop:

```bash
lightning cp -r ./data/ lit://OWNER/TEAMSPACE/uploads/data/
```

Inside the Studio, uploaded files are normally under `/teamspace/uploads/data/`.
Lightning's shared Drive lives under `/teamspace/lightning_storage/`. You can either
copy files into the project's `data/` directory or change YAML input paths to their
absolute mounted locations. **Do not assume** Google Drive files appear there:
Lightning Drive and Google Drive are different services.

## 7. Download or reuse pretrained model snapshots

There are **two distinct kinds of model files**:

1. **Hugging Face pretrained model snapshots:** `config.json`, tokenizer and weight
   files. You can download, upload to cloud, retrieve later, and give the project a
   *local directory* path.
2. **Fine-tuned UD parser checkpoints:** the project's `runs/<experiment>/model_hf/`
   containing `last.pt` / `best.pt`. These are **not** interchangeable with the
   pretrained Hugging Face snapshot.

For example, after installing `huggingface_hub` and meeting the model's license/access
conditions:

```bash
python -m pip install -U huggingface_hub
mkdir -p assets/models
hf download xlm-roberta-base --local-dir assets/models/xlm-roberta-base
# Optional; check access, terms, and compatibility before downloading:
# hf auth login
# hf download ai4bharat/indictrans2-indic-indic-dist-320M \
#   --local-dir assets/models/indictrans2-320m

bash scripts/cloud_io.sh push-models drive:UDTreebankLab/models
```

If `hf` isn't recognized, inspect `python -m huggingface_hub --help` or follow the
Hugging Face Hub CLI guide for your installed version. When model snapshots already
exist on Google Drive/Dropbox, **use `pull-models` instead**, avoiding an unnecessary
redownload. Do not upload authentication tokens with model files.

For models staged in `assets/models/`, you can use the supplied
`configs/hindi_to_marathi_lightning_cloud.yaml`, which expects translated target
sentences at `data/parallel/marathi.jsonl` and a locally staged XLM-R snapshot.
It also provides commented IndicTrans2 local-snapshot settings.

Alternatively, edit `configs/hindi_to_marathi.yaml` as follows:

```yaml
pairs:
  - source_ud: ../data/ud_hindi_hdtb/hi_hdtb-ud-train.conllu
    translate:
      source_language: hin_Deva
      target_language: mar_Deva
      model: ../assets/models/indictrans2-320m   # local model snapshot (optional)
      batch_size: 8
training:
  backend: hf
  hf:
    pretrained: ../assets/models/xlm-roberta-base # local model snapshot
    epochs: 6
    batch_size: 4
    accumulate: 4
    save_steps: 200
```

**v0.1.1 enhancement:** model paths beginning `../` or `./` are resolved relative
to the YAML config directory; `~/` expands from the home directory and `/` paths
are absolute. Ordinary Hub IDs such as `xlm-roberta-base` are left unchanged. Keep the rest of the original YAML
settings (projection constraints, alignment, etc.). For SimAlign, the `model: xlmr`
setting is a SimAlign alias, not the above local snapshot path; it may still need to
fetch its own pretrained weights initially.

For a standard Hugging Face model ID rather than an explicit snapshot, keep the
ordinary name and use a persistent Hugging Face cache in the Studio:

```bash
export HF_HOME="$HOME/.cache/huggingface"
# Add that export to ~/.bashrc if you want the same value in new Terminal sessions.
```

**Do not back up `$HF_HOME` wholesale to a cloud folder**: Hugging Face authentication
tokens may be stored alongside caches, and its Hub cache may contain symlinks that
some cloud-copy operations do not preserve. Use `hf download --local-dir` and
`push-models` / `pull-models` for portable snapshots instead.

## 8. Switch the Studio to a GPU and run

Inside the Studio, after CPU setup and data staging:

1. Open the top **environment / machine** control, select **Switch to GPU**.
2. Pick a single GPU with approximately **12–16 GB VRAM or more** if offered at
   an acceptable cost. The exact choices and prices depend on the current account;
   check them in the UI. A lower-memory GPU may work for frozen-encoder pilots.
3. Confirm that the Studio has switched and verify the runtime:

   ```bash
   nvidia-smi
   python - <<'PY'
   import torch
   print('Torch:', torch.__version__)
   print('CUDA available:', torch.cuda.is_available())
   if torch.cuda.is_available():
       print('GPU:', torch.cuda.get_device_name(0))
   PY
   ```

4. If CUDA is **False**, fix the PyTorch/CUDA stack before starting training; don't
   interpret a successful `nvidia-smi` alone as proof that PyTorch can use the GPU.
5. **Pilot a small test** before processing a whole HDTB treebank. The bundled
   `examples/demo.yaml` verifies projection/evaluation offline but is *not* a GPU
   throughput benchmark. For a meaningful pilot, create a config referencing a
   500–2,000 sentence source subset and independent target gold dev/test sets.

   ```bash
   udtl run configs/hindi_to_marathi.yaml --train none  # full projection, no training
   # After checking silver output and reducing test-data size as appropriate:
   udtl run configs/hindi_to_marathi.yaml             # projection + configured training
   ```

**Warning:** the commands above run the **full configured source dataset**, not a
small pilot unless you edit the input file and YAML first. To create an actual pilot,
prepare a separate CoNLL-U subset and point a copied YAML preset at it. Do not
use the gold test set as projected training input. A source-only test on CPU is
cheaper than discovering model/version problems during paid GPU training.

If running for many hours without interactive activity, inspect the Studio's
**auto-sleep** settings. Default idle-sleep saves money, but you must prevent the
Studio from sleeping unexpectedly during an unattended run. Disable or extend
auto-sleep **only for the active experiment**, then restore it afterward.
Billing accrues according to current compute usage, not the fact that the user
has a Pro subscription; check the live billing panel. If you still need the Studio
for CPU-only work, **switch GPU -> CPU** instead of sleeping the Studio. If you
need no compute at all, **Sleep** it. Verify the selected machine after either action.

## 9. Save outputs to Google Drive/Dropbox, stop GPU, resume later

After a checkpointed training step is complete (or after gracefully interrupting
and confirming checkpoint creation):

```bash
cd /teamspace/studios/this_studio/ud-treebank-lab
ls -lh runs/marathi_full/model_hf/last.pt  # when HF training actually saved a checkpoint

# Copy a reproducible snapshot; choose a new cloud folder for each backup date/run.
bash scripts/cloud_io.sh push-runs dropbox:UDTreebankLab/backups/runs_2026-11-06
# Optional second independent copy:
bash scripts/cloud_io.sh push-runs drive:UDTreebankLab/backups/runs_2026-11-06
```

`rclone copy` does **not remove** destination-only files, but **can overwrite files
with the same name** if they differ; use unique dated/experiment-specific backup
folders if you need a *history* of checkpoints. Don't transfer `last.pt` while
PyTorch is actively rewriting it. Upload only the data/models permitted by their
licenses and your cloud-sharing permissions.

### Stop / sleep the Studio (web interface)

1. Verify you have a **complete checkpoint**, and do the optional cloud copy.
2. End interactive runs gracefully. Be aware `Ctrl+C` alone is **not** a guarantee
   of a new checkpoint; inspect the saved file and logs.
3. Inside Studio, click the **top-right machine indicator** and use the **Sleep**
   toggle/control. From the Teamspace, use the **three-dot menu** for the Studio
   and choose **Sleep** (Lightning's documented interface).
4. Confirm the Studio actually shows **Sleeping** / stopped and that paid GPU
   compute is no longer active; check the usage panel if uncertain.
5. Alternatively, from the laptop:

   ```bash
   lightning studio stop --name ud-treebank-lab --teamspace OWNER/TEAMSPACE
   ```

**Do not select Delete** when you mean Sleep: deletion has different consequences.
Lightning documents that Studio files, installed software, and environment persist
through Sleep/Switch, but storage quotas, retention settings and billing can still
apply. Do not treat Studio persistence as the **only** backup.

### Resume a day later

1. Open the *same* Studio; select **Start/Wake**. It may start on CPU or on the last
   selected hardware depending on account and current UI; check the machine display.
2. Open the terminal, verify data and checkpoints, and switch to GPU if required.
3. If files are missing or you are recreating a Studio, restore from cloud:

   ```bash
   cd /teamspace/studios/this_studio/ud-treebank-lab
   bash scripts/cloud_io.sh pull-data drive:UDTreebankLab/data
   bash scripts/cloud_io.sh pull-models drive:UDTreebankLab/models
   bash scripts/cloud_io.sh pull-runs dropbox:UDTreebankLab/backups/runs_2026-11-06
   ```

4. Re-run **the same** configuration:

   ```bash
   udtl run configs/hindi_to_marathi.yaml
   ```

Translation, alignment and projection use resumable stage files. The bundled HF
parser uses `last.pt` plus optimizer and RNG state for checkpoint resume. It may
redo work since its previous `save_steps` interval. The Trankit wrapper can
skip **completed stages only**; it does not guarantee mid-epoch recovery.

5. Once finished, run evaluation/visualization, back up outputs and **Sleep** again.

## 10. Optional: direct cloud folders and rclone mounts

For data **already available** in Lightning Drive at `/teamspace/uploads/` or a
native `/teamspace/lightning_storage/...` folder, YAML fields accepting file paths
can point to **absolute mounted paths**. This can avoid copying the same corpus
into the Studio. Check that access/read speed is adequate.

`rclone mount drive:... ./mountpoint` is possible on Linux **only if the Studio allows
FUSE**; don't assume it does. Even when available, mounts can disconnect and random
small-file reads may be much slower than local storage. The recommended default is
**copy to local persistent disk, process, then copy results out**. Never place the
active `model_hf/last.pt` checkpoint on an unreliable FUSE mount.

## 11. Troubleshooting

| Problem | What to check |
|---|---|
| Studio starts but no GPU | Current selected machine; switch to GPU; `nvidia-smi` |
| Torch says CUDA unavailable | Torch/CUDA build mismatch; check `torch.__version__`, `torch.version.cuda` |
| `python -m venv` or conda environment fails | Lightning uses one managed environment; use `setup_lightning.sh` |
| Studio stopped mid-training | Restore previous complete checkpoint; run same YAML; check auto-sleep |
| `rclone lsd drive:` fails | OAuth refresh token, own Google client ID, `rclone config file`, config permissions |
| A new Drive remote won't authenticate | Follow rclone's 2026 OAuth-client guidance and headless `rclone authorize` |
| Dropbox cloud folder appears empty | Check remote name, Dropbox app folder vs full Dropbox access, path and permissions |
| Google Drive model file won't load | Ensure a full HF *snapshot* (config/tokenizer/weights), not just an individual `.bin` |
| HF repo access denied | Accept model conditions, `hf auth login`, check token/scopes; keep tokens secret |
| HF model downloads again | Check local snapshot model path; SimAlign may independently download embeddings |
| Stage restarts from beginning | Compare config, file paths, `.state.json`, `.stage.jsonl`, fingerprints |
| HF training refuses resume | Same training data/parameters/model path required; see `docs/TRAINING.md` |
| Long shutdown/startup | Large Studio filesystem; move datasets/weights to cloud storage/Lightning Drive |
| Pro credits are being consumed | Switch GPU -> CPU or Sleep; verify the machine actually stopped |

## 12. Security and cost checklist

- Keep Google Drive and Dropbox **OAuth credentials outside the project directory**.
  `chmod 600 ~/.config/rclone/rclone.conf` and never publish the file.
- Use a private folder/bucket and permissions appropriate for corpus and checkpoint
  license terms; encrypted rclone `crypt` remote is an option for sensitive data.
- Never copy `$HF_HOME/token`, `.netrc`, Lightning access tokens, or OAuth JSON
  client secrets into public backups.
- Stage data on **CPU**, use **GPU only while needed**, then **Sleep**.
- Pilot on a small corpus, watch `nvidia-smi` and the Lightning billing meter.
- Keep at least one **off-Studio copy** of the source data, run state and checkpoints.
- **Gold test data must remain independent**. Alignment projection and training on
  pseudo-gold are not substitutes for evaluation against human annotations.
- Read the licensing terms of Hindi HDTB, translation models, Hugging Face models,
  Apertium resources, and any derived treebank you intend to redistribute.

## Official references (review for UI changes)

- Lightning — [Getting started](https://lightning.ai/docs/platform/overview/getting-started)
- Lightning — [AI Studio](https://lightning.ai/docs/platform/build/ai-studio)
- Lightning — [Start and stop Studio](https://lightning.ai/docs/platform/build/ai-studio/start-and-stop-studio)
- Lightning — [Environment persistence](https://lightning.ai/docs/platform/build/ai-studio/environment-persistence)
- Lightning — [CLI](https://lightning.ai/docs/platform/build/ai-studio/cli)
- Lightning — [Add data / Lightning Drive](https://lightning.ai/docs/platform/build/ai-studio/add-data)
- rclone — [Installation](https://rclone.org/install/)
- rclone — [Google Drive and own OAuth client ID](https://rclone.org/drive/#making-your-own-client-id)
- rclone — [Dropbox](https://rclone.org/dropbox/)
- rclone — [Headless remote setup](https://rclone.org/remote_setup/)
- rclone — [Copy semantics](https://rclone.org/commands/rclone_copy/)
- Hugging Face — [Hub CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)
