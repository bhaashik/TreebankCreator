#!/usr/bin/env python3
"""Build release-based handoff archives using original Git tag history."""
import argparse,hashlib,json,subprocess,tempfile,zipfile
from pathlib import Path
VERSIONS=["v0.1.0","v0.1.1","v0.2.0","v0.3.0","v0.3.1","v0.3.2","v0.3.3","v0.3.4"]
DOCS=["README.md","AGENTS.md","docs/CONVERSATION_HANDOFF.md","docs/CODEX_START_HERE.md","docs/RELEASE_PROVENANCE.md","docs/TITAN_V.md","docs/EXTERNAL_TOOL_INSTALL.md","docs/EXTERNAL_RESOURCE_IMPORT.md","docs/CONDITIONAL_RESOURCE_SETUP.md","docs/CORPUS_REFINEMENT.md","docs/BILINGUAL_TRAINING.md","docs/LIMITATIONS.md","notebooks/Kaggle_Dual_T4.ipynb"]
def git(*args):
    try: return subprocess.check_output(["git",*args],stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError: return None
def main():
    p=argparse.ArgumentParser();p.add_argument("--kind",choices=["complete","essential"],required=True);p.add_argument("--out",required=True);o=p.parse_args()
    items={}
    record=git("show","origin/handoff/conversation-2026-11-13:handoff/TreebankCreator_Conversation_Record_2026-11-13.md")
    if record: items["conversation/TreebankCreator_Conversation_Record_2026-11-13.md"]=record
    for tag in (VERSIONS if o.kind=="complete" else ["v0.3.4"]):
        data=git("archive","--format=zip",tag)
        if data:items[f"releases/TreebankCreator-source-{tag}.zip"]=data
    with tempfile.TemporaryDirectory() as td:
        name=Path(td)/"release-history.bundle"
        subprocess.check_call(["git","bundle","create",str(name),"--all"])
        items["history/TreebankCreator-release-history.bundle"]=name.read_bytes()
    for path in DOCS:
        data=git("show",f"v0.3.4:{path}")
        if data:items["latest/"+path]=data
    items["README_ARCHIVE_SCOPE.txt"]=(
       "GitHub reconstruction, not an exact replica of the local ChatGPT ZIP.\n"
       "Original local full ZIP captures 75 recoverable files, including duplicates\n"
       "and intermediate attachments not all committed to GitHub. GitHub copy\n"
       "reconstructs source versions, history and latest key documents.\n"
       "Conversation document is a reconstructed handoff, not a verbatim export.\n"
    ).encode()
    rows=[{"path":key,"size":len(data),"sha256":hashlib.sha256(data).hexdigest()} for key,data in sorted(items.items())]
    items["MANIFEST.json"]=json.dumps({"kind":o.kind,"type":"GitHub source-based reconstruction","files":rows},ensure_ascii=False,indent=2).encode()
    out=Path(o.out);out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for key,data in sorted(items.items()):z.writestr(key,data)
    with zipfile.ZipFile(out) as z:assert z.testzip() is None
    print(f"Built {out}: {out.stat().st_size} bytes, {len(items)} entries")
if __name__=="__main__":main()
