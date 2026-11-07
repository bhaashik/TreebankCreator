"""Standalone HTML report, inline SVG dependency trees and projection diagnostics."""
from __future__ import annotations

from collections import Counter
import html
import json
from pathlib import Path

from .conllu import Sentence, read_conllu
from .utils import jsonl

CSS = """
:root {font-family: system-ui, sans-serif; color-scheme: light; background:#f7f9fc;color:#132038}
body {max-width:1180px;margin:26px auto;padding:0 22px;line-height:1.5}
h1,h2 {line-height:1.15} h1 {font-size:28px}h2 {font-size:21px;margin-top:28px}
.card {border:1px solid #dbe2ed;background:white;padding:16px;border-radius:10px;margin:14px 0;box-shadow:0 1px 3px #dce5f0}
.stat {display:inline-block;min-width:130px;margin:6px 12px 6px 0;padding:12px;border-radius:8px;background:#e9f0fc}
.stat b{display:block;font-size:21px}p.small,small{color:#52637d;font-size:13px}
svg.tree {max-width:100%;overflow:visible} .words {overflow-x:auto;overflow-y:hidden;padding:8px}
.arc{fill:none;stroke:#3e67ad;stroke-width:1.6}.arc.root{stroke:#17996e}
.token{font-size:14px;fill:#132038}.pos{font-size:11px;fill:#64748b}.relation{font-size:10px;fill:#154c9d}
code {padding:2px 4px;background:#eef1f6}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:7px;text-align:left;border-bottom:1px solid #e5e7eb} th{background:#f1f5f9}
.bar{height:12px;border-radius:4px;background:#3767ad}details{border-top:1px solid #eef1f4}
"""


def tree_svg(sentence: Sentence) -> str:
    # UTF-8 text with inline SVG; arcs encode directional head/dependent structures.
    n = len(sentence.words)
    cell = 96
    width = max(400, n * cell + 50)
    base = 160
    height = 222
    nodes = {w.id: 30 + (w.id - .5) * cell for w in sentence.words}
    paths = []
    for w in sentence.words:
        x = nodes[w.id]
        if w.head == 0:
            paths.append(f'<path class="arc root" d="M{x:.0f},{base} Q{x-30:.0f},30 {x:.0f},12"/>')
            paths.append(f'<text class="relation" x="{x+10:.0f}" y="20">root</text>')
            continue
        hx = nodes[w.head]
        distance = abs(w.head - w.id)
        top = max(18, 150 - min(110, 26 + 15 * distance))
        midx = (hx + x) / 2
        paths.append(f'<path class="arc" d="M{hx:.0f},{base} Q{midx:.0f},{top:.0f} {x:.0f},{base}"/>')
        paths.append(f'<text class="relation" x="{midx:.0f}" y="{top+8:.0f}" text-anchor="middle">{html.escape(w.deprel)}</text>')
    for w in sentence.words:
        x = nodes[w.id]
        word = html.escape(w.form)
        if len(word) > 14: word = word[:13] + "…"
        paths.append(f'<text class="token" x="{x:.0f}" y="186" text-anchor="middle">{word}</text>')
        paths.append(f'<text class="pos" x="{x:.0f}" y="204" text-anchor="middle">{html.escape(w.upos)}</text>')
    return (f'<div class="words"><svg class="tree" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">'
            + "".join(paths) + '</svg></div>')


def render_html(conllu: str | Path, out_html: str | Path, metrics_json: str | Path | None = None,
                metadata_jsonl: str | Path | None = None, limit: int = 40) -> dict:
    metrics = json.loads(Path(metrics_json).read_text(encoding="utf-8")) if metrics_json else {}
    meta = list(jsonl(metadata_jsonl)) if metadata_jsonl else []
    sents = list(read_conllu(conllu))
    parts = ["<!doctype html><html lang='en'><head><meta charset='utf-8'>",
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>UD Treebank Lab report</title>', f'<style>{CSS}</style></head><body>',
             '<h1>UD Treebank Lab — Treebank inspection</h1>',
             f'<p class="small">Source: {html.escape(str(conllu))} · {len(sents)} sentences</p>']
    if metrics:
        parts.append('<section class="card"><h2>Gold evaluation</h2>')
        for name, value in metrics.get("scores", {}).items():
            shown = "n/a" if value is None else f"{100*value:.1f}%"
            parts.append(f'<div class="stat"><small>{html.escape(name)}</small><b>{shown}</b></div>')
        parts.append("</section>")
        worst = [(x["gold"], x["pred"], x["count"]) for x in metrics.get("deprel_confusion", [])
                 if x["gold"] != x["pred"]][:15]
        if worst:
            parts.append('<section class="card"><h2>Frequent relation confusions</h2><table><tr><th>Gold</th><th>Predicted</th><th>Count</th></tr>')
            for g, p, num in worst:
                parts.append(f'<tr><td>{html.escape(g)}</td><td>{html.escape(p)}</td><td>{num}</td></tr>')
            parts.append('</table></section>')
    if meta:
        avg = sum(float(m.get("quality", 0)) for m in meta) / len(meta)
        parts.append(f'<section class="card"><h2>Projection diagnostics</h2><div class="stat"><small>Mean quality (heuristic)</small><b>{avg:.3f}</b></div>')
        buckets = Counter(min(9, int(10 * float(m.get("quality", 0)))) for m in meta)
        maximum = max(buckets.values()) if buckets else 1
        parts.append('<table><tr><th>Quality interval</th><th>Count</th><th>Distribution</th></tr>')
        for i in range(10):
            count = buckets.get(i, 0)
            parts.append(f'<tr><td>{i/10:.1f}–{(i+1)/10:.1f}</td><td>{count}</td>'
                         f'<td><div class="bar" style="width:{count/max(1,maximum)*90:.1f}%"></div></td></tr>')
        parts.append('</table></section>')
    parts.append('<h2>Dependency trees</h2><p class="small">Arrows are represented by curves from head to dependent; scrolling works for long sentences.</p>')
    for i, sent in enumerate(sents[:limit]):
        parts.append(f'<section class="card"><b>{i+1}. {html.escape(sent.sent_id)}</b><small> — {len(sent.words)} words</small>')
        parts.append(tree_svg(sent))
        parts.append('<details><summary>CoNLL-U</summary><pre style="overflow:auto">'
                     + html.escape(sent.to_conllu()) + '</pre></details></section>')
    parts.append('</body></html>')
    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    Path(out_html).write_text("\n".join(parts), encoding="utf-8")
    return {"html": str(out_html), "rendered_trees": min(len(sents), limit)}
