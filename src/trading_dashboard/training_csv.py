"""Generic Training Data panel for bots that commit a training CSV.

Any bot whose config sets ``training_data_path`` and has no bespoke
training renderer (hormuz / billboard / world-cup keep theirs) gets
this: the full committed training table, every column as the trainer
wrote it, paged. The CSV *is* the model's training grain — showing it
verbatim is the point (user 2026-09-14: "show all the pertinent data
in training data").
"""
from __future__ import annotations

import csv
import html
from pathlib import Path
from typing import Dict, List


def _load_rows(path: str | None) -> tuple[List[str], List[Dict[str, str]]]:
    if not path or not Path(path).exists():
        return [], []
    try:
        with open(path, newline="") as f:
            r = csv.DictReader(f)
            rows = list(r)
            return list(r.fieldnames or []), rows
    except (OSError, csv.Error):
        return [], []


def _fmt(col: str, v: str) -> str:
    """Round long floats for the cell; leave text/ints alone."""
    if v is None or v == "":
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return html.escape(str(v))
    if f == int(f) and abs(f) < 1e9:
        return str(int(f))
    return f"{f:.3f}"


def render_training_data_panel(*, bot: dict, current_bot: str | None,
                               page: int = 1, page_size: int = 25,
                               current_tab: str = "training",
                               period_key: str = "all") -> str:
    cols, rows = _load_rows(bot.get("training_data_path"))
    name = bot.get("name") or (current_bot or "bot")
    out: List[str] = []
    out.append("<section class='card'><div class='body'>")
    out.append(f"<h2>Training Data — {html.escape(name)}</h2>")
    if not rows:
        out.append(
            "<p class='small gray'>No committed training table on this "
            "host yet (config key <code>training_data_path</code>: "
            f"<code>{html.escape(str(bot.get('training_data_path')))}"
            "</code>). Run the bot's trainer and pull the repo here."
            "</p></div></section>")
        return "".join(out)

    total = len(rows)
    pages = max(1, (total + page_size - 1) // page_size)
    page = max(1, min(page, pages))
    chunk = rows[(page - 1) * page_size: page * page_size]

    out.append(
        f"<p class='small gray'>{total} rows × {len(cols)} columns — "
        f"the exact table the model trains on. Page {page} of {pages}."
        "</p>")
    out.append("<div style='overflow-x:auto;'><table><thead><tr>")
    for c in cols:
        out.append(f"<th>{html.escape(c)}</th>")
    out.append("</tr></thead><tbody>")
    for r in chunk:
        out.append("<tr>")
        for c in cols:
            out.append(f"<td>{_fmt(c, r.get(c))}</td>")
        out.append("</tr>")
    out.append("</tbody></table></div>")

    if pages > 1:
        from urllib.parse import urlencode

        def _page_link(p: int) -> str:
            params = [("tab", current_tab)]
            if current_bot:
                params.append(("bot", current_bot))
            if period_key and period_key != "all":
                params.append(("period", period_key))
            params.append(("page", str(p)))
            return "/?" + urlencode(params)

        out.append("<div class='small' style='margin-top:8px;display:flex;"
                   "align-items:center;gap:12px;'>")
        if page > 1:
            out.append(f"<a class='tab-pill' href='{_page_link(page-1)}'>"
                       "← Prev</a>")
        out.append(f"<span>page {page}/{pages}</span>")
        if page < pages:
            out.append(f"<a class='tab-pill' href='{_page_link(page+1)}'>"
                       "Next →</a>")
        out.append("</div>")
    out.append("</div></section>")
    return "".join(out)
