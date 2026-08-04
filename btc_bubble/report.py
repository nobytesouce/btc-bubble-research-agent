from __future__ import annotations

from html import escape
from pathlib import Path
import json

import pandas as pd


def write_report(path: str | Path, metadata: dict, summaries: dict, optimization: dict, events: pd.DataFrame) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = "".join(
        f"<tr><td>{escape(str(key))}</td><td>{escape(str(value))}</td></tr>"
        for key, value in summaries.items()
    )
    event_preview = events.head(100).to_html(index=False, border=0, classes="events") if not events.empty else "<p>No events detected.</p>"
    payload = escape(json.dumps(optimization, indent=2, default=str))
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BTC Bubble Model Results</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;color:#18212f}}h1{{margin-bottom:4px}}.note{{background:#fff6d8;padding:12px;border-left:4px solid #e1a700}}table{{border-collapse:collapse;width:100%;margin:16px 0}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}pre{{background:#f5f7fa;padding:16px;overflow:auto}}</style>
</head><body><h1>BTC Bubble Model Results</h1>
<p>{escape(str(metadata.get('exchange')))} · {escape(str(metadata.get('product')))} · {escape(str(metadata.get('date')))}</p>
<p class="note">Research only. No orders were placed. Depth metrics are exact only when synchronized 10 bp L2 data is present.</p>
<h2>Summary</h2><table>{rows}</table><h2>Walk-forward selection</h2><pre>{payload}</pre>
<h2>Event preview</h2>{event_preview}</body></html>"""
    target.write_text(html, encoding="utf-8")
    return target

