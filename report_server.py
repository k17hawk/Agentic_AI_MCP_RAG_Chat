#!/usr/bin/env python3
"""
FastAPI server to serve generated HTML reports.
Run with: uvicorn agentic_trading_system.reporting.report_server:app --reload
"""

import os
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn

app = FastAPI(title="Trading Report Server")

# Where reports are stored (matches your structure)
REPORT_DIRS = [
    Path("discovery_outputs"),   # from stg_6_analytics.py
    Path("data/reports"),        # from reporting module
]
STATIC_DIR = Path("static_reports")

def find_all_html_files() -> list:
    """Recursively find all .html files in configured report directories."""
    reports = []
    for base in REPORT_DIRS:
        if not base.exists():
            continue
        for html_file in base.rglob("*.html"):
            if "template" in html_file.name.lower():
                continue
            rel_path = html_file.relative_to(base) if base != html_file.parent else html_file.name
            reports.append({
                "path": str(html_file),
                "name": html_file.name,
                "dir": base.name,
                "modified": datetime.fromtimestamp(html_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "size_kb": round(html_file.stat().st_size / 1024, 1)
            })
    reports.sort(key=lambda x: x["modified"], reverse=True)
    return reports

@app.get("/", response_class=HTMLResponse)
async def index():
    reports = find_all_html_files()
    html = """
    <!DOCTYPE html>
    <html>
    <head><title>Trading Reports</title>
    <style>
        body { font-family: Arial; margin: 40px; background: #f5f7fa; }
        h1 { color: #333; }
        table { width: 100%; border-collapse: collapse; background: white; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #4CAF50; color: white; }
        tr:hover { background-color: #f5f5f5; }
        a { text-decoration: none; color: #2196F3; }
        .badge { background: #e0e0e0; padding: 3px 8px; border-radius: 12px; font-size: 12px; }
    </style>
    </head>
    <body>
        <h1>📊 Trading Performance Reports</h1>
        <table>
            <thead><tr><th>Report</th><th>Directory</th><th>Modified</th><th>Size (KB)</th></tr></thead>
            <tbody>
    """
    for r in reports:
        html += f"<tr><td><a href='/view?path={r['path']}'>{r['name']}</a></td><td>{r['dir']}</td><td>{r['modified']}</td><td>{r['size_kb']}</td></tr>"
    if not reports:
        html += "<tr><td colspan='4'>No HTML reports found. Run stg_6_analytics.py first.</td></tr>"
    html += """
            </tbody>
        </table>
        <p><a href='/latest'>📌 Latest Report</a></p>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.get("/view")
async def view_report(path: str):
    """Serve an HTML report from absolute or relative path."""
    file_path = Path(path)
    if not file_path.exists():
        # Try relative to known report dirs
        for base in REPORT_DIRS:
            candidate = base / path
            if candidate.exists():
                file_path = candidate
                break
        else:
            raise HTTPException(404, "Report not found")
    if not file_path.suffix == '.html':
        raise HTTPException(403, "Only HTML files allowed")
    return FileResponse(file_path, media_type="text/html")

@app.get("/latest")
async def latest():
    reports = find_all_html_files()
    if not reports:
        return HTMLResponse("<h2>No reports found</h2>", 404)
    latest_path = reports[0]['path']
    return HTMLResponse(f'<meta http-equiv="refresh" content="0; url=/view?path={latest_path}">')

@app.get("/health")
async def health():
    return {"status": "ok", "reports_found": len(find_all_html_files())}

if __name__ == "__main__":
    print("🚀 Starting report server at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)