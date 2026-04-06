"""
generate_report.py — SceneMetric HTML Report Generator
Run AFTER run_pipeline.py, or import generate_html_report() directly.

Usage:
    python generate_report.py
    
Or call from run_pipeline.py:
    from generate_report import generate_html_report
    generate_html_report(labels, confidences, fps, result, VIDEO_PATH)
"""

import os
import webbrowser
from collections import Counter
from datetime import datetime
import numpy as np


def generate_html_report(labels, confidences, fps, result, video_path,
                         output_path="outputs/scenemetric_report.html"):
    """
    Generates a cinematic HTML report from pipeline results.
    
    Args:
        labels:       list of per-frame shot scale predictions (Module 1)
        confidences:  list of per-frame confidence scores
        fps:          video fps
        result:       NarrativeResult from Module 2
        video_path:   path to source video
        output_path:  where to save the HTML file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    total_f      = len(labels)
    counts       = Counter(labels)
    avg_conf     = sum(confidences) / len(confidences) * 100
    duration_sec = total_f / fps
    mins         = int(duration_sec // 60)
    secs         = int(duration_sec  % 60)
    duration_str = f"{mins}m {secs:02d}s" if mins > 0 else f"{secs}s"
    video_name   = os.path.basename(video_path)
    timestamp    = datetime.now().strftime("%d %b %Y, %H:%M")

    # ── Module 1: shot distribution ──────────────────────────────────────────
    close_pct  = counts.get("CLOSE",  0) / total_f * 100
    medium_pct = counts.get("MEDIUM", 0) / total_f * 100
    wide_pct   = counts.get("WIDE",   0) / total_f * 100

    # ── Module 1: timeline blocks (every 3s) ─────────────────────────────────
    window      = max(1, int(fps * 3))
    timeline_html = ""
    for start in range(0, total_f, window):
        chunk    = labels[start: start + window]
        if not chunk:
            continue
        dominant = Counter(chunk).most_common(1)[0][0]
        ts       = start / fps
        color_map = {"CLOSE": "#e85d26", "MEDIUM": "#f0a500", "WIDE": "#3b9ede"}
        color     = color_map.get(dominant, "#555")
        conf_chunk = confidences[start: start + window]
        avg_c      = sum(conf_chunk) / len(conf_chunk) * 100
        timeline_html += (
            f'<div class="tl-block" style="background:{color};opacity:{0.4 + avg_c/200:.2f}" '
            f'title="{dominant} @ {ts:.1f}s ({avg_c:.0f}%)"></div>'
        )

    # ── Module 2: segment rows ────────────────────────────────────────────────
    tag_colors = {
        "Symmetrical": "#3b9ede",
        "Dynamic":     "#e85d26",
        "Dramatic":    "#c0392b",
        "Minimal":     "#95a5a6",
        "Neutral":     "#7f8c8d",
    }
    scale_icons = {"CLOSE": "◉", "MEDIUM": "◎", "WIDE": "○", "UNKNOWN": "·"}

    seg_rows = ""
    for s in result.segments:
        tag_color  = tag_colors.get(s.dominant_tag, "#888")
        scale_icon = scale_icons.get(s.shot_scale, "·")
        motion_bar = int(s.avg_motion * 800)
        sym_bar    = int(s.avg_symmetry * 100)
        edge_bar   = int(s.avg_edges * 1000)
        seg_rows += f"""
        <tr>
            <td class="mono dim">{s.segment_idx:02d}</td>
            <td class="mono">{s.start_sec:.1f}s – {s.end_sec:.1f}s</td>
            <td><span class="tag-pill" style="border-color:{tag_color};color:{tag_color}">{s.dominant_tag}</span></td>
            <td class="mono">{scale_icon} {s.shot_scale}</td>
            <td>
                <div class="mini-bar-wrap">
                    <div class="mini-bar" style="width:{min(motion_bar,100)}%;background:#e85d26"></div>
                </div>
                <span class="mono dim">{s.avg_motion:.3f}</span>
            </td>
            <td>
                <div class="mini-bar-wrap">
                    <div class="mini-bar" style="width:{min(edge_bar,100)}%;background:#f0a500"></div>
                </div>
                <span class="mono dim">{s.avg_edges:.3f}</span>
            </td>
            <td>
                <div class="mini-bar-wrap">
                    <div class="mini-bar" style="width:{sym_bar}%;background:#3b9ede"></div>
                </div>
                <span class="mono dim">{s.avg_symmetry:.3f}</span>
            </td>
        </tr>"""

    # ── Transitions ───────────────────────────────────────────────────────────
    trans_html = ""
    for t in result.transitions:
        # parse "  [Xs → Ys]  Label" format
        parts = t.strip()
        trans_html += f'<div class="transition-row"><span class="mono dim">{parts}</span></div>'

    if not result.transitions:
        trans_html = '<div class="transition-row dim">No significant transitions detected.</div>'

    # ── Narrative ─────────────────────────────────────────────────────────────
    narrative_html = ""
    for line in result.narrative.split("\n"):
        line = line.strip()
        if not line:
            narrative_html += "<br>"
        elif line.startswith("  "):
            narrative_html += f'<div class="narrative-indent mono">{line}</div>'
        else:
            narrative_html += f'<p class="narrative-line">{line}</p>'

    # ── Full HTML ─────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SceneMetric — {video_name}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=IBM+Plex+Mono:wght@300;400;500&family=Crimson+Pro:ital,wght@0,300;0,400;1,300&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg:        #0a0a0a;
    --bg2:       #111111;
    --bg3:       #181818;
    --border:    #242424;
    --text:      #d4cfc8;
    --dim:       #5a5550;
    --close:     #e85d26;
    --medium:    #f0a500;
    --wide:      #3b9ede;
    --accent:    #f0a500;
    --font-head: 'Bebas Neue', sans-serif;
    --font-mono: 'IBM Plex Mono', monospace;
    --font-body: 'Crimson Pro', serif;
  }}

  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-body);
    font-size: 16px;
    line-height: 1.6;
    min-height: 100vh;
  }}

  /* grain overlay */
  body::before {{
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.04'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 1000;
    opacity: 0.4;
  }}

  .container {{
    max-width: 1100px;
    margin: 0 auto;
    padding: 0 2rem 4rem;
  }}

  /* ── HEADER ── */
  .header {{
    border-bottom: 1px solid var(--border);
    padding: 3rem 0 2rem;
    margin-bottom: 3rem;
    display: grid;
    grid-template-columns: 1fr auto;
    align-items: end;
    gap: 1rem;
  }}

  .logo {{
    font-family: var(--font-head);
    font-size: 4.5rem;
    letter-spacing: 0.12em;
    color: #fff;
    line-height: 1;
  }}

  .logo span {{ color: var(--accent); }}

  .header-meta {{
    text-align: right;
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--dim);
    line-height: 1.8;
  }}

  .header-meta strong {{
    color: var(--text);
    display: block;
    font-size: 0.85rem;
  }}

  /* ── FILMSTRIP TIMELINE ── */
  .filmstrip-wrap {{
    margin-bottom: 3rem;
  }}

  .section-label {{
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    color: var(--dim);
    text-transform: uppercase;
    margin-bottom: 0.75rem;
  }}

  .filmstrip {{
    display: flex;
    height: 48px;
    border-radius: 3px;
    overflow: hidden;
    gap: 1px;
    background: var(--border);
    padding: 1px;
  }}

  .tl-block {{
    flex: 1;
    min-width: 2px;
    border-radius: 1px;
    transition: opacity 0.2s;
    cursor: default;
  }}

  .tl-block:hover {{ opacity: 1 !important; }}

  .tl-legend {{
    display: flex;
    gap: 1.5rem;
    margin-top: 0.75rem;
    font-family: var(--font-mono);
    font-size: 0.7rem;
  }}

  .legend-dot {{
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 5px;
    vertical-align: middle;
  }}

  /* ── STATS GRID ── */
  .stats-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 3rem;
  }}

  .stat-card {{
    background: var(--bg2);
    padding: 1.5rem;
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
    transition: background 0.2s;
  }}

  .stat-card:hover {{ background: var(--bg3); }}

  .stat-label {{
    font-family: var(--font-mono);
    font-size: 0.6rem;
    letter-spacing: 0.18em;
    color: var(--dim);
    text-transform: uppercase;
  }}

  .stat-value {{
    font-family: var(--font-head);
    font-size: 2.2rem;
    letter-spacing: 0.05em;
    line-height: 1;
    color: #fff;
  }}

  .stat-sub {{
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--dim);
  }}

  /* ── DISTRIBUTION BARS ── */
  .dist-section {{
    margin-bottom: 3rem;
  }}

  .dist-row {{
    display: grid;
    grid-template-columns: 80px 1fr 60px;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.6rem;
  }}

  .dist-label {{
    font-family: var(--font-mono);
    font-size: 0.7rem;
    letter-spacing: 0.1em;
  }}

  .dist-bar-bg {{
    height: 6px;
    background: var(--bg3);
    border-radius: 3px;
    overflow: hidden;
  }}

  .dist-bar-fill {{
    height: 100%;
    border-radius: 3px;
    transition: width 1s cubic-bezier(.16,1,.3,1);
  }}

  .dist-pct {{
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--dim);
    text-align: right;
  }}

  /* ── TWO COLUMN ── */
  .two-col {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
    margin-bottom: 3rem;
  }}

  /* ── PANELS ── */
  .panel {{
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
  }}

  .panel-header {{
    padding: 0.75rem 1.25rem;
    border-bottom: 1px solid var(--border);
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    color: var(--dim);
    text-transform: uppercase;
    background: var(--bg3);
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }}

  .panel-header::before {{
    content: '';
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--accent);
  }}

  .panel-body {{
    padding: 1.25rem;
  }}

  /* ── SEGMENTS TABLE ── */
  .seg-table-wrap {{
    margin-bottom: 3rem;
  }}

  table {{
    width: 100%;
    border-collapse: collapse;
    font-family: var(--font-mono);
    font-size: 0.7rem;
  }}

  thead tr {{
    border-bottom: 1px solid var(--border);
  }}

  th {{
    padding: 0.5rem 0.75rem;
    text-align: left;
    color: var(--dim);
    font-weight: 400;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-size: 0.6rem;
  }}

  tbody tr {{
    border-bottom: 1px solid var(--border);
    transition: background 0.15s;
  }}

  tbody tr:hover {{ background: var(--bg3); }}

  td {{
    padding: 0.5rem 0.75rem;
    vertical-align: middle;
  }}

  .tag-pill {{
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 2px;
    border: 1px solid;
    font-size: 0.6rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
  }}

  .mini-bar-wrap {{
    width: 60px;
    height: 3px;
    background: var(--bg3);
    border-radius: 2px;
    overflow: hidden;
    display: inline-block;
    vertical-align: middle;
    margin-right: 4px;
  }}

  .mini-bar {{
    height: 100%;
    border-radius: 2px;
  }}

  /* ── TRANSITIONS ── */
  .transition-row {{
    padding: 0.4rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.75rem;
    line-height: 1.4;
  }}

  .transition-row:last-child {{ border-bottom: none; }}

  /* ── NARRATIVE ── */
  .narrative-block {{
    margin-bottom: 3rem;
  }}

  .narrative-line {{
    font-family: var(--font-body);
    font-size: 1.05rem;
    line-height: 1.8;
    color: var(--text);
    margin-bottom: 0.5rem;
    font-weight: 300;
  }}

  .narrative-indent {{
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--dim);
    padding-left: 1.5rem;
    margin-bottom: 0.25rem;
    border-left: 2px solid var(--border);
  }}

  /* ── UTILS ── */
  .mono  {{ font-family: var(--font-mono); font-size: 0.72rem; }}
  .dim   {{ color: var(--dim); }}
  .accent {{ color: var(--accent); }}

  .module-badge {{
    display: inline-block;
    font-family: var(--font-mono);
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    padding: 0.2rem 0.6rem;
    border: 1px solid var(--accent);
    color: var(--accent);
    border-radius: 2px;
    margin-bottom: 1rem;
  }}

  .divider {{
    border: none;
    border-top: 1px solid var(--border);
    margin: 2.5rem 0;
  }}

  /* ── FOOTER ── */
  .footer {{
    border-top: 1px solid var(--border);
    padding-top: 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--dim);
  }}

  /* ── ANIMATIONS ── */
  @keyframes fadeUp {{
    from {{ opacity: 0; transform: translateY(16px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
  }}

  .anim {{ animation: fadeUp 0.5s ease both; }}
  .anim-1 {{ animation-delay: 0.05s; }}
  .anim-2 {{ animation-delay: 0.12s; }}
  .anim-3 {{ animation-delay: 0.20s; }}
  .anim-4 {{ animation-delay: 0.28s; }}
  .anim-5 {{ animation-delay: 0.36s; }}
  .anim-6 {{ animation-delay: 0.44s; }}

  @media (max-width: 768px) {{
    .stats-grid {{ grid-template-columns: repeat(2, 1fr); }}
    .two-col    {{ grid-template-columns: 1fr; }}
    .logo       {{ font-size: 3rem; }}
  }}
</style>
</head>
<body>
<div class="container">

  <!-- HEADER -->
  <header class="header anim">
    <div>
      <div class="logo">Scene<span>Metric</span></div>
      <div class="mono dim" style="margin-top:0.5rem">Cinematic Visual Analysis Pipeline</div>
    </div>
    <div class="header-meta">
      <strong>{video_name}</strong>
      {timestamp}<br>
      Module 1 · Module 2
    </div>
  </header>

  <!-- FILMSTRIP TIMELINE -->
  <div class="filmstrip-wrap anim anim-1">
    <div class="section-label">Shot Scale Timeline — every 3 seconds</div>
    <div class="filmstrip">
      {timeline_html}
    </div>
    <div class="tl-legend">
      <span><span class="legend-dot" style="background:var(--close)"></span>CLOSE</span>
      <span><span class="legend-dot" style="background:var(--medium)"></span>MEDIUM</span>
      <span><span class="legend-dot" style="background:var(--wide)"></span>WIDE</span>
      <span class="dim">hover for details</span>
    </div>
  </div>

  <!-- STATS GRID -->
  <div class="stats-grid anim anim-2">
    <div class="stat-card">
      <div class="stat-label">Duration</div>
      <div class="stat-value accent">{duration_str}</div>
      <div class="stat-sub">{total_f} frames · {fps:.0f}fps</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Avg Confidence</div>
      <div class="stat-value">{avg_conf:.0f}<span style="font-size:1.2rem">%</span></div>
      <div class="stat-sub">Module 1 · ResNet18</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Segments</div>
      <div class="stat-value">{len(result.segments)}</div>
      <div class="stat-sub">Module 2 · 2s windows</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Transitions</div>
      <div class="stat-value">{len(result.transitions)}</div>
      <div class="stat-sub">visual narrative shifts</div>
    </div>
  </div>

  <!-- MODULE 1 -->
  <div class="anim anim-3">
    <div class="module-badge">MODULE 01 — SHOT SCALE CLASSIFICATION · ResNet18 · Fine-tuned</div>
    <div class="dist-section">
      <div class="dist-row">
        <div class="dist-label" style="color:var(--close)">CLOSE</div>
        <div class="dist-bar-bg">
          <div class="dist-bar-fill" style="width:{close_pct:.1f}%;background:var(--close)"></div>
        </div>
        <div class="dist-pct">{close_pct:.1f}%</div>
      </div>
      <div class="dist-row">
        <div class="dist-label" style="color:var(--medium)">MEDIUM</div>
        <div class="dist-bar-bg">
          <div class="dist-bar-fill" style="width:{medium_pct:.1f}%;background:var(--medium)"></div>
        </div>
        <div class="dist-pct">{medium_pct:.1f}%</div>
      </div>
      <div class="dist-row">
        <div class="dist-label" style="color:var(--wide)">WIDE</div>
        <div class="dist-bar-bg">
          <div class="dist-bar-fill" style="width:{wide_pct:.1f}%;background:var(--wide)"></div>
        </div>
        <div class="dist-pct">{wide_pct:.1f}%</div>
      </div>
    </div>
  </div>

  <hr class="divider">

  <!-- MODULE 2 -->
  <div class="anim anim-4">
    <div class="module-badge">MODULE 02 — VISUAL STORY GENERATOR · Classical CV · Rule-based</div>
  </div>

  <!-- SEGMENTS TABLE -->
  <div class="seg-table-wrap panel anim anim-4">
    <div class="panel-header">Scene Segments — Classical CV Signals</div>
    <div style="overflow-x:auto">
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>Time</th>
          <th>Style Tag</th>
          <th>Shot Scale</th>
          <th>Motion (Optical Flow)</th>
          <th>Edges (Canny)</th>
          <th>Symmetry</th>
        </tr>
      </thead>
      <tbody>
        {seg_rows}
      </tbody>
    </table>
    </div>
  </div>

  <!-- TRANSITIONS + NARRATIVE -->
  <div class="two-col anim anim-5">
    <div class="panel">
      <div class="panel-header">Cinematic Transitions</div>
      <div class="panel-body" style="max-height:320px;overflow-y:auto">
        {trans_html}
      </div>
    </div>
    <div class="panel">
      <div class="panel-header">Narrative Analysis</div>
      <div class="panel-body narrative-block" style="max-height:320px;overflow-y:auto">
        {narrative_html}
      </div>
    </div>
  </div>

  <!-- FOOTER -->
  <footer class="footer anim anim-6">
    <span>SceneMetric · Computer Vision Project</span>
    <span>Module 1: CNN Shot Classification · Module 2: Classical CV Narrative Synthesis</span>
    <span class="accent">F1: 0.9451</span>
  </footer>

</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n✅ Report saved → {os.path.abspath(output_path)}")
    return output_path


# ── Standalone runner (imports from run_pipeline results) ────────────────────
if __name__ == "__main__":
    import torch
    import cv2
    from torchvision import transforms, models
    import torch.nn as nn
    from modules.visual_story import analyse_video

    VIDEO_PATH  = "local_videos/2.mp4"
    MODEL_PATH  = "models/resnet18_revamped_f109451_20260405_131846.pt"
    CLASS_NAMES = ["CLOSE", "MEDIUM", "WIDE"]
    IMG_SIZE    = 224
    WINDOW_SEC  = 2.0
    DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, 3)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE).eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    print(f"Running Module 1 on {VIDEO_PATH}...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    labels, confidences = [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp  = transform(rgb).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(inp)
            probs  = torch.softmax(logits, dim=1)[0]
            pred   = probs.argmax().item()
        pred_class = CLASS_NAMES[pred]
        conf       = probs[pred].item()
        if conf < 0.70 and pred_class == "WIDE":
            pred_class = "MEDIUM"
        labels.append(pred_class)
        confidences.append(conf)
    cap.release()
    print(f"Module 1 done — {len(labels)} frames.")

    print("Running Module 2...")
    labels_sampled = labels[::2]
    result = analyse_video(
        video_path   = VIDEO_PATH,
        shot_labels  = labels_sampled,
        window_sec   = WINDOW_SEC,
        sample_every = 2,
    )

    path = generate_html_report(labels, confidences, fps, result, VIDEO_PATH)
    webbrowser.open(f"file:///{os.path.abspath(path)}")