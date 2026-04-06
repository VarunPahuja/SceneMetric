"""
run_pipeline.py — SceneMetric Full Pipeline
Runs Module 1 (shot scale classification) + Module 2 (visual story)
and prints a clean report you can show your teacher.
"""

import torch
import cv2
import numpy as np
from torchvision import transforms, models
import torch.nn as nn
from collections import Counter
from modules.visual_story import analyse_video

# ── CONFIG ───────────────────────────────────────────────────────────────────
VIDEO_PATH  = "local_videos/2.mp4"
MODEL_PATH  = "models/resnet18_revamped_f109451_20260405_131846.pt"
CLASS_NAMES = ["CLOSE", "MEDIUM", "WIDE"]
IMG_SIZE    = 224
WINDOW_SEC  = 2.0
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── LOAD MODULE 1 MODEL ──────────────────────────────────────────────────────
print("Loading Module 1 model...")
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
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── MODULE 1: CLASSIFY EVERY FRAME ───────────────────────────────────────────
print(f"Running Module 1 on: {VIDEO_PATH}\n")
cap    = cv2.VideoCapture(VIDEO_PATH)
fps    = cap.get(cv2.CAP_PROP_FPS)
total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
labels = []
confidences = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inp = transform(rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(inp)
        probs  = torch.softmax(logits, dim=1)[0]
        pred   = probs.argmax().item()

    pred_class = CLASS_NAMES[pred]
    conf       = probs[pred].item()

    # Indoor bias: low confidence + WIDE -> downgrade to MEDIUM
    if conf < 0.70 and pred_class == "WIDE":
        pred_class = "MEDIUM"

    labels.append(pred_class)
    confidences.append(conf)

cap.release()

# ── MODULE 1 RESULTS ─────────────────────────────────────────────────────────
counts     = Counter(labels)
total_f    = len(labels)
avg_conf   = sum(confidences) / len(confidences)

print("=" * 60)
print("  MODULE 1 — Shot Scale Classification Results")
print("=" * 60)
print(f"  Video        : {VIDEO_PATH}")
print(f"  Total frames : {total_f}")
print(f"  FPS          : {fps:.1f}")
print(f"  Duration     : {total_f/fps:.1f}s")
print(f"  Avg confidence: {avg_conf*100:.1f}%")
print()
print(f"  {'Class':<10} {'Frames':>8} {'%':>8}  Bar")
print(f"  {'-'*45}")
for cls in CLASS_NAMES:
    n    = counts.get(cls, 0)
    pct  = n / total_f * 100
    bar  = "█" * int(pct / 2)
    print(f"  {cls:<10} {n:>8} {pct:>7.1f}%  {bar}")

# Per-segment timeline (5s windows for readability)
print()
print(f"  Shot Timeline (every 5 seconds)")
print(f"  {'-'*45}")
window = int(fps * 5)
for start in range(0, total_f, window):
    chunk     = labels[start: start + window]
    if not chunk:
        continue
    dominant  = Counter(chunk).most_common(1)[0][0]
    end_frame = min(start + window, total_f)
    ts_start  = start / fps
    ts_end    = end_frame / fps
    bar_map   = {"CLOSE": "🟥", "MEDIUM": "🟨", "WIDE": "🟦"}
    icon      = bar_map.get(dominant, "⬜")
    print(f"  {ts_start:>5.1f}s – {ts_end:>5.1f}s   {icon}  {dominant}")

print()

# ── MODULE 2: VISUAL STORY ───────────────────────────────────────────────────
print("=" * 60)
print("  MODULE 2 — Visual Story Generator")
print("=" * 60)

labels_sampled = labels[::2]

result = analyse_video(
    video_path   = VIDEO_PATH,
    shot_labels  = labels_sampled,
    window_sec   = WINDOW_SEC,
    sample_every = 2,
)

# Segment table
print(f"\n  Scene Segments")
print(f"  {'-'*65}")
print(f"  {'Seg':>4}  {'Time':>13}  {'Tag':>12}  {'Scale':>7}  "
      f"{'Motion':>7}  {'Edges':>7}  {'Sym':>6}")
print(f"  {'-'*65}")
for s in result.segments:
    print(f"  {s.segment_idx:>4}  "
          f"{s.start_sec:>5.1f}s-{s.end_sec:>5.1f}s  "
          f"{s.dominant_tag:>12}  {s.shot_scale:>7}  "
          f"{s.avg_motion:>7.3f}  {s.avg_edges:>7.3f}  "
          f"{s.avg_symmetry:>6.3f}")

# Transitions
print(f"\n  Transitions Detected")
print(f"  {'-'*65}")
if result.transitions:
    for t in result.transitions:
        print(f"  {t}")
else:
    print("  No significant transitions detected.")

# Narrative
print(f"\n  Narrative")
print(f"  {'-'*65}")
if result.narrative:
    for line in result.narrative.split("\n"):
        print(f"  {line}")
else:
    print("  [No narrative generated — check visual_story.py generate_narrative() return statement]")

print("\n" + "=" * 60)
print("  Pipeline complete.")
print("=" * 60)