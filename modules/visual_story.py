"""
visual_story.py  —  SceneMetric Module 2
========================================
Takes a video + Module 1 shot-scale predictions and applies classical
computer vision to perform rule-based narrative synthesis.

Classical CV techniques used:
  1. Frame Differencing       → motion intensity
  2. Farneback Optical Flow   → motion direction + magnitude
  3. Canny Edge Detection     → scene complexity / busyness
  4. Symmetry Analysis        → compositional balance
  5. Brightness & Contrast    → mood / tone
  6. Contour Centrality       → subject position (isolated vs engaged)

Usage:
    from modules.visual_story import analyse_video

    result = analyse_video(
        video_path   = "local_videos/sample.mp4",
        shot_labels  = ["WIDE", "WIDE", "CLOSE", ...],  # from Module 1
        window_sec   = 3.0,   # seconds per scene segment
    )
    print(result["narrative"])

    # Or run directly:
    python modules/visual_story.py
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from collections import Counter


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FrameSignals:
    """All classical-CV signals extracted for a single frame."""
    frame_idx:       int
    timestamp_sec:   float
    shot_scale:      str        # from Module 1: CLOSE / MEDIUM / WIDE

    # Classical CV outputs
    motion_score:    float      # 0–1  (frame differencing magnitude)
    flow_magnitude:  float      # 0–1  (Farneback optical flow mean)
    edge_density:    float      # 0–1  (Canny edges / total pixels)
    symmetry_score:  float      # 0–1  (1 = perfect mirror symmetry)
    brightness:      float      # 0–1  (mean pixel intensity)
    contrast:        float      # 0–1  (std of pixel intensity)
    centrality:      float      # 0–1  (1 = subject dead-centre)

    # Derived tag
    style_tag:       str = ""   # Symmetrical / Dynamic / Minimal / Dramatic


@dataclass
class SceneSegment:
    """A time window of frames collapsed into a single scene description."""
    segment_idx:   int
    start_sec:     float
    end_sec:       float
    dominant_tag:  str
    shot_scale:    str          # majority shot scale in this segment
    avg_motion:    float
    avg_edges:     float
    avg_symmetry:  float
    avg_brightness: float


@dataclass
class NarrativeResult:
    """Final output of Module 2."""
    segments:        List[SceneSegment]
    transitions:     List[str]
    narrative:       str
    per_frame:       List[FrameSignals] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL EXTRACTION  (pure classical CV)
# ─────────────────────────────────────────────────────────────────────────────

def _motion_from_diff(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """
    Frame Differencing — subtracts consecutive grayscale frames.
    High difference = lots of movement between frames.
    Returns normalised score 0–1.
    """
    diff = cv2.absdiff(prev_gray, curr_gray)
    return float(diff.mean()) / 255.0


def _optical_flow_magnitude(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """
    Farneback Dense Optical Flow — estimates per-pixel velocity vectors.
    We take the mean vector magnitude across the frame.
    High = fast / large motion; Low = static shot.
    Returns normalised score 0–1 (clipped at reasonable max ~10px/frame).
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        flow=None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mean_mag = float(magnitude.mean())
    return min(mean_mag / 10.0, 1.0)   # clip: 10px/frame ≈ very fast motion


def _edge_density(gray: np.ndarray) -> float:
    """
    Canny Edge Detection — finds boundaries in the image.
    High edge density = busy, complex, detailed frame.
    Low edge density  = sparse, minimal, clean frame.
    Returns fraction of pixels that are edges (0–1).
    """
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    return float(edges.sum() / 255) / float(gray.size)


def _symmetry_score(gray: np.ndarray) -> float:
    """
    Symmetry Analysis — flips the frame horizontally and compares to original.
    High score = balanced, mirror-like composition (order, control).
    Low score  = off-centre, asymmetric composition (tension, chaos).
    Returns normalised similarity 0–1.
    """
    flipped = cv2.flip(gray, 1)
    diff    = cv2.absdiff(gray, flipped)
    # SSIM-lite: 1 - (mean_diff / 255)
    return 1.0 - float(diff.mean()) / 255.0


def _brightness_contrast(gray: np.ndarray):
    """
    Histogram Statistics — mean and std of pixel intensities.
    Bright + high contrast = dramatic / intense mood.
    Dark + low contrast    = moody / flat / oppressive.
    Both normalised 0–1.
    """
    mean = float(gray.mean()) / 255.0
    std  = float(gray.std())  / 128.0   # std rarely exceeds 128 in practice
    return mean, min(std, 1.0)


def _subject_centrality(gray: np.ndarray) -> float:
    """
    Contour Centrality — finds the largest contour (dominant subject) and
    measures how close its centroid is to the frame centre.
    High = subject dead-centre (isolated, confrontational).
    Low  = subject off-centre (environmental, context-heavy).
    Returns 0–1.
    """
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.5   # neutral if nothing found

    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return 0.5

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    h, w = gray.shape

    # Distance from frame centre, normalised — closer = higher score
    dist = np.sqrt(((cx - w / 2) / w) ** 2 + ((cy - h / 2) / h) ** 2)
    return float(1.0 - min(dist * 2, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# FRAME TAGGING  (maps signals → narrative style tag)
# ─────────────────────────────────────────────────────────────────────────────

def _tag_frame(sig: FrameSignals) -> str:
    """
    Rule-based tagging — converts 6 CV signals into one human-readable tag.

    Priority order (highest signal wins):
      Dynamic    → high motion (optical flow or frame diff)
      Dramatic   → high contrast + low symmetry + CLOSE shot
      Symmetrical→ high symmetry + low motion
      Minimal    → low edges + low motion + WIDE shot
      Neutral    → fallback
    """
    motion   = max(sig.motion_score, sig.flow_magnitude)
    edges    = sig.edge_density
    sym      = sig.symmetry_score
    contrast = sig.contrast
    scale    = sig.shot_scale

    if motion > 0.25:
        return "Dynamic"

    if contrast > 0.55 and sym < 0.60 and scale == "CLOSE":
        return "Dramatic"

    if sym > 0.72 and motion < 0.15:
        return "Symmetrical"

    if edges < 0.06 and motion < 0.12 and scale == "WIDE":
        return "Minimal"

    return "Neutral"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXTRACTION LOOP
# ─────────────────────────────────────────────────────────────────────────────

def extract_frame_signals(
    video_path:  str,
    shot_labels: Optional[List[str]] = None,
    sample_every: int = 1,
) -> List[FrameSignals]:
    """
    Runs all classical CV extractors over every sampled frame.

    Args:
        video_path:   path to video file
        shot_labels:  list of Module 1 predictions (one per frame).
                      If None, defaults to "UNKNOWN" for all frames.
        sample_every: process every Nth frame (1 = all frames)

    Returns:
        List of FrameSignals, one per sampled frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps         = cap.get(cv2.CAP_PROP_FPS)
    total       = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    prev_gray   = None
    results     = []
    frame_idx   = 0

    print(f"[Module 2] Extracting signals from: {video_path}")
    print(f"           {total} frames @ {fps:.1f} fps | sampling every {sample_every} frame(s)\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every != 0:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 180))   # downsample for speed

        # Shot scale from Module 1 (or fallback)
        if shot_labels and frame_idx < len(shot_labels):
            scale = shot_labels[frame_idx]
        else:
            scale = "UNKNOWN"

        # ── Classical CV signals ──────────────────────────────────────────
        if prev_gray is not None:
            motion  = _motion_from_diff(prev_gray, gray)
            flow    = _optical_flow_magnitude(prev_gray, gray)
        else:
            motion, flow = 0.0, 0.0

        edges       = _edge_density(gray)
        symmetry    = _symmetry_score(gray)
        bright, con = _brightness_contrast(gray)
        centrality  = _subject_centrality(gray)
        # ─────────────────────────────────────────────────────────────────

        sig = FrameSignals(
            frame_idx      = frame_idx,
            timestamp_sec  = frame_idx / fps,
            shot_scale     = scale,
            motion_score   = motion,
            flow_magnitude = flow,
            edge_density   = edges,
            symmetry_score = symmetry,
            brightness     = bright,
            contrast       = con,
            centrality     = centrality,
        )
        sig.style_tag = _tag_frame(sig)
        results.append(sig)

        prev_gray = gray
        frame_idx += 1

    cap.release()
    print(f"[Module 2] Processed {len(results)} frames.\n")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# TEMPORAL SEGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def segment_timeline(
    signals:    List[FrameSignals],
    window_sec: float = 3.0,
    fps:        float = 30.0,
) -> List[SceneSegment]:
    """
    Groups frames into fixed-length time windows.
    Each window is summarised by majority vote on style tag and shot scale,
    and mean values of all CV signals.
    """
    if not signals:
        return []

    window_frames = max(1, int(window_sec * fps))
    segments      = []
    seg_idx       = 0

    for start in range(0, len(signals), window_frames):
        chunk = signals[start: start + window_frames]
        if not chunk:
            continue

        tags   = [f.style_tag  for f in chunk]
        scales = [f.shot_scale for f in chunk]

        dominant_tag = Counter(tags).most_common(1)[0][0]
        shot_scale   = Counter(scales).most_common(1)[0][0]

        seg = SceneSegment(
            segment_idx    = seg_idx,
            start_sec      = chunk[0].timestamp_sec,
            end_sec        = chunk[-1].timestamp_sec,
            dominant_tag   = dominant_tag,
            shot_scale     = shot_scale,
            avg_motion     = float(np.mean([f.motion_score    for f in chunk])),
            avg_edges      = float(np.mean([f.edge_density    for f in chunk])),
            avg_symmetry   = float(np.mean([f.symmetry_score  for f in chunk])),
            avg_brightness = float(np.mean([f.brightness      for f in chunk])),
        )
        segments.append(seg)
        seg_idx += 1

    return segments


# ─────────────────────────────────────────────────────────────────────────────
# TRANSITION DETECTION
# ─────────────────────────────────────────────────────────────────────────────

# Human-readable transition descriptions
TRANSITION_MAP = {
    ("Symmetrical", "Dynamic"):    "Controlled → Chaos",
    ("Symmetrical", "Dramatic"):   "Calm → Intensity",
    ("Symmetrical", "Minimal"):    "Order → Emptiness",
    ("Dynamic",     "Symmetrical"):"Chaos → Control",
    ("Dynamic",     "Minimal"):    "Action → Isolation",
    ("Dynamic",     "Dramatic"):   "Movement → Confrontation",
    ("Minimal",     "Dynamic"):    "Stillness → Eruption",
    ("Minimal",     "Dramatic"):   "Emptiness → Intensity",
    ("Minimal",     "Symmetrical"):"Isolation → Balance",
    ("Dramatic",    "Minimal"):    "Intensity → Emptiness",
    ("Dramatic",    "Dynamic"):    "Confrontation → Chaos",
    ("Dramatic",    "Symmetrical"):"Emotion → Resolution",
    ("Neutral",     "Dynamic"):    "Calm → Action",
    ("Dynamic",     "Neutral"):    "Action → Calm",
}

def detect_transitions(segments: List[SceneSegment]) -> List[str]:
    """
    Scans adjacent segments for style-tag changes.
    Maps each change to a human-readable narrative transition.
    """
    transitions = []
    for i in range(1, len(segments)):
        prev = segments[i - 1].dominant_tag
        curr = segments[i].dominant_tag
        if prev != curr:
            label = TRANSITION_MAP.get(
                (prev, curr),
                f"{prev} → {curr}"
            )
            t = (f"  [{segments[i-1].start_sec:.1f}s → {segments[i].end_sec:.1f}s]  "
                 f"{label}")
            transitions.append(t)
    return transitions


# ─────────────────────────────────────────────────────────────────────────────
# NARRATIVE GENERATION  (rule-based template synthesis)
# ─────────────────────────────────────────────────────────────────────────────

# Opening sentence templates per tag
OPENING = {
    "Symmetrical": "The scene opens with a controlled, balanced composition — a sense of order and calm.",
    "Dynamic":     "The scene opens in motion — restless, kinetic, with visual instability from the first frame.",
    "Minimal":     "The scene opens sparsely, with wide framing and an almost empty visual field — a feeling of isolation.",
    "Dramatic":    "The scene opens with intensity — tight framing and high contrast creating immediate emotional pressure.",
    "Neutral":     "The scene opens in a measured, unremarkable visual register.",
}

# Middle sentence templates
MIDDLE = {
    "Symmetrical": "The middle maintains compositional balance, suggesting a sustained sense of control.",
    "Dynamic":     "The middle is dominated by movement and visual chaos, building tension through instability.",
    "Minimal":     "The middle strips back visual information — sparse and quiet, emphasising isolation.",
    "Dramatic":    "The middle escalates emotional intensity through close framing and sharp tonal contrast.",
    "Neutral":     "The middle holds a neutral visual register, neither escalating nor retreating.",
}

# Closing sentence templates
CLOSING = {
    "Symmetrical": "The scene resolves into symmetry and balance — a return to order.",
    "Dynamic":     "The scene ends without resolution, still in motion — unsettled and unresolved.",
    "Minimal":     "The scene closes on emptiness — sparse framing that lingers in isolation.",
    "Dramatic":    "The scene ends on a note of emotional confrontation, the tension unbroken.",
    "Neutral":     "The scene closes quietly, without dramatic resolution.",
}

# Shot scale descriptors
SCALE_DESC = {
    "CLOSE":   "intimate, character-focused framing",
    "MEDIUM":  "balanced mid-range framing",
    "WIDE":    "expansive, environment-dominant framing",
    "UNKNOWN": "varied framing",
}


def generate_narrative(segments: List[SceneSegment],
                       transitions: List[str]) -> str:
    """
    Builds a paragraph narrative from segment summaries and transitions.
    Pure rule-based — no LLM, no external API.
    """
    if not segments:
        return "No segments detected — video may be too short."

    n   = len(segments)
    mid = n // 2

    opening_tag = segments[0].dominant_tag
    middle_tag  = segments[mid].dominant_tag
    closing_tag = segments[-1].dominant_tag

    opening_scale = SCALE_DESC.get(segments[0].shot_scale,   "varied framing")
    closing_scale = SCALE_DESC.get(segments[-1].shot_scale,  "varied framing")

    lines = []

    # Opening
    lines.append(OPENING.get(opening_tag, OPENING["Neutral"]))
    lines.append(f"Shot scale: {opening_scale}.")

    # Transitions
    if transitions:
        lines.append("\nKey visual transitions detected:")
        lines.extend(transitions)

    # Middle
    lines.append(f"\n{MIDDLE.get(middle_tag, MIDDLE['Neutral'])}")

    # Closing
    lines.append(CLOSING.get(closing_tag, CLOSING["Neutral"]))
    lines.append(f"Shot scale: {closing_scale}.")

    # One-liner summary
    # REPLACE the arc block at the bottom of generate_narrative() with:
    unique_tags = list(dict.fromkeys([s.dominant_tag for s in segments]))
    if len(unique_tags) == 1:
        arc = f"The scene maintains a consistent {unique_tags[0].lower()} visual register throughout."
    else:
        arc_map = TRANSITION_MAP.get(
            (opening_tag, closing_tag),
            f"{opening_tag} shifting to {closing_tag}"
        )
        arc = f"Overall narrative arc: {arc_map}."
    lines.append(f"\n{arc}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def analyse_video(
    video_path:   str,
    shot_labels:  Optional[List[str]] = None,
    window_sec:   float = 3.0,
    sample_every: int   = 1,
) -> NarrativeResult:
    """
    Full Module 2 pipeline.

    Args:
        video_path:   path to video
        shot_labels:  per-frame predictions from Module 1 (optional)
        window_sec:   length of each scene segment in seconds
        sample_every: process every Nth frame (use 2–5 for speed)

    Returns:
        NarrativeResult with segments, transitions, narrative, and per-frame signals
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    signals     = extract_frame_signals(video_path, shot_labels, sample_every)
    segments    = segment_timeline(signals, window_sec, fps)
    transitions = detect_transitions(segments)
    narrative   = generate_narrative(segments, transitions)

    return NarrativeResult(
        segments    = segments,
        transitions = transitions,
        narrative   = narrative,
        per_frame   = signals,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI — run directly for quick demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    VIDEO = sys.argv[1] if len(sys.argv) > 1 else "local_videos/sample.mp4"

    print("=" * 60)
    print("  SceneMetric — Module 2: Visual Story Generator")
    print("=" * 60)
    print(f"  Video : {VIDEO}")
    print(f"  Note  : Running without Module 1 labels (shot_labels=None)")
    print(f"          Pass shot_labels list for full pipeline output.")
    print("=" * 60 + "\n")

    result = analyse_video(
        video_path   = VIDEO,
        shot_labels  = None,   # replace with Module 1 output for full pipeline
        window_sec   = 3.0,
        sample_every = 2,      # every 2nd frame — fast enough for demo
    )

    # ── Segment Table ────────────────────────────────────────────────────────
    print("SCENE SEGMENTS")
    print("-" * 70)
    print(f"{'Seg':>4}  {'Start':>6}  {'End':>6}  {'Tag':>12}  {'Scale':>7}  "
          f"{'Motion':>7}  {'Edges':>7}  {'Sym':>6}")
    print("-" * 70)
    for s in result.segments:
        print(f"{s.segment_idx:>4}  {s.start_sec:>5.1f}s  {s.end_sec:>5.1f}s  "
              f"{s.dominant_tag:>12}  {s.shot_scale:>7}  "
              f"{s.avg_motion:>7.3f}  {s.avg_edges:>7.3f}  {s.avg_symmetry:>6.3f}")

    # ── Transitions ──────────────────────────────────────────────────────────
    print("\nTRANSITIONS DETECTED")
    print("-" * 70)
    if result.transitions:
        for t in result.transitions:
            print(t)
    else:
        print("  No significant transitions detected.")

    # ── Narrative ────────────────────────────────────────────────────────────
    print("\nNARRATIVE")
    print("-" * 70)
    print(result.narrative)
    print("\n" + "=" * 60)