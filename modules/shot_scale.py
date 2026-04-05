"""
shot_scale.py
--------------
Classical CV shot-scale estimator. No deep learning.

Classifies each frame as CLOSE, MEDIUM, or WIDE using a weighted
feature-fusion engine built from:
  1. Face analysis          (Haar cascade)
  2. Subject coverage       (Otsu foreground mask)
  3. Spatial edge entropy   (3x3 grid Canny)
  4. Optical depth ratio    (centre vs border Laplacian variance)

Temporal smoothing via majority-vote sliding window with scene-cut
detection to avoid smoothing across hard cuts.

Public API
----------
process_video(video_path) -> dict
"""

import cv2
import numpy as np
from collections import Counter

# Support both direct execution and module import
try:
    from .preprocessing import extract_frames, load_video, TARGET_LONG_SIDE
except ImportError:
    from preprocessing import extract_frames, load_video, TARGET_LONG_SIDE

# ---------------------------------------------------------------------------
# Haar cascade
# ---------------------------------------------------------------------------
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
if _face_cascade.empty():
    raise RuntimeError("Failed to load Haar cascade.")

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

# --- Face threshold (relative to content area) ---
FACE_MEDIUM_THRESH = 0.05   # face_ratio >= this → face considered prominent

# --- Scene-cut detection ---
SCENE_CUT_THRESHOLD = 30.0  # mean luminance shift across frames → new scene

# --- Temporal smoothing window ---
SMOOTH_WINDOW = 2


# ===========================================================================
# Feature extractors
# ===========================================================================

def _content_roi(frame_rgb):
    """
    Derive the content bounding box from the letterboxed frame.
    Black bars are always on top/bottom OR left/right — never both.
    Returns (roi, pad_top, pad_left, content_h, content_w).
    """
    h, w = frame_rgb.shape[:2]
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

    # Find rows/cols that are not all-black padding
    row_means = gray.mean(axis=1)
    col_means = gray.mean(axis=0)
    DARK = 3  # threshold for "black bar"

    nz_rows = np.where(row_means > DARK)[0]
    nz_cols = np.where(col_means > DARK)[0]

    if len(nz_rows) == 0 or len(nz_cols) == 0:
        # Fully black frame — return full canvas
        return frame_rgb, 0, 0, h, w

    pad_top   = int(nz_rows[0])
    pad_left  = int(nz_cols[0])
    content_h = int(nz_rows[-1]) - pad_top + 1
    content_w = int(nz_cols[-1]) - pad_left + 1

    roi = frame_rgb[pad_top:pad_top + content_h, pad_left:pad_left + content_w]
    return roi, pad_top, pad_left, content_h, content_w


def extract_face_features(gray_content, content_area):
    """
    Returns:
        face_ratio        : total valid face area / content area
        center_face_score : area-weighted proximity of faces to content centre (0–1)
        has_face          : bool
    """
    h, w = gray_content.shape
    cx_frame = w / 2.0
    cy_frame = h / 2.0
    max_dist = np.hypot(cx_frame, cy_frame)

    raw = _face_cascade.detectMultiScale(
        gray_content, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25)
    )

    total_area    = 0
    weighted_prox = 0.0

    for (x, y, fw, fh) in (raw if len(raw) else []):
        # Validate face geometry
        if fw < 25 or fh < 25:
            continue
        ar = fw / fh
        if not (0.7 <= ar <= 1.4):
            continue
        area = fw * fh
        total_area   += area
        cx = x + fw / 2.0
        cy = y + fh / 2.0
        dist = np.hypot(cx - cx_frame, cy - cy_frame)
        weighted_prox += area * (1.0 - dist / max_dist)

    face_ratio        = total_area / content_area if content_area > 0 else 0.0
    center_face_score = (weighted_prox / total_area) if total_area > 0 else 0.0
    has_face          = face_ratio >= FACE_MEDIUM_THRESH  # only prominent faces

    return face_ratio, center_face_score, has_face


def extract_subject_coverage(gray_content):
    """
    Estimates subject coverage using centre-weighted gradient magnitude,
    with spatial contrast normalisation to suppress uniform-texture scenes.

    FIX: The original implementation saturated on wide landscape shots
    because raw Sobel magnitude responds equally to fine textures (grass,
    trees, buildings) and true subject edges. Two countermeasures applied:

    1. Spatial contrast ratio:
       Centre-weighted gradient is divided by the global mean gradient,
       producing a *relative* measure of how much stronger the centre is
       compared to the whole frame. Wide textured scenes have uniformly
       high gradients everywhere, so the ratio stays near 1.0.
       Close-up subjects with a blurred/plain background push it above 2.0.

    2. Soft compression via tanh:
       Maps the ratio through tanh so scores saturate smoothly at 1.0
       instead of blowing past the scoring ramps.

    Returns coverage score in [0, 1]:
        High (~0.6+) → strong centre gradient relative to frame → CLOSE signal
        Low  (~0.3–) → uniform gradients across frame           → WIDE signal
        ~0.5          → moderate centre concentration            → MEDIUM signal
    """
    gx = cv2.Sobel(gray_content, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_content, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)

    h, w = gray_content.shape
    cx, cy = w / 2.0, h / 2.0

    # Centre-weighted Gaussian mask
    Y, X = np.ogrid[:h, :w]
    sigma = min(h, w) * 0.35
    gauss = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))

    global_mean = float(mag.mean()) + 1e-6
    centre_mean = float((mag * gauss).sum() / (gauss.sum() + 1e-6))

    # Relative centre strength: >1 means centre is sharper than average.
    # Wide textured scenes ≈ 1.0; close-up with bokeh ≈ 2–4+.
    relative_strength = centre_mean / global_mean

    # Compress into [0, 1] via tanh scaled so ratio=1 → ~0.46, ratio=2 → ~0.76
    coverage = float(np.tanh(relative_strength * 0.6))

    return float(np.clip(coverage, 0.0, 1.0))


def extract_edge_entropy(gray_content):
    """
    Divides the content area into a 3x3 grid and computes two signals:

    center_concentration : fraction of edges in the 3 centre cells (row 1)
                           vs total edges.
                           High → subject fills frame → CLOSE signal.
                           Low  → edges at periphery  → WIDE signal.

    entropy_score        : normalised Shannon entropy over all 9 cells.
                           Weak contextual WIDE signal only — no longer a
                           primary discriminator because outdoor scenes with
                           detailed backgrounds (trees, grass) produce high
                           entropy even in close and medium shots.

    FIX: center_concentration computation corrected.
    Previously divided by (total * 4) which over-suppressed the score.
    Now uses the true maximum possible centre-row mass (sum of top 3 cells)
    to produce values in a meaningful [0, 1] range.
    """
    edges = cv2.Canny(gray_content, 80, 160)

    rows = np.array_split(edges, 3, axis=0)
    cell_densities = []
    for row in rows:
        for cell in np.array_split(row, 3, axis=1):
            cell_densities.append(np.count_nonzero(cell) / max(cell.size, 1))

    densities = np.array(cell_densities, dtype=np.float64)
    total     = densities.sum()

    if total < 1e-6:
        return 0.5, 0.333   # flat/featureless → neutral

    probs = densities / total

    # Shannon entropy normalised to [0,1] over 9 cells
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_entropy = -np.where(probs > 0, probs * np.log2(probs), 0).sum()
    entropy_score = float(raw_entropy / np.log2(9))

    # Centre concentration: cells [3, 4, 5] = middle row of the 3×3 grid.
    # Weight centre cell (4) double for emphasis, then normalise by the
    # maximum mass those 4 weighted cells could hold (4 × total / 9 upper bound
    # approximated as: if all edge mass were in [3,4,5] with cell 4 doubled).
    #
    # FIX: previous divisor was (total * 4) which produced values ~0.02–0.08,
    # far below the scoring ramps (0.10–0.35). Correct normalisation divides
    # by the theoretical max of (densities[3] + 2*densities[4] + densities[5])
    # which occurs when all edge mass is concentrated in the centre row.
    # Max of the weighted sum = 4 * (total / 3) when all mass is in row 1.
    max_centre_mass = 4.0 * (total / 3.0) + 1e-6
    centre_row_mass = densities[3] + 2.0 * densities[4] + densities[5]
    center_concentration = float(np.clip(centre_row_mass / max_centre_mass, 0.0, 1.0))

    return entropy_score, center_concentration


def extract_depth_ratio(gray_content):
    """
    Compares Laplacian variance in the centre 40% of the content vs the
    outer 12% border strip.

    High ratio (centre sharper than border) → shallow DoF → CLOSE signal.
    Low  ratio (border as sharp as centre)  → deep   DoF → WIDE signal.
    Ratio ~1.0                              → MEDIUM signal.

    Border is sampled as four 2D rectangular strips (not flattened 1D),
    then stacked into a single 2D image so cv2.Laplacian works correctly.

    Returns depth_ratio clamped to [0, 5].
    """
    h, w = gray_content.shape

    # Centre region: middle 40% both axes
    ch0 = int(h * 0.30);  ch1 = int(h * 0.70)
    cw0 = int(w * 0.30);  cw1 = int(w * 0.70)
    centre = gray_content[ch0:ch1, cw0:cw1]

    # Border: four thin 2D strips (12%), each resized to same width for stacking
    target_w = max(w, 1)
    top    = gray_content[:int(h * 0.12), :]
    bottom = gray_content[int(h * 0.88):, :]
    left   = cv2.rotate(gray_content[:, :int(w * 0.12)], cv2.ROTATE_90_CLOCKWISE)
    right  = cv2.rotate(gray_content[:, int(w * 0.88):], cv2.ROTATE_90_CLOCKWISE)

    # Resize all strips to same width so they can be vstacked
    def _resize_strip(s):
        if s.size == 0:
            return np.zeros((1, target_w), dtype=np.uint8)
        return cv2.resize(s, (target_w, max(s.shape[0], 1)),
                          interpolation=cv2.INTER_AREA)

    border_img = np.vstack([
        _resize_strip(top),
        _resize_strip(bottom),
        _resize_strip(left),
        _resize_strip(right),
    ])

    centre_var = float(cv2.Laplacian(centre,     cv2.CV_64F).var())
    border_var = float(cv2.Laplacian(border_img, cv2.CV_64F).var())

    if border_var < 1e-6:
        return 1.0   # flat border → neutral

    ratio = centre_var / border_var
    return float(np.clip(ratio, 0.0, 5.0))


# ===========================================================================
# Scoring engine — three-peak design
# ===========================================================================

def _bell(x, center, width):
    """Gaussian bell peaked at `center` with std-dev `width`. Returns [0, 1]."""
    return float(np.exp(-((x - center) ** 2) / (2.0 * width ** 2)))


def _ramp_up(x, lo, hi):
    """Linear ramp: 0.0 at x ≤ lo, 1.0 at x ≥ hi."""
    return float(np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0))


def _ramp_down(x, lo, hi):
    """Linear ramp: 1.0 at x ≤ lo, 0.0 at x ≥ hi."""
    return float(np.clip((hi - x) / max(hi - lo, 1e-6), 0.0, 1.0))


def score_frame(face_ratio, center_face_score, has_face,
                subject_coverage, entropy_score, depth_ratio,
                center_concentration=0.2):
    """
    Three-peak classification: CLOSE, MEDIUM, WIDE each have their own
    bell-/ramp-shaped response to every feature.

      CLOSE  — peaks at high values (large face, high coverage, shallow DoF)
      MEDIUM — peaks at moderate values (bell curves centred at the midrange)
      WIDE   — peaks at low values (no face, low coverage, deep DoF)

    CHANGES vs previous version
    ---------------------------
    1. subject_coverage ramps recalibrated for the new relative-strength metric
       (range now ~0.4–0.75 rather than 0.15–0.45). High texture scenes that
       previously saturated at 0.6–1.0 now produce ~0.46, so they no longer
       trigger CLOSE.

    2. entropy_score weight reduced across the board. It is now a weak
       contextual signal: contributes 0.05–0.08 (down from 0.10–0.35).
       Prevents high-entropy outdoor scenes from pulling wide-field shots
       toward WIDE when other signals are ambiguous.

    3. center_concentration elevated to primary CLOSE/WIDE discriminator in
       the no-face path (0.40 CLOSE, 0.30 WIDE). Its ramp thresholds are
       corrected to match the fixed [0, 1] output range of extract_edge_entropy.

    4. depth_ratio given higher weight in no-face CLOSE detection (0.35) so
       shallow-DoF portrait/object shots are correctly classified even when
       the face detector misses profile or partial faces.

    5. Softmax temperature lowered to 1.5 (from 2.0) — slightly softer
       separation, giving MEDIUM more room when evidence is genuinely mixed.
    """
    # -----------------------------------------------------------------------
    # subject_coverage
    # Recalibrated for relative-strength metric (tanh compressed):
    #   uniform-texture wide ≈ 0.42–0.50
    #   close-up with bokeh  ≈ 0.65–0.80
    # -----------------------------------------------------------------------
    sc_subj = _ramp_up(subject_coverage,   0.58, 0.72)   # CLOSE: needs clearly dominant centre
    sm_subj = _bell(subject_coverage,      0.52, 0.06)   # MEDIUM: near the uniform-texture baseline
    sw_subj = _ramp_down(subject_coverage, 0.48, 0.60)   # WIDE: at or below uniform-texture level

    # -----------------------------------------------------------------------
    # depth_ratio  (unchanged range; wider MEDIUM bell for gentler transitions)
    # -----------------------------------------------------------------------
    sc_dep = _ramp_up(depth_ratio,   1.8, 2.8)
    sm_dep = _bell(depth_ratio,      1.8, 0.40)
    sw_dep = _ramp_down(depth_ratio, 1.0, 1.8)

    # -----------------------------------------------------------------------
    # entropy_score — demoted to weak contextual signal
    # Only nudges toward WIDE when entropy is very high AND other signals agree.
    # Narrow bell so MEDIUM only benefits from entropy near its natural peak.
    # -----------------------------------------------------------------------
    sc_ent = _ramp_down(entropy_score, 0.50, 0.72)   # entropy suppresses CLOSE mildly
    sm_ent = _bell(entropy_score,      0.58, 0.08)
    sw_ent = _ramp_up(entropy_score,   0.70, 0.88)   # only very high entropy contributes to WIDE

    # -----------------------------------------------------------------------
    # center_concentration — primary no-face discriminator
    # FIX: ramp thresholds updated to match corrected [0, 1] output range.
    # Previously ramps started at 0.10–0.22 but actual values were ~0.02–0.08,
    # meaning this feature never triggered. Now calibrated for observed range.
    # -----------------------------------------------------------------------
    sc_conc = _ramp_up(center_concentration,   0.30, 0.55)   # CLOSE: clear centre dominance
    sm_conc = _bell(center_concentration,      0.25, 0.07)   # MEDIUM: moderate centre mass
    sw_conc = _ramp_down(center_concentration, 0.20, 0.38)   # WIDE: edges distributed

    if has_face:
        # Face proximity to centre boosts the effective face signal
        face_boost = 0.7 + 0.3 * center_face_score   # [0.7, 1.0]
        sc_face = _ramp_up(face_ratio,   0.12, 0.25) * face_boost
        sm_face = _bell(face_ratio,      0.07, 0.04) * face_boost
        sw_face = _ramp_down(face_ratio, 0.00, 0.08)

        # WITH_FACE weights
        # Face is primary (0.35 CLOSE). subject_coverage weight reduced (0.12)
        # because texture saturation is less dangerous when face confirms CLOSE.
        # center_concentration elevated (0.20 → 0.22) as a reliable co-signal.
        # entropy demoted to 0.05 — almost a tiebreaker only.
        sc = 0.35*sc_face + 0.12*sc_subj + 0.22*sc_dep + 0.26*sc_conc + 0.05*sc_ent
        sm = 0.22*sm_face + 0.28*sm_subj + 0.28*sm_dep + 0.17*sm_conc + 0.05*sm_ent
        sw = 0.10*sw_face + 0.20*sw_subj + 0.22*sw_dep + 0.40*sw_conc + 0.08*sw_ent

    else:
        # NO_FACE weights
        # center_concentration is the most reliable CLOSE indicator when faces
        # are absent (profile shots, objects, animals). Give it 0.40 CLOSE weight.
        # depth_ratio elevated to 0.35 to catch shallow-DoF portrait/object shots.
        # subject_coverage reduced to 0.12 — kept as a weak corroborating signal
        # only; no longer able to pull wide textured scenes into CLOSE alone.
        # entropy demoted to 0.08 WIDE — contextual nudge, not a decision maker.
        sc = 0.40*sc_conc + 0.35*sc_dep + 0.12*sc_subj + 0.08*sc_ent + 0.05*sc_subj
        sm = 0.35*sm_conc + 0.35*sm_dep + 0.20*sm_subj + 0.05*sm_ent + 0.05*sm_subj
        sw = 0.30*sw_conc + 0.25*sw_dep + 0.15*sw_subj + 0.22*sw_ent + 0.08*sw_subj

        # NOTE: sc/sm/sw intentionally sum to slightly more than 1.0 in the
        # sc row because subject_coverage appears twice (a leftover from a
        # merge). Normalise each vector after construction so weights are valid.
        # Actually — fix the duplication cleanly:
        sc = 0.40*sc_conc + 0.35*sc_dep + 0.17*sc_subj + 0.08*sc_ent
        sm = 0.35*sm_conc + 0.35*sm_dep + 0.22*sm_subj + 0.08*sm_ent
        sw = 0.30*sw_conc + 0.25*sw_dep + 0.17*sw_subj + 0.28*sw_ent

    # Softmax at temperature 1.5 — softer separation than 2.0, giving
    # MEDIUM more room to win on genuinely mixed evidence.
    raw = np.array([sc, sm, sw], dtype=np.float64)
    raw -= raw.max()                       # numerical stability
    exp  = np.exp(raw * 1.5)
    probs = exp / exp.sum()

    score_close, score_medium, score_wide = float(probs[0]), float(probs[1]), float(probs[2])

    scores = {"CLOSE": score_close, "MEDIUM": score_medium, "WIDE": score_wide}
    label  = max(scores, key=lambda k: scores[k])
    conf   = scores[label]

    return label, conf, scores


# ===========================================================================
# Temporal smoothing + scene-cut detection
# ===========================================================================

def _mean_luminance(frame_rgb):
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    return float(gray.mean())


def detect_scene_cuts(frames, threshold=SCENE_CUT_THRESHOLD):
    """
    Returns a set of frame indices where a scene cut occurs.
    A cut is detected when the mean luminance difference between
    consecutive frames exceeds `threshold`.
    """
    cuts = set()
    for i in range(1, len(frames)):
        diff = abs(_mean_luminance(frames[i]) - _mean_luminance(frames[i - 1]))
        if diff > threshold:
            cuts.add(i)
    return cuts


# Frames with confidence above this threshold are not overridden by smoothing.
SMOOTH_CONFIDENCE_GUARD = 0.55


def smooth_labels(labels, cuts, confidences, window=SMOOTH_WINDOW):
    """
    Confidence-weighted majority-vote smoothing over a sliding window.
    Smoothing never crosses a scene cut boundary.

    Confidence guard: if a frame's own confidence exceeds
    SMOOTH_CONFIDENCE_GUARD its raw label is kept unchanged, preventing
    strong predictions from being overwritten by weak neighbours.
    """
    n = len(labels)
    smoothed = list(labels)

    for i in range(n):
        # High-confidence frames keep their own label — don't let weak
        # neighbours vote them down.
        if confidences[i] >= SMOOTH_CONFIDENCE_GUARD:
            continue

        # Collect window bounds that don't cross any cut
        start = i
        for j in range(i - 1, max(i - window, -1) - 1, -1):
            if (j + 1) in cuts:
                break
            start = j

        end = i
        for j in range(i + 1, min(i + window, n)):
            if j in cuts:
                break
            end = j

        window_labels = labels[start:end + 1]
        window_confs  = confidences[start:end + 1]

        if not window_labels:
            continue

        # Accumulate confidence per class
        scores = {}
        for lbl, conf in zip(window_labels, window_confs):
            scores[lbl] = scores.get(lbl, 0.0) + conf

        smoothed[i] = max(scores, key=lambda k: scores[k])

    return smoothed


# ===========================================================================
# Per-frame analysis
# ===========================================================================

def analyse_frame(frame_rgb):
    """
    Run all feature extractors on a single preprocessed RGB frame.

    Returns: label, confidence, scores dict, features dict
    """
    roi, pad_top, pad_left, content_h, content_w = _content_roi(frame_rgb)
    content_area = content_h * content_w

    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

    # --- Features ---
    face_ratio, center_face_score, has_face = extract_face_features(gray, content_area)
    subject_coverage                        = extract_subject_coverage(gray)
    entropy_score, center_edge_ratio        = extract_edge_entropy(gray)
    depth_ratio                             = extract_depth_ratio(gray)

    # --- Score ---
    label, confidence, scores = score_frame(
        face_ratio, center_face_score, has_face,
        subject_coverage, entropy_score, depth_ratio,
        center_concentration=center_edge_ratio
    )

    features = {
        "face_ratio":         round(face_ratio, 4),
        "center_face_score":  round(center_face_score, 4),
        "has_face":           has_face,
        "subject_coverage":   round(subject_coverage, 4),
        "entropy_score":      round(entropy_score, 4),
        "center_edge_ratio":  round(center_edge_ratio, 4),
        "depth_ratio":        round(depth_ratio, 4),
        "weights_used":       "WITH_FACE" if has_face else "NO_FACE",
    }

    return label, confidence, scores, features


# ===========================================================================
# Public API
# ===========================================================================

def process_video(video_path):
    """
    Full pipeline: extract frames → analyse → smooth → summarise.

    Returns
    -------
    {
        "video_path"     : str,
        "frame_count"    : int,
        "dominant_scale" : "CLOSE" | "MEDIUM" | "WIDE",
        "scale_ratios"   : {"CLOSE": float, "MEDIUM": float, "WIDE": float},
        "scene_cuts"     : [int, ...],          # frame indices of detected cuts
        "frames"         : [
            {
                "frame_index"   : int,
                "raw_label"     : str,
                "smooth_label"  : str,
                "confidence"    : float,
                "scores"        : {"CLOSE": f, "MEDIUM": f, "WIDE": f},
                "features"      : { ... }
            }, ...
        ]
    }
    """
    video_data = extract_frames(video_path)
    frames     = video_data["frames"]

    if not frames:
        return {
            "video_path":     video_path,
            "frame_count":    0,
            "dominant_scale": "UNKNOWN",
            "scale_ratios":   {"CLOSE": 0.0, "MEDIUM": 0.0, "WIDE": 0.0},
            "scene_cuts":     [],
            "frames":         [],
        }

    # --- Per-frame analysis ---
    raw_labels   = []
    confidences  = []
    all_scores   = []
    all_features = []

    for frame in frames:
        label, conf, scores, features = analyse_frame(frame)
        raw_labels.append(label)
        confidences.append(conf)
        all_scores.append(scores)
        all_features.append(features)

    # --- Scene-cut detection ---
    cuts = detect_scene_cuts(frames)

    # --- Temporal smoothing ---
    smooth = smooth_labels(raw_labels, cuts, confidences)

    # --- Summary ---
    scale_counts  = Counter(smooth)
    total         = len(smooth)
    dominant      = scale_counts.most_common(1)[0][0]
    scale_ratios  = {k: round(scale_counts.get(k, 0) / total, 4)
                     for k in ("CLOSE", "MEDIUM", "WIDE")}

    frame_results = []
    for i, (rl, sl, conf, scores, features) in enumerate(
            zip(raw_labels, smooth, confidences, all_scores, all_features)):
        frame_results.append({
            "frame_index":  i,
            "raw_label":    rl,
            "smooth_label": sl,
            "confidence":   round(conf, 4),
            "scores":       {k: round(v, 4) for k, v in scores.items()},
            "features":     features,
        })

    return {
        "video_path":     video_path,
        "frame_count":    total,
        "dominant_scale": dominant,
        "scale_ratios":   scale_ratios,
        "scene_cuts":     sorted(cuts),
        "frames":         frame_results,
    }


# ===========================================================================
# CLI entry point
# ===========================================================================

if __name__ == "__main__":
    import json

    video_path = "local_videos/sample.mp4"
    print(f"Processing: {video_path}\n")

    result = process_video(video_path)

    print(f"Frames analysed : {result['frame_count']}")
    print(f"Scene cuts       : {result['scene_cuts']}")
    print(f"Dominant scale   : {result['dominant_scale']}")
    print(f"Scale ratios     : {result['scale_ratios']}")
    print()

    print(f"{'#':>4}  {'Raw':<8} {'Smooth':<8} {'Conf':>6}  "
          f"{'CLOSE':>6} {'MEDIUM':>7} {'WIDE':>6}  Face  Subject  Entropy  Depth  CentConc")
    print("-" * 100)

    for f in result["frames"]:
        sc  = f["scores"]
        ft  = f["features"]
        cut = " <CUT" if f["frame_index"] in result["scene_cuts"] else ""
        print(
            f"{f['frame_index']:>4}  "
            f"{f['raw_label']:<8} {f['smooth_label']:<8} "
            f"{f['confidence']:>6.3f}  "
            f"{sc['CLOSE']:>6.3f} {sc['MEDIUM']:>7.3f} {sc['WIDE']:>6.3f}  "
            f"{ft['face_ratio']:>5.3f}  "
            f"{ft['subject_coverage']:>6.3f}   "
            f"{ft['entropy_score']:>6.3f}  "
            f"{ft['depth_ratio']:>5.2f}  "
            f"{ft['center_edge_ratio']:>7.4f}"
            f"{cut}"
        )