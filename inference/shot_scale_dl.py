"""
shot_scale_dl.py
----------------
Shot-aware deep learning shot-scale estimator using a trained ResNet-18 model.

Pipeline highlights:
1) Frame-difference filtering to skip near-duplicates
2) Hard-cut shot boundary detection
3) Shot-level batched inference on representative frames
4) Temporal smoothing across shots
5) Shot-level output (not per-frame output)
"""

from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# Support both direct execution and module import
try:
    from .preprocessing import extract_frames
except ImportError:
    from modules.preprocessing import extract_frames


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASS_NAMES = ["CLOSE", "MEDIUM", "WIDE"]
MODEL_PATH = Path("models/resnet18_mixup_f108089.pt")
DEVICE = torch.device("cpu")

DEFAULT_TEMPERATURE = 0.7
UNCERTAIN_MARGIN_THRESHOLD = 0.1

# Shot-aware thresholds requested by user
FRAME_CHANGE_THRESHOLD = 0.25
SHOT_CUT_THRESHOLD = 1.0

# For shot-level representative sampling
MAX_REP_FRAMES_PER_SHOT = 3


# ImageNet normalization used by ResNet pretraining
_PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Loaded once globally
_MODEL = None


def _build_model():
    """Create ResNet-18 with a 3-class classification head."""
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, 3),
    )
    return model


def _load_model_once():
    """Load model weights once and keep model cached globally."""
    global _MODEL

    if _MODEL is not None:
        return _MODEL

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

    model = _build_model().to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    _MODEL = model
    return _MODEL


def _frame_diff_score(frame_a, frame_b):
    """
    Return frame-difference score using Chi-Square distance on normalized
    grayscale histograms. Score range is unbounded, making both 0.25 and 1.0
    thresholds practical.
    """
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_RGB2GRAY)

    hist_a = cv2.calcHist([gray_a], [0], None, [64], [0, 256]).astype(np.float32)
    hist_b = cv2.calcHist([gray_b], [0], None, [64], [0, 256]).astype(np.float32)

    cv2.normalize(hist_a, hist_a, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
    cv2.normalize(hist_b, hist_b, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)

    return float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CHISQR))


def _compute_frame_differences(frames):
    """Compute diff score for consecutive frame pairs."""
    if len(frames) < 2:
        return []
    return [_frame_diff_score(frames[i], frames[i + 1]) for i in range(len(frames) - 1)]


def filter_frames(frames, frame_diffs, change_threshold=FRAME_CHANGE_THRESHOLD):
    """
    Keep only meaningful frames where diff from previous frame exceeds threshold.
    Returns a boolean mask aligned with frame indices.
    """
    n = len(frames)
    if n == 0:
        return []

    keep = [False] * n
    keep[0] = True

    for i, diff in enumerate(frame_diffs, start=1):
        if diff > change_threshold:
            keep[i] = True

    return keep


def detect_shots(frame_diffs, cut_threshold=SHOT_CUT_THRESHOLD):
    """
    Detect hard cuts based on consecutive-frame difference.
    Returns list of (start_idx, end_idx) shot intervals.
    """
    if not frame_diffs:
        return [(0, 0)]

    boundaries = [0]
    for i, diff in enumerate(frame_diffs, start=1):
        if diff > cut_threshold:
            boundaries.append(i)

    boundaries.append(len(frame_diffs) + 1)

    shots = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1] - 1
        if end >= start:
            shots.append((start, end))
    return shots


def _choose_representative_indices(start_idx, end_idx, keep_mask,
                                   max_reps=MAX_REP_FRAMES_PER_SHOT):
    """Pick up to max_reps representative frame indices from one shot."""
    candidates = [i for i in range(start_idx, end_idx + 1) if keep_mask[i]]

    if not candidates:
        mid = (start_idx + end_idx) // 2
        candidates = [start_idx, mid, end_idx]

    ordered = sorted(set(candidates))
    if len(ordered) <= max_reps:
        return ordered

    positions = np.linspace(0, len(ordered) - 1, num=max_reps, dtype=int)
    return [ordered[p] for p in positions]


def _infer_batch_probs(frames_rgb, temperature=DEFAULT_TEMPERATURE):
    """Run one batched model forward pass and return probabilities [N, 3]."""
    if not frames_rgb:
        return np.empty((0, len(CLASS_NAMES)), dtype=np.float32)
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    model = _load_model_once()

    batch_tensors = []
    for frame_rgb in frames_rgb:
        pil_img = Image.fromarray(frame_rgb.astype(np.uint8), mode="RGB")
        batch_tensors.append(_PREPROCESS(pil_img))

    batch = torch.stack(batch_tensors, dim=0).to(DEVICE)

    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits / temperature, dim=1).cpu().numpy()

    return probs


def analyse_frame_dl(frame_rgb, temperature=DEFAULT_TEMPERATURE,
                     uncertain_margin=UNCERTAIN_MARGIN_THRESHOLD):
    """Single-frame inference helper."""
    probs = _infer_batch_probs([frame_rgb], temperature=temperature)[0]

    top2_idx = np.argsort(probs)[-2:][::-1]
    top1_idx = int(top2_idx[0])
    top2_idx_val = int(top2_idx[1])

    top1_prob = float(probs[top1_idx])
    top2_prob = float(probs[top2_idx_val])
    margin = top1_prob - top2_prob

    label = CLASS_NAMES[top1_idx]
    if margin < uncertain_margin:
        label = "UNCERTAIN"

    scores = {name: float(probs[i]) for i, name in enumerate(CLASS_NAMES)}
    return label, top1_prob, float(margin), scores


def classify_shot(frames, start_idx, end_idx, keep_mask,
                  temperature=DEFAULT_TEMPERATURE,
                  uncertain_margin=UNCERTAIN_MARGIN_THRESHOLD,
                  max_rep_frames=MAX_REP_FRAMES_PER_SHOT,
                  fps=10.0):
    """
    Classify one shot by aggregating probabilities over representative frames.
    """
    rep_indices = _choose_representative_indices(
        start_idx, end_idx, keep_mask, max_reps=max_rep_frames
    )
    rep_frames = [frames[i] for i in rep_indices]

    probs = _infer_batch_probs(rep_frames, temperature=temperature)
    mean_probs = probs.mean(axis=0)

    top2_idx = np.argsort(mean_probs)[-2:][::-1]
    top1_idx = int(top2_idx[0])
    top2_idx_val = int(top2_idx[1])

    top1_prob = float(mean_probs[top1_idx])
    top2_prob = float(mean_probs[top2_idx_val])
    margin = top1_prob - top2_prob

    label = CLASS_NAMES[top1_idx]
    if margin < uncertain_margin:
        label = "UNCERTAIN"

    duration_sec = float((end_idx - start_idx + 1) / max(fps, 1e-6))

    return {
        "shot_index": -1,
        "start_frame": int(start_idx),
        "end_frame": int(end_idx),
        "duration": round(duration_sec, 3),
        "raw_label": label,
        "smooth_label": label,
        "confidence": top1_prob,
        "margin": float(margin),
        "top1_class": CLASS_NAMES[top1_idx],
        "top2_class": CLASS_NAMES[top2_idx_val],
        "scores": {k: float(mean_probs[i]) for i, k in enumerate(CLASS_NAMES)},
        "representative_frames": rep_indices,
    }


def smooth_predictions(shot_results, low_confidence=0.6,
                       uncertain_margin=UNCERTAIN_MARGIN_THRESHOLD):
    """
    Smooth shot labels using local temporal consistency.

    Rule:
    - If current shot disagrees with both neighbors and neighbors agree,
      replace current label when it is low-confidence or uncertain.
    """
    if not shot_results:
        return shot_results

    smoothed = [dict(s) for s in shot_results]

    for i in range(1, len(smoothed) - 1):
        prev_label = smoothed[i - 1]["smooth_label"]
        curr_label = smoothed[i]["smooth_label"]
        next_label = smoothed[i + 1]["smooth_label"]

        weak = (
            smoothed[i]["confidence"] < low_confidence
            or smoothed[i]["margin"] < uncertain_margin
            or curr_label == "UNCERTAIN"
        )

        if weak and prev_label == next_label and curr_label != prev_label:
            smoothed[i]["smooth_label"] = prev_label

    return smoothed


def _print_debug_logs(frame_diffs, keep_mask, shots, shot_results):
    """Verbose debug logs for diffs, cuts, filtering, and shot predictions."""
    print("\nFRAME DIFFERENCES")
    print("-" * 80)
    for i, diff in enumerate(frame_diffs):
        keep_flag = "keep" if keep_mask[i + 1] else "skip"
        cut_flag = " | CUT" if diff > SHOT_CUT_THRESHOLD else ""
        print(f"Frame {i} -> {i+1}: diff={diff:.3f} | {keep_flag}{cut_flag}")

    print("\nSHOT BOUNDARIES")
    print("-" * 80)
    for idx, (start, end) in enumerate(shots, start=1):
        print(f"Shot {idx}: frames {start}-{end}")

    print("\nSHOT-LEVEL PREDICTIONS")
    print("-" * 80)
    for shot in shot_results:
        print(
            f"Shot {shot['shot_index']}: "
            f"raw={shot['raw_label']:<9} smooth={shot['smooth_label']:<9} "
            f"conf={shot['confidence']:.3f} margin={shot['margin']:.3f} "
            f"reps={shot['representative_frames']}"
        )


def process_video_dl(video_path, temperature=DEFAULT_TEMPERATURE,
                     uncertain_margin=UNCERTAIN_MARGIN_THRESHOLD,
                     frame_change_threshold=FRAME_CHANGE_THRESHOLD,
                     shot_cut_threshold=SHOT_CUT_THRESHOLD,
                     max_rep_frames=MAX_REP_FRAMES_PER_SHOT,
                     debug=False):
    """
    Shot-aware deep-learning video analysis pipeline.

    Returns shot-level output only (no per-frame predictions).
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    video_data = extract_frames(video_path)
    frames = video_data["frames"]
    fps = float(video_data.get("fps", 10.0))

    if not frames:
        return {
            "video_path": video_path,
            "total_shots": 0,
            "dominant_scale": "UNKNOWN",
            "scale_ratios": {"CLOSE": 0.0, "MEDIUM": 0.0, "WIDE": 0.0},
            "shots": [],
        }

    frame_diffs = _compute_frame_differences(frames)
    keep_mask = filter_frames(frames, frame_diffs, change_threshold=frame_change_threshold)
    shots = detect_shots(frame_diffs, cut_threshold=shot_cut_threshold)

    shot_results = []
    for shot_idx, (start_idx, end_idx) in enumerate(shots, start=1):
        shot = classify_shot(
            frames=frames,
            start_idx=start_idx,
            end_idx=end_idx,
            keep_mask=keep_mask,
            temperature=temperature,
            uncertain_margin=uncertain_margin,
            max_rep_frames=max_rep_frames,
            fps=fps,
        )
        shot["shot_index"] = shot_idx
        shot_results.append(shot)

    shot_results = smooth_predictions(
        shot_results,
        low_confidence=0.6,
        uncertain_margin=uncertain_margin,
    )

    # Distribution uses smoothed labels and excludes UNCERTAIN from ratios.
    smoothed_labels = [s["smooth_label"] for s in shot_results]
    valid_labels = [lbl for lbl in smoothed_labels if lbl in CLASS_NAMES]
    label_counts = Counter(valid_labels)

    valid_total = len(valid_labels)
    if valid_total == 0:
        scale_ratios = {k: 0.0 for k in CLASS_NAMES}
        dominant_scale = "UNKNOWN"
    else:
        scale_ratios = {
            k: round(label_counts.get(k, 0) / valid_total, 4)
            for k in CLASS_NAMES
        }
        dominant_scale = max(CLASS_NAMES, key=lambda k: label_counts.get(k, 0))

    if debug:
        _print_debug_logs(frame_diffs, keep_mask, shots, shot_results)

    # Human-readable shot summary
    print("\nSHOT SUMMARY")
    print("-" * 80)
    for shot in shot_results:
        print(
            f"Shot {shot['shot_index']}: {shot['smooth_label']:<7} "
            f"({shot['confidence']:.2f} confidence, duration: {shot['duration']:.1f}s)"
        )

    return {
        "video_path": video_path,
        "total_shots": len(shot_results),
        "dominant_scale": dominant_scale,
        "scale_ratios": scale_ratios,
        "shots": [
            {
                "shot_index": s["shot_index"],
                "start_frame": s["start_frame"],
                "end_frame": s["end_frame"],
                "duration": round(float(s["duration"]), 3),
                "label": s["smooth_label"],
                "raw_label": s["raw_label"],
                "confidence": round(float(s["confidence"]), 4),
                "margin": round(float(s["margin"]), 4),
                "scores": {k: round(v, 4) for k, v in s["scores"].items()},
                "representative_frames": s["representative_frames"],
            }
            for s in shot_results
        ],
    }


if __name__ == "__main__":
    sample_video = "local_videos/sample.mp4"

    print(f"Processing (DL, shot-aware): {sample_video}")
    result = process_video_dl(sample_video, debug=True)

    print(f"\nTotal shots    : {result['total_shots']}")
    print(f"Dominant scale : {result['dominant_scale']}")
    print(f"Scale ratios   : {result['scale_ratios']}")
