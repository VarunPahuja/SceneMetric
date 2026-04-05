# SceneMetric Current Implementation Context

This file summarizes what is implemented in the codebase as of now.

## 1) Project Goal (Current Direction)

- Estimate shot scale labels (`CLOSE`, `MEDIUM`, `WIDE`) from video using classical computer vision (no CNN/deep model).
- Emphasis is on interpretable feature engineering and transparent scoring.

Reference notes:
- `VARUN_NOTES.md`

## 2) Implemented Core Pipeline

### A. Video Preprocessing (`modules/preprocessing.py`)

Implemented:
- Video loading via OpenCV (`load_video`).
- Letterbox resize to fixed square canvas (`TARGET_LONG_SIDE = 720`) while preserving aspect ratio (`letterbox_resize`).
- Brightness normalization using CLAHE on LAB-L channel (`normalize_brightness`).
- Content-aware color stabilization toward `TARGET_LUMINANCE` with clamped scale (`stabilize_color`).
- Two-stage blur rejection:
  - Fast pre-check on half-resolution (`_quick_blur_check`).
  - Main blur check on processed frame (`is_blurry`).
- Full per-frame preprocessing pipeline (`preprocess_frame`) with early rejection of blurry frames.
- Timestamp-based frame extraction and sampling (`extract_frames`) with:
  - Target sampling rate (`SAMPLING_FPS = 10`)
  - Duration cap (`MAX_DURATION = 120s`)
  - Blur skip accounting (`skipped_blur`)

Current behavior/output:
- Returns processed RGB frames and metadata (`fps`, `duration`, `frame_count`, `skipped_blur`).

### B. Shot Scale Estimation (`modules/shot_scale.py`)

Implemented:
- End-to-end frame classification into `CLOSE`, `MEDIUM`, `WIDE` using handcrafted features.
- Content ROI extraction that removes letterbox bars (`_content_roi`).
- Feature extractors:
  - Face prominence and center proximity via Haar cascade (`extract_face_features`).
  - Center-weighted subject coverage from Sobel gradients (`extract_subject_coverage`).
  - 3x3 spatial edge entropy and center concentration from Canny edges (`extract_edge_entropy`).
  - Depth-of-field proxy using center-vs-border Laplacian variance ratio (`extract_depth_ratio`).
- Weighted three-class scoring engine (`score_frame`) with separate behavior for face-present vs no-face cases.
- Softmax-based class confidence.
- Scene-cut detection by luminance jump (`detect_scene_cuts`).
- Confidence-guarded temporal smoothing that avoids crossing scene cuts (`smooth_labels`).
- Public end-to-end API (`process_video`) returning:
  - dominant scale
  - per-class ratios
  - scene cuts
  - per-frame raw/smoothed labels, scores, confidence, and feature values

## 3) Verification and Testing Utilities

### A. Preprocessing Validation

- Automated sanity + behavior checks: `modules/test_preprocessing.py`
  - Validates I/O, transform invariants, blur behavior, extraction counts, and sampling tolerance.
- Interactive visual debugger for preprocessing: `verify_preprocessing.py`
  - Step-by-step mode and before/after comparison mode.
  - Displays acceptance/rejection statistics for blur filtering.

### B. Shot Scale Visual Verification

- Interactive frame-level classifier viewer: `verify_shot_scale.py`
  - Overlays label, confidence, class scores, and feature values on sampled frames.
  - Prints end summary with scale distribution.

## 4) Data and Assets Present

- Dataset metadata JSON files under `datasets/ECCV20Shot/`:
  - `v1_full_trailer.json`
  - `v1_split_trailer.json`
  - `v2_full_trailer.json`
  - `v3_full.json`
  - `v3_split.json`
- Local video folder exists: `local_videos/`
- `assets/` and `models/` directories exist.

## 5) Not Yet Implemented / Placeholder Areas

- `modules/motion_analysis.py` is currently empty.
- `modules/fusion.py` is currently empty.
- `modules/__init__.py` is currently empty.
- No training pipeline or learned model artifacts are implemented in active code paths.

## 6) Legacy / Scratch File Signals

- `tempCodeRunnerFile.py` at project root contains an older heuristic analyzer snippet.
- `modules/tempCodeRunnerFile.py` appears to be a temporary/partial duplicate of verification code.

## 7) Practical Status Summary

What is working now:
- A full classical CV preprocessing + shot-scale inference pipeline with interpretable features.
- Frame-level and video-level outputs with confidence and temporal smoothing.
- Manual visual verification tools and preprocessing tests.

What remains for expansion:
- Implement motion branch (`motion_analysis.py`) and multi-signal fusion (`fusion.py`) if a multi-modal design is intended.
- Add integration/regression tests for `process_video` outputs and expected distributions on benchmark clips.
- Clean up temporary runner files to reduce confusion.
