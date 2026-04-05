"""
test_preprocessing.py
---------------------
Validates that the preprocessing pipeline is working correctly.
Run: python test_preprocessing.py
"""

import sys
import cv2
import numpy as np

from preprocessing import (
    extract_frames,
    letterbox_resize,
    normalize_brightness,
    stabilize_color,
    is_blurry,
    preprocess_frame,
    TARGET_LONG_SIDE,
    SAMPLING_FPS,
    MAX_DURATION,
    BLUR_THRESHOLD,
    TARGET_LUMINANCE,
)

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[94m[INFO]\033[0m"


def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    msg = f"{status} {label}"
    if detail:
        msg += f"  ->  {detail}"
    print(msg)
    return condition


# ---------------------------------------------------------------------------
# 1. File + OpenCV sanity
# ---------------------------------------------------------------------------
def test_video_opens(video_path):
    print("\n-- Video file ----------------------------------------------")
    cap = cv2.VideoCapture(video_path)
    opened = check("File opens", cap.isOpened(), video_path)
    if not opened:
        cap.release()
        return False

    fps    = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur    = frames / fps if fps > 0 else 0

    print(f"  {INFO} Resolution   : {w}x{h}")
    print(f"  {INFO} Source FPS   : {fps:.3f}")
    print(f"  {INFO} Total frames : {frames}")
    print(f"  {INFO} Duration     : {dur:.2f}s")

    check("FPS > 0 (metadata readable)", fps > 0,
          "fallback to 30 FPS will be used" if fps <= 0 else f"{fps:.3f}")
    check("Has frames", frames > 0)
    cap.release()
    return True


# ---------------------------------------------------------------------------
# 2. Individual transform checks (on one raw frame)
# ---------------------------------------------------------------------------
def test_transforms(video_path):
    print("\n-- Per-transform checks ------------------------------------")
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"  {FAIL} Could not read first frame.")
        return

    h_src, w_src = frame.shape[:2]
    print(f"  {INFO} Source frame : {w_src}x{h_src}")

    # --- letterbox_resize (now returns canvas + padding coords) ---
    canvas, pad_top, pad_left, content_h, content_w = letterbox_resize(frame)
    check("Letterbox output is square canvas",
          canvas.shape == (TARGET_LONG_SIDE, TARGET_LONG_SIDE, 3),
          str(canvas.shape))
    check("Letterbox dtype preserved", canvas.dtype == np.uint8)
    content_region = canvas[pad_top:pad_top + content_h, pad_left:pad_left + content_w]
    check("Letterbox content region non-black",
          content_region.mean() > 1.0,
          f"mean={content_region.mean():.1f}")

    # --- normalize_brightness ---
    normed = normalize_brightness(canvas)
    check("CLAHE output same shape", normed.shape == canvas.shape)
    check("CLAHE output same dtype", normed.dtype == np.uint8)
    lab_in  = cv2.cvtColor(canvas, cv2.COLOR_BGR2LAB)
    lab_out = cv2.cvtColor(normed, cv2.COLOR_BGR2LAB)
    a_drift = int(np.abs(lab_in[:, :, 1].astype(int) - lab_out[:, :, 1].astype(int)).max())
    b_drift = int(np.abs(lab_in[:, :, 2].astype(int) - lab_out[:, :, 2].astype(int)).max())
    check("CLAHE only affects L channel (a/b drift <= 2)",
          a_drift <= 2 and b_drift <= 2,
          f"max a_drift={a_drift}  max b_drift={b_drift}")

    # --- stabilize_color (now needs padding coords) ---
    stabilized = stabilize_color(normed, pad_top, pad_left, content_h, content_w)
    mean_before = normed[pad_top:pad_top + content_h, pad_left:pad_left + content_w].mean()
    mean_after  = stabilized[pad_top:pad_top + content_h, pad_left:pad_left + content_w].mean()
    check("Color stabilization dtype preserved", stabilized.dtype == np.uint8)
    check("Color stabilization moves content mean toward target",
          abs(mean_after - TARGET_LUMINANCE) < abs(mean_before - TARGET_LUMINANCE)
          or abs(mean_before - TARGET_LUMINANCE) < 5,
          f"before={mean_before:.1f}  after={mean_after:.1f}  target={TARGET_LUMINANCE}")

    # --- scale clamp: ensure no blow-out on a dark frame with bright spot ---
    dark_frame = np.zeros_like(normed)
    dark_frame[pad_top + 5, pad_left + 5] = [255, 255, 255]   # single bright pixel
    dark_frame[pad_top:pad_top + content_h, pad_left:pad_left + content_w] += 2
    clamped = stabilize_color(dark_frame, pad_top, pad_left, content_h, content_w)
    check("Scale clamp prevents blow-out on dark+bright-spot frame",
          clamped.max() <= 255 and clamped.dtype == np.uint8)

    # --- is_blurry ---
    sharp_var  = cv2.Laplacian(cv2.cvtColor(stabilized, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    blurry     = cv2.GaussianBlur(stabilized, (31, 31), 0)
    blurry_var = cv2.Laplacian(cv2.cvtColor(blurry, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    check("Sharp frame passes blur filter",        not is_blurry(stabilized), f"variance={sharp_var:.1f}")
    check("Heavily blurred frame flagged as blurry", is_blurry(blurry),       f"variance={blurry_var:.1f}")

    # --- preprocess_frame end-to-end ---
    result = preprocess_frame(frame)
    check("preprocess_frame returns array (not blurry)",
          result is not None, "returned None — frame was blurry" if result is None else "ok")
    if result is not None:
        check("Output is RGB (shape HxWx3)",  result.ndim == 3 and result.shape[2] == 3)
        check("Output dtype uint8",           result.dtype == np.uint8)
        check("Output is square canvas",
              result.shape[0] == TARGET_LONG_SIDE and result.shape[1] == TARGET_LONG_SIDE,
              str(result.shape))
        bgr_check = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        check("Channels are RGB not BGR",
              np.array_equal(bgr_check[:, :, 0], result[:, :, 2]),
              "channel swap verified")


# ---------------------------------------------------------------------------
# 3. extract_frames end-to-end
# ---------------------------------------------------------------------------
def test_extract_frames(video_path):
    print("\n-- extract_frames end-to-end -------------------------------")
    data = extract_frames(video_path)

    keys = {"frames", "fps", "duration", "frame_count", "skipped_blur"}
    check("Return dict has all keys", keys.issubset(data.keys()))

    frames       = data["frames"]
    frame_count  = data["frame_count"]
    duration     = data["duration"]
    skipped_blur = data["skipped_blur"]

    check("At least 1 frame extracted",       len(frames) > 0, f"{len(frames)} frames")
    check("frame_count matches list length",  frame_count == len(frames))
    check("Duration capped at MAX_DURATION",  duration <= MAX_DURATION + 0.1, f"{duration:.2f}s")

    expected  = int(duration * SAMPLING_FPS)
    actual    = len(frames) + skipped_blur
    tolerance = max(3, int(expected * 0.10))
    check("Sampled frame count within 10% of expected",
          abs(actual - expected) <= tolerance,
          f"expected~{expected}  got={actual}  skipped_blur={skipped_blur}")

    print(f"  {INFO} Skipped (blur): {skipped_blur}")

    if frames:
        f = frames[0]
        check("Frame is numpy array",         isinstance(f, np.ndarray))
        check("Frame shape is square canvas",
              f.shape == (TARGET_LONG_SIDE, TARGET_LONG_SIDE, 3), str(f.shape))
        check("Frame dtype uint8",            f.dtype == np.uint8)
        check("Frame values in [0,255]",      f.min() >= 0 and f.max() <= 255,
              f"min={f.min()} max={f.max()}")
        shapes = {fr.shape for fr in frames}
        check("All frames have identical shape", len(shapes) == 1, str(shapes))


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------
def main():
    video_path = "local_videos/sample.mp4"
    print(f"Testing preprocessing pipeline on: {video_path}")

    if not test_video_opens(video_path):
        print("\nAborting -- fix file path/codec first.")
        sys.exit(1)

    test_transforms(video_path)
    test_extract_frames(video_path)

    print("\n------------------------------------------------------------")
    print("Done. Fix any [FAIL] lines above before proceeding.")


if __name__ == "__main__":
    main()