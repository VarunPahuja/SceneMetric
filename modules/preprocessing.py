import cv2
import numpy as np

TARGET_LONG_SIDE  = 720
SAMPLING_FPS      = 10
MAX_DURATION      = 120      # seconds
BLUR_THRESHOLD    = 70.0    # variance of Laplacian below this → skip frame
TARGET_LUMINANCE  = 128.0   # target mean brightness for color stabilization
SCALE_CLAMP       = (0.6, 1.8)  # max darkening / brightening stabilize_color will apply


def load_video(video_path):
    return cv2.VideoCapture(video_path)


def letterbox_resize(frame):
    """Resize frame to fit within TARGET_LONG_SIDE × TARGET_LONG_SIDE,
    preserving aspect ratio and padding with black bars."""
    h, w = frame.shape[:2]
    scale = TARGET_LONG_SIDE / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((TARGET_LONG_SIDE, TARGET_LONG_SIDE, 3), dtype=np.uint8)
    pad_top  = (TARGET_LONG_SIDE - new_h) // 2
    pad_left = (TARGET_LONG_SIDE - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return canvas, pad_top, pad_left, new_h, new_w


def normalize_brightness(frame_bgr):
    """Apply CLAHE on the L channel (LAB space) to normalize brightness.
    clipLimit=3.0 preserves contrast in large dark cinema areas."""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_normalized = clahe.apply(l)
    lab_normalized = cv2.merge([l_normalized, a, b])
    return cv2.cvtColor(lab_normalized, cv2.COLOR_LAB2BGR)


def stabilize_color(frame_bgr, pad_top, pad_left, content_h, content_w):
    """Scale frame brightness toward TARGET_LUMINANCE using only the
    content region (not the black letterbox bars), then clamp the scale
    factor to SCALE_CLAMP to prevent highlight blow-out on dark frames
    with a bright spot.

    Fixes vs. previous version:
      - mean computed on content pixels only  → black bars no longer drag mean down
      - scale clamped to [0.5, 2.0]           → no blow-out from extreme scaling
      - threshold raised to 5.0               → robustly guards near-black frames
    """
    content = frame_bgr[pad_top:pad_top + content_h, pad_left:pad_left + content_w]
    mean_lum = content.mean()

    if mean_lum < 10.0:          # near-black frame — nothing useful to scale
        return frame_bgr

    scale = TARGET_LUMINANCE / mean_lum
    scale = float(np.clip(scale, SCALE_CLAMP[0], SCALE_CLAMP[1]))

    stabilized = np.clip(frame_bgr.astype(np.float32) * scale, 0, 255)
    return stabilized.astype(np.uint8)


def is_blurry(frame_bgr, threshold=BLUR_THRESHOLD):
    """Return True if the frame is too blurry to be useful.
    Called after resize so Laplacian variance is resolution-consistent."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold


def _quick_blur_check(frame_bgr, threshold=BLUR_THRESHOLD):
    """Fast pre-check on a half-resolution version of the raw frame.
    Rejects obviously blurry frames before the expensive CLAHE pipeline.
    Less aggressive than final blur check to avoid false rejections."""
    small = cv2.resize(frame_bgr, (0, 0), fx=0.5, fy=0.5,
                       interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    # At 0.5x scale, Laplacian variance drops roughly 4×.
    # threshold / 3 ≈ 23.3 is stricter than the old threshold / 5 = 14,
    # so obviously blurry frames are caught before the expensive CLAHE pipeline.
    return cv2.Laplacian(gray, cv2.CV_64F).var() < (threshold / 3)


def preprocess_frame(frame_bgr):
    """
    Per-frame preprocessing pipeline (BGR input -> RGB output):

      1. Quick blur pre-check      — rejects obviously blurry frames cheaply
      2. Letterbox resize          — resolution-consistent, aspect-safe;
                                     also returns padding coords for step 5
      3. CLAHE brightness norm     — handles cinema lighting
      4. Accurate blur check       — after CLAHE for fair baseline, before
                                     stabilize_color to avoid contrast distortion
      5. Color stabilization       — content-region mean, clamped scale
      6. BGR -> RGB conversion

    Returns None if the frame is too blurry.
    """
    # 1. Cheap early-exit on obviously blurry raw frames
    if _quick_blur_check(frame_bgr):
        return None

    # 2. Letterbox resize — unpack padding coords for stabilize_color
    canvas, pad_top, pad_left, content_h, content_w = letterbox_resize(frame_bgr)

    # 3. CLAHE
    canvas = normalize_brightness(canvas)

    # 4. Accurate blur check — after CLAHE (fair brightness baseline),
    #    before stabilize_color (which scales contrast and would corrupt the measurement)
    if is_blurry(canvas):
        return None

    # 5. Color stabilization using content region only
    canvas = stabilize_color(canvas, pad_top, pad_left, content_h, content_w)

    # 6. BGR -> RGB
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def extract_frames(video_path):
    """
    Extract frames from a video file.

    Pipeline:
      video -> clip to MAX_DURATION -> timestamp-based sampling (10 FPS)
            -> quick blur pre-check -> letterbox resize
            -> CLAHE normalization -> color stabilization (content region)
            -> final blur check -> RGB frames

    Returns:
        dict with keys: frames, fps, duration, frame_count, skipped_blur
    """
    cap = load_video(video_path)

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if original_fps <= 0:
        original_fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 0
            while cap.read()[0]:
                total_frames += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    duration        = min(total_frames / original_fps, MAX_DURATION)
    sample_interval = 1.0 / SAMPLING_FPS

    frames       = []
    skipped_blur = 0
    frame_index  = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_index / original_fps
        if timestamp > duration:
            break

        # Robust next_sample_t: recalculate from total sampled so far.
        # Prevents burst-sampling when timestamp jumps past multiple intervals
        # (e.g. high-FPS source or system stutter).
        total_sampled = len(frames) + skipped_blur
        next_sample_t = total_sampled * sample_interval

        if timestamp >= next_sample_t:
            processed = preprocess_frame(frame)
            if processed is None:
                skipped_blur += 1
            else:
                frames.append(processed)

        frame_index += 1

    cap.release()

    return {
        "frames":       frames,
        "fps":          SAMPLING_FPS,
        "duration":     duration,
        "frame_count":  len(frames),
        "skipped_blur": skipped_blur,
    }


def to_grayscale(frame_rgb):
    """Convert an RGB frame to grayscale."""
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)