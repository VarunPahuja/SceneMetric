"""
label_video.py
Extracts frames from a video and sorts them into class folders
based on defined time segments.

Usage:
    python label_video.py
    
Output:
    datasets/video_labeled/CLOSE/
    datasets/video_labeled/MEDIUM/
    datasets/video_labeled/WIDE/
    (garbage segments are skipped)
"""

import cv2
import os

# ── CONFIG ──────────────────────────────────────────────────────────────────
VIDEO_PATH = "local_videos/sample.mp4"          # change to sample.mp4 if needed

OUTPUT_BASE = "datasets/video_labeled"

# Time segments: (start_seconds, end_seconds, class_label)
# "garbage" segments are skipped entirely
SEGMENTS = [
    (0,    9,    None),       # 0:00 - 0:09  garbage → skip
    (9,    18,   "WIDE"),     # 0:09 - 0:18  long
    (19,   25,   "MEDIUM"),   # 0:19 - 0:25  medium close up → MEDIUM
    (26,   33,   "WIDE"),     # 0:26 - 0:33  long
    (34,   41,   "CLOSE"),    # 0:34 - 0:41  close up
    (42,   47,   "MEDIUM"),   # 0:42 - 0:47  medium
    (48,   54,   "MEDIUM"),   # 0:48 - 0:54  medium long → MEDIUM
    (55,   63,   None),       # 0:55 - 1:03  garbage → skip
    (66,   None, "CLOSE"),    # 1:06 - end   close up
]
# ────────────────────────────────────────────────────────────────────────────


def seconds_to_frame(seconds, fps):
    return int(seconds * fps)


def extract_labeled_frames(video_path, segments, output_base):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"Video: {video_path}")
    print(f"FPS: {fps:.2f} | Total frames: {total_frames} | Duration: {duration:.1f}s\n")

    # Create output dirs
    for cls in ["CLOSE", "MEDIUM", "WIDE"]:
        os.makedirs(os.path.join(output_base, cls), exist_ok=True)

    counters = {"CLOSE": 0, "MEDIUM": 0, "WIDE": 0, "skipped": 0}

    for seg_idx, (start_s, end_s, label) in enumerate(segments):
        if label is None:
            print(f"Segment {seg_idx+1}: {start_s}s - {end_s or 'end'}s → GARBAGE (skipping)")
            counters["skipped"] += 1
            continue

        start_frame = seconds_to_frame(start_s, fps)
        end_frame = seconds_to_frame(end_s, fps) if end_s is not None else total_frames

        print(f"Segment {seg_idx+1}: {start_s}s - {end_s or 'end'}s → {label} "
              f"(frames {start_frame} to {end_frame})")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for frame_num in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break

            filename = f"vid_seg{seg_idx+1:02d}_f{frame_num:05d}.jpg"
            out_path = os.path.join(output_base, label, filename)
            cv2.imwrite(out_path, frame)
            counters[label] += 1

    cap.release()

    print("\n── DONE ──────────────────────────────────────")
    print(f"  CLOSE  : {counters['CLOSE']} frames")
    print(f"  MEDIUM : {counters['MEDIUM']} frames")
    print(f"  WIDE   : {counters['WIDE']} frames")
    print(f"  Skipped segments: {counters['skipped']}")
    print(f"  Total extracted: {sum(v for k,v in counters.items() if k != 'skipped')}")
    print(f"\nSaved to: {os.path.abspath(output_base)}")


if __name__ == "__main__":
    extract_labeled_frames(VIDEO_PATH, SEGMENTS, OUTPUT_BASE)