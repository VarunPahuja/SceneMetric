import cv2
import numpy as np

TARGET_WIDTH = 640
TARGET_HEIGHT = 360
SAMPLING_FPS = 2
MAX_DURATION = 15  # seconds


def load_video(video_path):
    return cv2.VideoCapture(video_path)


def extract_frames(video_path):
    cap = load_video(video_path)

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / original_fps

    duration = min(duration, MAX_DURATION)

    frame_interval = int(original_fps / SAMPLING_FPS)

    frames = []
    frame_index = 0
    extracted = 0
    max_frames = int(duration * SAMPLING_FPS)

    while cap.isOpened() and extracted < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_interval == 0:
            frame = preprocess_frame(frame)
            frames.append(frame)
            extracted += 1

        frame_index += 1

    cap.release()

    return {
        "frames": frames,
        "fps": SAMPLING_FPS,
        "duration": duration,
        "frame_count": len(frames)
    }


def preprocess_frame(frame):
    # BGR → RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize
    frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

    return frame


def to_grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
