import cv2
import numpy as np
from preprocessing import extract_frames


# Load Haar cascade once (global)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
if face_cascade.empty():
    raise RuntimeError("Failed to load Haar cascade.")

def analyze_shot_scale(frames):
    """
    Shot scale estimation using:
    1) Face area ratio (primary signal)
    2) Edge + spatial fallback (secondary signal)

    Args:
        frames: List of RGB frames (360x640x3, uint8)

    Returns:
        dict with close_ratio, medium_ratio, wide_ratio
    """

    if not frames:
        return {
            "close_ratio": 0.0,
            "medium_ratio": 0.0,
            "wide_ratio": 0.0
        }

    close_count = 0
    medium_count = 0
    wide_count = 0
    total_frames = len(frames)

    # Thresholds for face area ratio
    CLOSE_FACE_THRESHOLD = 0.20
    MEDIUM_FACE_THRESHOLD = 0.08

    # Fallback thresholds (edge-based)
    GLOBAL_CLOSE_THRESHOLD = 0.02
    GLOBAL_WIDE_THRESHOLD = 0.06

    for frame in frames:

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        frame_area = h * w

        # ------------------------------
        # 1️⃣ Face Detection
        # ------------------------------
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        largest_face_area = 0
        for (x, y, fw, fh) in faces:
            area = fw * fh
            if area > largest_face_area:
                largest_face_area = area

        face_area_ratio = largest_face_area / frame_area


        # ------------------------------
        # Face-based classification
        # ------------------------------
        if face_area_ratio > CLOSE_FACE_THRESHOLD:
            close_count += 1
            continue

        elif face_area_ratio > MEDIUM_FACE_THRESHOLD:
            medium_count += 1
            continue

        # ------------------------------
        # 2️⃣ Fallback: Edge-based logic
        # ------------------------------
        edges = cv2.Canny(gray, 100, 200)

        total_edges = np.count_nonzero(edges)
        edge_density = total_edges / frame_area

        third = w // 3
        left = edges[:, :third]
        center = edges[:, third:2 * third]
        right = edges[:, 2 * third:]

        left_edges = np.count_nonzero(left)
        center_edges = np.count_nonzero(center)
        right_edges = np.count_nonzero(right)

        center_ratio = center_edges / total_edges if total_edges > 0 else 0
        spread_ratio = (left_edges + right_edges) / total_edges if total_edges > 0 else 0

        # Wide condition
        if edge_density > GLOBAL_WIDE_THRESHOLD and spread_ratio > 0.65:
            wide_count += 1

        # Close fallback
        elif edge_density < GLOBAL_CLOSE_THRESHOLD and center_ratio > 0.5:
            close_count += 1

        else:
            medium_count += 1

    return {
        "close_ratio": close_count / total_frames,
        "medium_ratio": medium_count / total_frames,
        "wide_ratio": wide_count / total_frames
    }


# -----------------------------------
# Direct Test
# -----------------------------------
if __name__ == "__main__":

    print("Loading sample.mp4...")

    video_path = "local_videos/sample.mp4"

    video_data = extract_frames(video_path)
    frames = video_data["frames"]

    print(f"Extracted {len(frames)} frames")

    result = analyze_shot_scale(frames)

    print("Shot Scale Result:")
    print(result)
