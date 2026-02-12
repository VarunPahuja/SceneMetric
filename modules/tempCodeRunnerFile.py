import cv2
import numpy as np

# Import preprocessing
from preprocessing import extract_frames


def analyze_shot_scale(frames):
    """
    Estimates shot scale distribution (Close, Medium, Wide)
    using edge density + spatial distribution heuristics.
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

    # Thresholds tuned for 360p resolution
    GLOBAL_CLOSE_THRESHOLD = 0.02
    GLOBAL_WIDE_THRESHOLD = 0.06
    CENTER_DOMINANCE_THRESHOLD = 0.45

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)

        total_pixels = edges.size
        total_edges = np.count_nonzero(edges)
        edge_density = total_edges / total_pixels

        h, w = edges.shape
        third = w // 3

        left_region = edges[:, :third]
        center_region = edges[:, third:2 * third]
        right_region = edges[:, 2 * third:]

        left_edges = np.count_nonzero(left_region)
        center_edges = np.count_nonzero(center_region)
        right_edges = np.count_nonzero(right_region)

        center_ratio = center_edges / total_edges if total_edges > 0 else 0
        spread_ratio = (left_edges + right_edges) / total_edges if total_edges > 0 else 0

        # ---- Classification Logic ----

        # Very high global edge density → likely wide
        if edge_density > GLOBAL_WIDE_THRESHOLD:
            wide_count += 1

        # Very low global density + strong center dominance → close
        elif edge_density < GLOBAL_CLOSE_THRESHOLD and center_ratio > 0.5:
            close_count += 1

        # Spatial dominance rules
        else:
            if center_ratio > 0.55:
                close_count += 1
            elif spread_ratio > 0.65 and edge_density > 0.05:
                wide_count += 1
            else:
                medium_count += 1


    return {
        "close_ratio": close_count / total_frames,
        "medium_ratio": medium_count / total_frames,
        "wide_ratio": wide_count / total_frames
    }


# -----------------------------------
# Direct test using sample.mp4
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
