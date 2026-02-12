from modules.preprocessing import extract_frames
import os

video_path = "sample.mp4"  # put a 10–15 sec clip in root folder

if not os.path.exists(video_path):
    print("Video file not found!")
else:
    video_data = extract_frames(video_path)

    print("Duration:", video_data["duration"])
    print("Frame Count:", video_data["frame_count"])
    print("FPS Used:", video_data["fps"])

    if video_data["frame_count"] > 0:
        frame = video_data["frames"][0]
        print("Frame Shape:", frame.shape)
        print("Data Type:", frame.dtype)
    