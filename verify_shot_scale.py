"""
verify_shot_scale.py
-------------------
Visual verification tool for shot scale classifications.
Displays sampled frames with their predicted labels overlaid.
"""

import cv2
import json
from modules.preprocessing import extract_frames
from modules.shot_scale import process_video

def verify_classifications(video_path, sample_every=5):
    """
    Display frames with their classifications for visual verification.
    Press any key to advance to next frame, 'q' to quit.
    """
    # Run classification
    print("Processing video...")
    result = process_video(video_path)
    
    # Extract frames
    print("Loading frames...")
    video_data = extract_frames(video_path)
    frames = video_data["frames"]
    
    print(f"\nShowing every {sample_every}th frame. Press any key to continue, 'q' to quit.\n")
    
    for i, frame_result in enumerate(result["frames"]):
        if i % sample_every != 0 and i != len(result["frames"]) - 1:
            continue
        
        frame_idx = frame_result["frame_index"]
        frame = frames[frame_idx].copy()
        
        # Get classification info
        raw = frame_result["raw_label"]
        smooth = frame_result["smooth_label"]
        conf = frame_result["confidence"]
        scores = frame_result["scores"]
        features = frame_result["features"]
        
        # Overlay info on frame
        h, w = frame.shape[:2]
        
        # Color coding
        color = {
            "CLOSE": (0, 255, 0),    # Green
            "MEDIUM": (0, 165, 255), # Orange
            "WIDE": (0, 0, 255)      # Red
        }[smooth]
        
        # Draw background rectangle for text
        cv2.rectangle(frame, (10, 10), (w - 10, 200), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (w - 10, 200), color, 3)
        
        # Add text
        y = 35
        cv2.putText(frame, f"Frame: {frame_idx}/{len(result['frames'])-1}", 
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 30
        cv2.putText(frame, f"Classification: {smooth} (conf: {conf:.3f})", 
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        y += 30
        cv2.putText(frame, f"Scores - C:{scores['CLOSE']:.2f} M:{scores['MEDIUM']:.2f} W:{scores['WIDE']:.2f}", 
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y += 25
        cv2.putText(frame, f"Face: {features['face_ratio']:.3f} | Subject: {features['subject_coverage']:.3f}", 
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y += 25
        cv2.putText(frame, f"Depth ratio: {features['depth_ratio']:.2f} | Entropy: {features['entropy_score']:.3f}", 
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        if raw != smooth:
            y += 30
            cv2.putText(frame, f"(Raw: {raw})", 
                       (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 1)
        
        # Display
        cv2.imshow("Shot Scale Verification", frame)
        
        key = cv2.waitKey(0)
        if key == ord('q') or key == 27:  # q or ESC
            break
    
    cv2.destroyAllWindows()
    
    # Print summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    print(f"Total frames: {result['frame_count']}")
    print(f"Dominant scale: {result['dominant_scale']}")
    print(f"Distribution: {result['scale_ratios']}")
    print("\nLook for these validation signs:")
    print("  ✓ CLOSE: faces visible, subject fills frame, blurred background")
    print("  ✓ MEDIUM: subject clearly visible but with context/space")
    print("  ✓ WIDE: subject small, lots of environment/context visible")
    print("="*70)


if __name__ == "__main__":
    import sys
    
    video_path = sys.argv[1] if len(sys.argv) > 1 else "local_videos/sample.mp4"
    sample_rate = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    verify_classifications(video_path, sample_rate)
