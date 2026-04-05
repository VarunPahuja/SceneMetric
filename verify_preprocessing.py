"""
verify_preprocessing.py
-----------------------
Visual verification tool for preprocessing pipeline.
Shows before/after for each preprocessing step to validate quality.
"""

import cv2
import numpy as np
import sys

from modules.preprocessing import (
    load_video,
    letterbox_resize,
    normalize_brightness,
    stabilize_color,
    is_blurry,
    _quick_blur_check,
    preprocess_frame,
    BLUR_THRESHOLD,
    MAX_DURATION,
    SAMPLING_FPS,
    TARGET_LONG_SIDE,
    SCALE_CLAMP,
)


def visualize_step_by_step(video_path, sample_every=10):
    """
    Display preprocessing pipeline step-by-step for verification.
    Shows: raw → letterbox → CLAHE → color stabilization → final
    
    Args:
        video_path: path to video file
        sample_every: show every Nth frame (set to 1 for all frames)
    """
    cap = load_video(video_path)
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = 30.0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = min(total_frames / original_fps, MAX_DURATION)
    
    print("="*80)
    print("PREPROCESSING VERIFICATION")
    print("="*80)
    print(f"Video: {video_path}")
    print(f"Original FPS: {original_fps:.2f}")
    print(f"Target resolution: {TARGET_LONG_SIDE}x{TARGET_LONG_SIDE}")
    print(f"Sampling rate: {SAMPLING_FPS} FPS")
    print(f"Blur threshold: {BLUR_THRESHOLD}")
    print(f"Scale clamp: {SCALE_CLAMP}")
    print(f"Duration: {duration:.2f}s")
    print()
    print("Controls: Press any key for next frame, 'q' to quit")
    print("="*80)
    print()
    
    frame_index = 0
    displayed = 0
    skipped_quick_blur = 0
    skipped_final_blur = 0
    accepted = 0
    
    sample_interval = 1.0 / SAMPLING_FPS
    samples_processed = 0
    
    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break
        
        timestamp = frame_index / original_fps
        if timestamp > duration:
            break
        
        # Check if this frame should be sampled
        next_sample_t = samples_processed * sample_interval
        
        if timestamp >= next_sample_t:
            samples_processed += 1
            
            # Only display every Nth sampled frame
            if (samples_processed - 1) % sample_every != 0:
                frame_index += 1
                continue
            
            displayed += 1
            
            # === Step 0: Original raw frame ===
            original = frame_bgr.copy()
            
            # === Step 1: Quick blur check ===
            quick_blur_failed = _quick_blur_check(frame_bgr)
            if quick_blur_failed:
                skipped_quick_blur += 1
            
            # === Step 2: Letterbox resize ===
            canvas, pad_top, pad_left, content_h, content_w = letterbox_resize(frame_bgr)
            letterboxed = canvas.copy()
            
            # === Step 3: CLAHE normalization ===
            after_clahe = normalize_brightness(canvas)
            
            # === Step 4: Final blur check ===
            final_blur_failed = is_blurry(after_clahe)
            if final_blur_failed and not quick_blur_failed:
                skipped_final_blur += 1
            
            # === Step 5: Color stabilization ===
            after_stabilization = stabilize_color(after_clahe, pad_top, pad_left, content_h, content_w)
            
            # === Step 6: Final RGB ===
            final_rgb = cv2.cvtColor(after_stabilization, cv2.COLOR_BGR2RGB)
            final_display = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
            
            if not quick_blur_failed and not final_blur_failed:
                accepted += 1
            
            # === Create comparison grid ===
            # Resize everything to same size for display
            display_size = 400
            
            def resize_for_display(img):
                return cv2.resize(img, (display_size, display_size), interpolation=cv2.INTER_AREA)
            
            img1 = resize_for_display(original)
            img2 = resize_for_display(letterboxed)
            img3 = resize_for_display(after_clahe)
            img4 = resize_for_display(after_stabilization)
            img5 = resize_for_display(final_display)
            
            # Add labels
            def add_label(img, text, color=(255, 255, 255), status_color=None):
                labeled = img.copy()
                # Black background for text
                cv2.rectangle(labeled, (0, 0), (display_size, 40), (0, 0, 0), -1)
                cv2.putText(labeled, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, color, 2)
                # Optional status indicator
                if status_color:
                    cv2.rectangle(labeled, (display_size - 50, 5), (display_size - 10, 35), 
                                 status_color, -1)
                return labeled
            
            # Status colors
            quick_blur_color = (0, 0, 255) if quick_blur_failed else (0, 255, 0)
            final_blur_color = (0, 0, 255) if final_blur_failed else (0, 255, 0)
            accept_color = (0, 255, 0) if (not quick_blur_failed and not final_blur_failed) else (0, 0, 255)
            
            img1 = add_label(img1, "1. RAW (Original)")
            img2 = add_label(img2, "2. LETTERBOX")
            img3 = add_label(img3, "3. CLAHE", status_color=final_blur_color)
            img4 = add_label(img4, "4. COLOR STABILIZED")
            img5 = add_label(img5, "5. FINAL (RGB)", status_color=accept_color)
            
            # Create 2x3 grid (5 images + info panel)
            row1 = np.hstack([img1, img2, img3])
            
            # Info panel
            info_panel = np.zeros((display_size, display_size, 3), dtype=np.uint8)
            y = 30
            
            def add_info_text(text, color=(255, 255, 255)):
                nonlocal y
                cv2.putText(info_panel, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, color, 1, cv2.LINE_AA)
                y += 25
            
            add_info_text(f"Frame: {frame_index}", (200, 200, 200))
            add_info_text(f"Time: {timestamp:.2f}s", (200, 200, 200))
            add_info_text(f"Sample: {samples_processed}", (200, 200, 200))
            y += 10
            
            add_info_text("BLUR CHECKS:", (255, 255, 0))
            quick_status = "REJECTED" if quick_blur_failed else "PASSED"
            quick_color = (0, 0, 255) if quick_blur_failed else (0, 255, 0)
            add_info_text(f"Quick: {quick_status}", quick_color)
            
            final_status = "REJECTED" if final_blur_failed else "PASSED"
            final_color = (0, 0, 255) if final_blur_failed else (0, 255, 0)
            add_info_text(f"Final: {final_status}", final_color)
            y += 10
            
            accept_status = "ACCEPTED" if (not quick_blur_failed and not final_blur_failed) else "REJECTED"
            add_info_text(f"Result: {accept_status}", accept_color)
            y += 20
            
            add_info_text("STATISTICS:", (255, 255, 0))
            add_info_text(f"Displayed: {displayed}")
            add_info_text(f"Accepted: {accepted}", (0, 255, 0))
            add_info_text(f"Quick blur: {skipped_quick_blur}", (255, 100, 0))
            add_info_text(f"Final blur: {skipped_final_blur}", (200, 0, 0))
            total_rejected = skipped_quick_blur + skipped_final_blur
            add_info_text(f"Total rejected: {total_rejected}", (0, 0, 255))
            
            if accepted + total_rejected > 0:
                accept_rate = 100 * accepted / (accepted + total_rejected)
                add_info_text(f"Accept rate: {accept_rate:.1f}%", (150, 150, 255))
            
            row2 = np.hstack([img4, img5, info_panel])
            
            display = np.vstack([row1, row2])
            
            # Show
            cv2.imshow("Preprocessing Pipeline Verification", display)
            
            key = cv2.waitKey(0)
            if key == ord('q') or key == 27:
                break
        
        frame_index += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Total frames displayed: {displayed}")
    print(f"Accepted frames: {accepted} ({100 * accepted / max(displayed, 1):.1f}%)")
    print(f"Rejected by quick blur check: {skipped_quick_blur}")
    print(f"Rejected by final blur check: {skipped_final_blur}")
    print(f"Total rejected: {skipped_quick_blur + skipped_final_blur}")
    print()
    
    if skipped_quick_blur + skipped_final_blur > accepted:
        print("⚠️  WARNING: More frames rejected than accepted!")
        print("    Consider lowering BLUR_THRESHOLD or adjusting blur check logic")
    elif skipped_quick_blur + skipped_final_blur == 0:
        print("⚠️  NOTE: No frames rejected for blur")
        print("    Blur threshold may be too permissive")
    else:
        print("✓ Rejection rate looks reasonable")
    
    print("="*80)


def compare_before_after(video_path):
    """
    Simple side-by-side comparison: raw input vs final preprocessed output.
    Useful for quick quality check.
    """
    from modules.preprocessing import extract_frames
    
    cap = load_video(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    print("Processing video...")
    result = extract_frames(video_path)
    frames_rgb = result["frames"]
    
    print(f"\n{'='*60}")
    print(f"Extracted {result['frame_count']} frames")
    print(f"Skipped {result['skipped_blur']} blurry frames")
    print(f"FPS: {result['fps']}, Duration: {result['duration']:.2f}s")
    print(f"{'='*60}\n")
    
    if not frames_rgb:
        print("No frames extracted!")
        return
    
    # Get corresponding raw frames
    sample_interval = 1.0 / SAMPLING_FPS
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    frame_index = 0
    raw_frames = []
    samples_collected = 0
    
    while cap.isOpened() and len(raw_frames) < len(frames_rgb) + result['skipped_blur']:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        
        timestamp = frame_index / original_fps
        next_sample_t = samples_collected * sample_interval
        
        if timestamp >= next_sample_t:
            samples_collected += 1
            raw_frames.append(frame_bgr)
        
        frame_index += 1
    
    cap.release()
    
    print(f"Showing {len(frames_rgb)} processed frames. Press any key to continue, 'q' to quit.\n")
    
    processed_idx = 0
    for i, raw_bgr in enumerate(raw_frames):
        # Check if this frame was kept
        processed = preprocess_frame(raw_bgr.copy())
        
        if processed is None:
            status = "REJECTED (blur)"
            status_color = (0, 0, 255)
            display_processed = np.zeros_like(raw_bgr)
            cv2.putText(display_processed, "REJECTED", 
                       (50, display_processed.shape[0]//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        else:
            status = "ACCEPTED"
            status_color = (0, 255, 0)
            display_processed = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
            processed_idx += 1
        
        # Resize for side-by-side
        h, w = raw_bgr.shape[:2]
        display_h = 600
        display_w = int(w * display_h / h)
        
        raw_display = cv2.resize(raw_bgr, (display_w, display_h))
        proc_display = cv2.resize(display_processed, (display_w, display_h))
        
        # Add labels
        cv2.rectangle(raw_display, (0, 0), (200, 40), (0, 0, 0), -1)
        cv2.putText(raw_display, "RAW INPUT", (10, 28), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.rectangle(proc_display, (0, 0), (300, 40), (0, 0, 0), -1)
        cv2.putText(proc_display, f"PREPROCESSED - {status}", (10, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        combined = np.hstack([raw_display, proc_display])
        
        cv2.imshow("Before/After Comparison", combined)
        
        key = cv2.waitKey(0)
        if key == ord('q') or key == 27:
            break
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python verify_preprocessing.py <video_path> [mode] [sample_rate]")
        print()
        print("Modes:")
        print("  step  - Show step-by-step preprocessing (default)")
        print("  compare - Show before/after comparison")
        print()
        print("Examples:")
        print("  python verify_preprocessing.py local_videos/sample.mp4")
        print("  python verify_preprocessing.py local_videos/sample.mp4 step 5")
        print("  python verify_preprocessing.py local_videos/sample.mp4 compare")
        sys.exit(1)
    
    video_path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "step"
    sample_rate = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    
    if mode == "compare":
        compare_before_after(video_path)
    else:
        visualize_step_by_step(video_path, sample_rate)
