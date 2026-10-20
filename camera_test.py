"""
YogaCare AI - Phase 1: Camera Test Application
Standalone live video test application for webcam capture, real-time pose detection,
skeleton visualization, and joint confidence validation.

Usage:
    python camera_test.py
    python camera_test.py --backend yolo --camera 0
    python camera_test.py --backend mediapipe
"""

import argparse
import sys
import time

import cv2

import config
from pose_detector import YogaPoseDetector


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the test runner."""
    parser = argparse.ArgumentParser(
        description="YogaCare AI - Live Webcam Pose Detection Test (Phase 1)"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=config.DEFAULT_CAMERA_INDEX,
        help=f"Index of video capture device (default: {config.DEFAULT_CAMERA_INDEX})",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=config.DEFAULT_BACKEND,
        choices=["yolo", "mediapipe"],
        help=f"Pose estimation backend (default: {config.DEFAULT_BACKEND})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional custom path to model weights or task file",
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=config.MIN_LANDMARK_CONFIDENCE,
        help=f"Minimum landmark confidence threshold (default: {config.MIN_LANDMARK_CONFIDENCE})",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum frames to process before exiting (0 for continuous streaming)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying OpenCV GUI window (useful for headless/CI testing)",
    )
    return parser.parse_args()


def run_camera_stream(
    camera_index: int = config.DEFAULT_CAMERA_INDEX,
    backend: str = config.DEFAULT_BACKEND,
    model_path: str = None,
    min_landmark_conf: float = config.MIN_LANDMARK_CONFIDENCE,
    max_frames: int = 0,
    headless: bool = False,
) -> int:
    """
    Initializes the pose detector, streams from webcam, and displays live skeleton overlay.

    Returns:
        Exit code (0 for clean exit, non-zero for error).
    """
    print("=" * 70)
    print("  YogaCare AI - Phase 1: Live Webcam Pose Detection")
    print("=" * 70)
    print(f"[*] Initializing Pose Detector Backend : {backend.upper()}")
    print(f"[*] Camera Index                     : {camera_index}")
    print(f"[*] Landmark Confidence Threshold   : {min_landmark_conf}")
    print(f"[*] Max Frames (0=unlimited)         : {max_frames}")
    print(f"[*] Headless Mode (no GUI window)   : {headless}")
    print("-" * 70)

    # 1. Initialize Pose Detector
    try:
        detector = YogaPoseDetector(
            backend=backend,
            model_path=model_path,
            min_landmark_confidence=min_landmark_conf,
        )
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize pose detector: {e}")
        return 1

    # 2. Open Webcam
    print(f"[*] Attempting to open webcam at index {camera_index}...")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)

    # Fallback to default backend if CAP_DSHOW fails
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"\n[ERROR] Could not open webcam at index {camera_index}.")
        print("  Troubleshooting steps:")
        print("  1. Verify your webcam is physically connected and not covered.")
        print("  2. Ensure other apps (Teams, Zoom, Discord, Windows Camera) are closed.")
        print("  3. Check Windows Camera Privacy Settings (Settings > Privacy > Camera).")
        print("  4. Try specifying a different index: python camera_test.py --camera 1")
        detector.release()
        return 2

    # Configure camera resolution for optimal performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[+] Webcam successfully opened! Resolution: {actual_w}x{actual_h}")
    print("[*] Controls: Press 'q' or 'ESC' in the video window to safely exit.")
    print("-" * 70)

    window_name = "YogaCare AI - Pose Detection (Phase 1)"
    if not headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    total_processed = 0
    fps_start_time = time.time()
    smoothed_fps = 0.0
    last_status = None

    try:
        while True:
            success, frame = cap.read()
            if not success or frame is None:
                print("\n[WARNING] Failed to grab frame from webcam. Stream interrupted.")
                break

            frame_count += 1
            total_processed += 1

            # Flip frame horizontally for intuitive mirror view
            frame = cv2.flip(frame, 1)

            # Process frame with detector
            result = detector.process_frame(frame, draw=True)

            # Maintain smoothed FPS display
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                smoothed_fps = frame_count / elapsed
                frame_count = 0
                fps_start_time = time.time()

            # Periodic console diagnostic update (every status change or 5 seconds)
            if result.status != last_status or total_processed % 30 == 0:
                last_status = result.status
                crit_visible = sum(1 for c in config.YOGA_CRITICAL_LANDMARKS if result.is_joint_valid(c))
                print(
                    f"[{time.strftime('%H:%M:%S')}] Frame {total_processed:4d} | Status: {result.status:<20} "
                    f"| Critical Joints: {crit_visible:2d}/{len(config.YOGA_CRITICAL_LANDMARKS)} "
                    f"| Latency: {result.processing_time_ms:5.1f}ms "
                    f"| Warnings: {', '.join(result.warnings) if result.warnings else 'None'}"
                )

            # Display annotated frame
            if not headless:
                cv2.imshow(window_name, result.annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    print("\n[*] Exit requested by user ('q' or ESC pressed).")
                    break

            if max_frames > 0 and total_processed >= max_frames:
                print(f"\n[*] Reached target maximum frames ({max_frames}). Exiting test.")
                break

    except KeyboardInterrupt:
        print("\n[*] Interrupted by user (Ctrl+C).")

    finally:
        # Cleanly release all hardware and window resources
        print("[*] Releasing webcam...", flush=True)
        cap.release()
        print("[*] Closing OpenCV windows...", flush=True)
        cv2.destroyAllWindows()
        print("[*] Releasing detector models...", flush=True)
        detector.release()
        print("[+] Cleanup complete. Application terminated safely.", flush=True)

    return 0


def main() -> None:
    args = parse_arguments()
    exit_code = run_camera_stream(
        camera_index=args.camera,
        backend=args.backend,
        model_path=args.model,
        min_landmark_conf=args.min_conf,
        max_frames=args.max_frames,
        headless=args.headless,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
