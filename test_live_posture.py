"""
YogaCare AI - Phase 2: Live Posture Guidance Test Application
Combines Phase 1 live pose estimation with Phase 2 real-time posture analysis,
joint angle calculation, hold timing, and corrective feedback.

Usage:
    python test_live_posture.py
    python test_live_posture.py --pose tree_pose
    python test_live_posture.py --pose warrior_ii
    python test_live_posture.py --pose auto
"""

import argparse
import sys
import time

import cv2

import config
from pose_detector import YogaPoseDetector
from posture_analyzer import PostureAnalyzer
from voice_assistant import VoiceAssistant


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YogaCare AI - Phase 2 Live Posture Guidance Application"
    )
    parser.add_argument(
        "--pose",
        type=str,
        default="warrior_ii",
        choices=["warrior_ii", "tree_pose", "auto"],
        help="Target yoga posture to evaluate (default: warrior_ii)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=config.DEFAULT_CAMERA_INDEX,
        help=f"Camera index (default: {config.DEFAULT_CAMERA_INDEX})",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=config.DEFAULT_BACKEND,
        choices=["yolo", "mediapipe"],
        help="Pose estimation model backend",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Max frames to process before exiting (0 for continuous streaming)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI window for automated testing",
    )
    parser.add_argument(
        "--no-voice",
        action="store_true",
        help="Disable spoken voice assistant feedback",
    )
    return parser.parse_args()


def run_posture_guidance(
    pose_name: str = "warrior_ii",
    camera_index: int = config.DEFAULT_CAMERA_INDEX,
    backend: str = config.DEFAULT_BACKEND,
    max_frames: int = 0,
    headless: bool = False,
    voice_enabled: bool = True,
) -> int:
    print("=" * 75)
    print("  YogaCare AI — Phase 2 & 3: Live Posture Analysis & Voice Guidance")
    print("=" * 75)
    print(f"[*] Target Pose           : {pose_name.upper()}")
    print(f"[*] Vision Backend        : {backend.upper()}")
    print(f"[*] Camera Device Index   : {camera_index}")
    print(f"[*] Voice Assistant       : {'ENABLED' if voice_enabled else 'DISABLED'}")
    print("-" * 75)
    print("  Hotkeys:")
    print("    [1] Switch to Tree Pose (Vrikshasana)")
    print("    [2] Switch to Warrior II (Virabhadrasana II)")
    print("    [A] Switch to Auto-Detect Pose")
    print("    [V] Toggle Voice Assistant On/Off")
    print("    [R] Reset Posture Hold Timer")
    print("    [Q] / [ESC] Exit Safely")
    print("-" * 75, flush=True)

    # 1. Initialize Engines
    try:
        detector = YogaPoseDetector(backend=backend)
        analyzer = PostureAnalyzer(default_pose=pose_name)
        voice = VoiceAssistant(enabled=voice_enabled)
    except Exception as e:
        print(f"\n[ERROR] Initialization failed: {e}", flush=True)
        return 1

    # 2. Open Camera
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"\n[ERROR] Unable to access camera at index {camera_index}.", flush=True)
        detector.release()
        voice.stop()
        return 2

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    window_name = "YogaCare AI — Posture Guidance (Phase 2 & 3)"
    if not headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    total_frames = 0
    fps_start = time.time()
    last_print = time.time()
    last_milestone = 0

    try:
        while True:
            success, frame = cap.read()
            if not success or frame is None:
                print("\n[WARNING] Webcam feed disconnected.", flush=True)
                break

            total_frames += 1
            frame = cv2.flip(frame, 1)  # Mirror view

            # Step 1: Detect pose skeleton (Phase 1)
            det_result = detector.process_frame(frame, draw=True)

            # Step 2: Analyze yoga posture angles & cues (Phase 2)
            eval_result = analyzer.evaluate_result(det_result)

            # Step 3: Voice Assistant Coaching (Phase 3)
            if eval_result.actionable_cues:
                top_cue = eval_result.actionable_cues[0]
                is_safety = "safety" in top_cue.lower()
                voice.speak(top_cue, priority=is_safety)

            # Check hold milestones (every 10 seconds of continuous hold)
            hold_sec = int(analyzer.current_hold_duration)
            if hold_sec >= 10 and hold_sec > last_milestone and hold_sec % 10 == 0:
                last_milestone = hold_sec
                voice.play_milestone_chime()
                voice.speak(f"Great hold! {hold_sec} seconds completed.", priority=True)
            elif hold_sec < 5:
                last_milestone = 0

            # Step 4: Render Phase 2 feedback cards and angle tags
            annotated_frame = analyzer.draw_posture_feedback(
                det_result.annotated_frame,
                det_result,
                eval_result,
            )

            # Periodic console report (every 3 seconds)
            if time.time() - last_print >= 3.0:
                last_print = time.time()
                print(
                    f"[{time.strftime('%H:%M:%S')}] {eval_result.pose_name.split(' (')[0]:<12} "
                    f"| Score: {eval_result.overall_score:4.0f}% "
                    f"| Hold: {analyzer.current_hold_duration:4.1f}s "
                    f"| Cue: {eval_result.actionable_cues[0] if eval_result.actionable_cues else 'Good form'}",
                    flush=True,
                )

            # Display window
            if not headless:
                cv2.imshow(window_name, annotated_frame)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), ord("Q"), 27):
                    print("\n[*] Quitting application...", flush=True)
                    break
                elif key == ord("1"):
                    analyzer.set_active_pose("tree_pose")
                    print("\n[*] Switched mode to: Tree Pose (Vrikshasana)", flush=True)
                    voice.speak("Switched to Tree Pose", priority=True)
                elif key == ord("2"):
                    analyzer.set_active_pose("warrior_ii")
                    print("\n[*] Switched mode to: Warrior II (Virabhadrasana II)", flush=True)
                    voice.speak("Switched to Warrior Two", priority=True)
                elif key in (ord("a"), ord("A")):
                    analyzer.set_active_pose("auto")
                    print("\n[*] Switched mode to: Auto-Detect Pose", flush=True)
                    voice.speak("Switched to auto detect mode", priority=True)
                elif key in (ord("v"), ord("V")):
                    voice.enabled = not voice.enabled
                    state_str = "ENABLED" if voice.enabled else "MUTED"
                    print(f"\n[*] Voice Assistant: {state_str}", flush=True)
                elif key in (ord("r"), ord("R")):
                    analyzer.reset_hold_timer()
                    last_milestone = 0
                    print("\n[*] Reset hold timer.", flush=True)

            if max_frames > 0 and total_frames >= max_frames:
                print(f"\n[*] Reached max frames limit ({max_frames}).", flush=True)
                break

    except KeyboardInterrupt:
        print("\n[*] Interrupted by user (Ctrl+C).", flush=True)

    finally:
        print("[*] Releasing camera...", flush=True)
        cap.release()
        print("[*] Closing windows...", flush=True)
        cv2.destroyAllWindows()
        print("[*] Stopping voice assistant...", flush=True)
        voice.stop()
        print("[*] Releasing detector resources...", flush=True)
        detector.release()
        print("[+] Application terminated cleanly.", flush=True)

    return 0


def main() -> None:
    args = parse_arguments()
    code = run_posture_guidance(
        pose_name=args.pose,
        camera_index=args.camera,
        backend=args.backend,
        max_frames=args.max_frames,
        headless=args.headless,
        voice_enabled=not args.no_voice,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
