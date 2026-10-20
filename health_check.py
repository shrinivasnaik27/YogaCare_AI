"""
YogaCare AI - Phase 4: System Health Check & Pre-Flight Diagnostic
Verifies hardware availability, model weights integrity, package dependencies,
and audio capabilities prior to demonstration or deployment.

Usage:
    python health_check.py
"""

from pathlib import Path
import sys
import time

import cv2
import numpy as np


def run_diagnostics() -> int:
    print("=" * 75)
    print("  YogaCare AI — System Diagnostic & Pre-Flight Health Check")
    print("=" * 75)
    all_passed = True
    base_dir = Path(__file__).resolve().parent

    # 1. Python Environment Check
    py_ver = sys.version.split()[0]
    print(f"\n[1/5] Python Environment")
    print(f"  • Python Version : {py_ver}")
    if sys.version_info >= (3, 10):
        print("  • Status         : [PASS] Compatible (Python >= 3.10)")
    else:
        print("  • Status         : [FAIL] Incompatible Python version")
        all_passed = False

    # 2. Package Dependencies Check
    print(f"\n[2/5] Package Dependencies")
    dependencies = [
        ("cv2", "OpenCV Image & Video Processing"),
        ("numpy", "NumPy Numerical Computing"),
        ("ultralytics", "Ultralytics YOLO Architecture"),
        ("torch", "PyTorch Neural Inference Engine"),
        ("pyttsx3", "Text-to-Speech Voice Assistant"),
        ("streamlit", "Streamlit Web Application Framework"),
        ("mediapipe", "MediaPipe Vision Tasks (Optional)"),
    ]

    for mod_name, desc in dependencies:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "Installed")
            print(f"  • {mod_name:<12} ({desc}) : [PASS] v{ver}")
        except ImportError:
            if mod_name == "mediapipe":
                print(f"  • {mod_name:<12} ({desc}) : [OPTIONAL] Not installed (YOLO primary)")
            else:
                print(f"  • {mod_name:<12} ({desc}) : [FAIL] Not installed")
                all_passed = False

    # 3. Model Weights Integrity Check
    print(f"\n[3/5] Pretrained Model Assets")
    yolo_model = base_dir / "models" / "yolov8n-pose.pt"
    mp_model = base_dir / "models" / "pose_landmarker_lite.task"

    if yolo_model.exists() and yolo_model.stat().st_size > 1_000_000:
        size_mb = yolo_model.stat().st_size / (1024 * 1024)
        print(f"  • YOLOv8-Pose Weights       : [PASS] Found ({size_mb:.1f} MB)")
    else:
        print(f"  • YOLOv8-Pose Weights       : [FAIL] Missing or corrupted at {yolo_model}")
        all_passed = False

    if mp_model.exists() and mp_model.stat().st_size > 1_000_000:
        size_mb = mp_model.stat().st_size / (1024 * 1024)
        print(f"  • MediaPipe Task Model      : [PASS] Found ({size_mb:.1f} MB)")
    else:
        print(f"  • MediaPipe Task Model      : [WARN] Optional model not found")

    # 4. Webcam Hardware Check
    print(f"\n[4/5] Camera Hardware Access")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if cap.isOpened():
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  • Default Webcam (Index 0)  : [PASS] Accessible (Resolution: {w}x{h})")
        cap.release()
    else:
        print("  • Default Webcam (Index 0)  : [WARN] Camera unavailable (in use or disconnected)")

    # 5. Audio Synthesis Check
    print(f"\n[5/5] Audio & Voice Synthesis")
    try:
        from voice_assistant import VoiceAssistant
        va = VoiceAssistant(enabled=True)
        print("  • Text-to-Speech Engine     : [PASS] Initialized successfully")
        va.stop()
    except Exception as e:
        print(f"  • Text-to-Speech Engine     : [FAIL] Audio error: {e}")
        all_passed = False

    # Final Verdict
    print("\n" + "=" * 75)
    if all_passed:
        print("  [SYSTEM HEALTH: OPTIMAL] ALL REQUIRED CHECKS PASSED!")
        print("  YogaCare AI is ready for demonstration, live testing, and viva evaluation.")
    else:
        print("  [SYSTEM HEALTH: WARNING] One or more required components need attention.")
    print("=" * 75)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(run_diagnostics())
