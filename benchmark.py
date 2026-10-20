"""
YogaCare AI - Phase 4: Performance Benchmark & Profiling Tool
Profiles throughput, frame latency breakdown (inference, posture analysis, HUD rendering),
and sustained FPS on the host system.

Usage:
    python benchmark.py --iterations 100
"""

import argparse
from pathlib import Path
import sys
import time
from typing import Dict, List

import cv2
import numpy as np

import config
from pose_detector import YogaPoseDetector
from posture_analyzer import PostureAnalyzer


def run_benchmark(iterations: int = 100, backend: str = "yolo") -> Dict[str, float]:
    print("=" * 75)
    print(f"  YogaCare AI — Performance Benchmark ({backend.upper()} Backend)")
    print("=" * 75)
    print(f"[*] Benchmark Iterations   : {iterations}")
    print(f"[*] Frame Resolution       : {config.FRAME_WIDTH}x{config.FRAME_HEIGHT}")
    print("[-] Warming up model...", flush=True)

    # Initialize engines
    detector = YogaPoseDetector(backend=backend)
    analyzer = PostureAnalyzer(default_pose="warrior_ii")

    # Load synthetic or real test frame
    base_dir = Path(__file__).resolve().parent
    img_path = base_dir.parent / "virat-kohli.jpeg"
    if img_path.exists():
        test_frame = cv2.imread(str(img_path))
        test_frame = cv2.resize(test_frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
    else:
        test_frame = np.random.randint(0, 256, (config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)

    # Warmup runs (5 frames)
    for _ in range(5):
        det = detector.process_frame(test_frame, draw=False)
        _ = analyzer.evaluate_result(det)

    print("[+] Warmup complete. Profiling pipeline...", flush=True)

    t_detect_list: List[float] = []
    t_analyze_list: List[float] = []
    t_render_list: List[float] = []
    t_total_list: List[float] = []

    total_start = time.perf_counter()

    for i in range(iterations):
        frame_start = time.perf_counter()

        # Step 1: Vision Inference & Landmark Extraction
        t0 = time.perf_counter()
        det = detector.process_frame(test_frame, draw=False)
        t_detect = (time.perf_counter() - t0) * 1000.0

        # Step 2: Posture Analysis & Biomechanical Angle Checks
        t1 = time.perf_counter()
        eval_res = analyzer.evaluate_result(det)
        t_analyze = (time.perf_counter() - t1) * 1000.0

        # Step 3: HUD Card & Skeleton Rendering
        t2 = time.perf_counter()
        _ = analyzer.draw_posture_feedback(test_frame.copy(), det, eval_res)
        t_render = (time.perf_counter() - t2) * 1000.0

        frame_total = (time.perf_counter() - frame_start) * 1000.0

        t_detect_list.append(t_detect)
        t_analyze_list.append(t_analyze)
        t_render_list.append(t_render)
        t_total_list.append(frame_total)

    total_time = time.perf_counter() - total_start
    avg_fps = iterations / total_time

    # Clean up
    detector.release()

    # Results calculation
    results = {
        "avg_detect_ms": float(np.mean(t_detect_list)),
        "avg_analyze_ms": float(np.mean(t_analyze_list)),
        "avg_render_ms": float(np.mean(t_render_list)),
        "avg_total_ms": float(np.mean(t_total_list)),
        "min_latency_ms": float(np.min(t_total_list)),
        "max_latency_ms": float(np.max(t_total_list)),
        "p95_latency_ms": float(np.percentile(t_total_list, 95)),
        "sustained_fps": float(avg_fps),
    }

    print("\n" + "-" * 75)
    print("  STAGE LATENCY BREAKDOWN (Average)")
    print("-" * 75)
    print(f"  • Pose Estimation (Detection) : {results['avg_detect_ms']:6.2f} ms")
    print(f"  • Posture Biomechanics Logic  : {results['avg_analyze_ms']:6.2f} ms")
    print(f"  • HUD & Skeleton Rendering    : {results['avg_render_ms']:6.2f} ms")
    print(f"  • Total Pipeline Latency      : {results['avg_total_ms']:6.2f} ms")
    print("-" * 75)
    print(f"  • P95 Latency                 : {results['p95_latency_ms']:6.2f} ms")
    print(f"  • Min / Max Latency           : {results['min_latency_ms']:.2f} ms / {results['max_latency_ms']:.2f} ms")
    print(f"  • Sustained Throughput        : {results['sustained_fps']:6.1f} FPS")
    print("=" * 75)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YogaCare AI Benchmark")
    parser.add_argument("--iterations", type=int, default=50, help="Number of benchmark iterations")
    parser.add_argument("--backend", type=str, default="yolo", choices=["yolo", "mediapipe"], help="Backend to benchmark")
    args = parser.parse_args()

    run_benchmark(iterations=args.iterations, backend=args.backend)
