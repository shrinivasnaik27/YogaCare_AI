"""
YogaCare AI - Phase 1: Pose Detector Module
High-performance, standalone human pose estimation module for live webcam feeds.
Supports Ultralytics YOLO-Pose (primary) and MediaPipe PoseLandmarker backends.
Provides structured, model-agnostic landmark extraction and skeleton visualization
tailored for yoga posture evaluation in Phase 2.
"""

from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

import config

# Configure module-level logger
logger = logging.getLogger("YogaCareAI.PoseDetector")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class Landmark:
    """
    Standardized landmark representation for yoga posture analysis.

    Coordinate System:
    - x_norm, y_norm: Normalized coordinates in range [0.0, 1.0].
      x_norm = 0.0 is the left boundary, 1.0 is the right boundary.
      y_norm = 0.0 is the top boundary, 1.0 is the bottom boundary.
    - x_px, y_px: Discrete pixel coordinates in range [0, width - 1] and [0, height - 1].
    - z: Estimated depth relative to mid-hip point (negative indicates closer to camera).
      Default is 0.0 when 3D depth is not supported by the underlying model.
    - confidence: Detection or visibility score in range [0.0, 1.0].
    - is_valid: Boolean flag indicating if the landmark meets minimum confidence
      and lies within frame bounds. MUST be checked by Phase 2 before computing angles.
    """
    name: str
    x_norm: float
    y_norm: float
    x_px: int
    y_px: int
    z: float = 0.0
    confidence: float = 0.0
    is_valid: bool = False

    def as_pixel_tuple(self) -> Tuple[int, int]:
        """Returns integer pixel coordinates as (x, y)."""
        return self.x_px, self.y_px

    def as_norm_tuple(self) -> Tuple[float, float]:
        """Returns normalized coordinates as (x, y)."""
        return self.x_norm, self.y_norm

    def as_dict(self) -> Dict[str, Any]:
        """Returns landmark as a plain dictionary."""
        return {
            "name": self.name,
            "x_norm": round(self.x_norm, 4),
            "y_norm": round(self.y_norm, 4),
            "x_px": self.x_px,
            "y_px": self.y_px,
            "z": round(self.z, 4),
            "confidence": round(self.confidence, 4),
            "is_valid": self.is_valid,
        }


@dataclass
class PoseDetectionResult:
    """
    Structured outcome of pose estimation for a single video frame.

    Consumed by Phase 2 for posture evaluation and angle calculations.
    """
    annotated_frame: np.ndarray
    pose_detected: bool
    landmarks: Dict[str, Landmark] = field(default_factory=dict)
    status: str = config.STATUS_NO_PERSON
    warnings: List[str] = field(default_factory=list)
    missing_critical_landmarks: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    fps: float = 0.0

    def get_landmark(self, name: str) -> Optional[Landmark]:
        """
        Safely retrieves a landmark by its anatomical name.
        Returns None if not detected or unsupported.
        """
        return self.landmarks.get(name)

    def is_joint_valid(self, name: str) -> bool:
        """
        Helper for Phase 2: returns True if joint exists and meets visibility threshold.
        """
        lm = self.landmarks.get(name)
        return lm.is_valid if lm else False

    def are_joints_valid(self, *names: str) -> bool:
        """
        Helper for Phase 2: returns True only if ALL specified joints are valid.
        Crucial check prior to calculating 3-point joint angles (e.g. knee flexion).
        """
        return all(self.is_joint_valid(name) for name in names)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes result into a plain dictionary (useful for logs or IPC)."""
        return {
            "pose_detected": self.pose_detected,
            "status": self.status,
            "warnings": self.warnings,
            "missing_critical_landmarks": self.missing_critical_landmarks,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "fps": round(self.fps, 1),
            "landmarks": {name: lm.as_dict() for name, lm in self.landmarks.items()},
        }


# ==============================================================================
# MAIN POSE DETECTOR CLASS
# ==============================================================================

class YogaPoseDetector:
    """
    Reusable Human Pose Detector for YogaCare AI.

    Features:
    - Pretrained full-body pose estimation (Ultralytics YOLO-Pose or MediaPipe Tasks).
    - Robust handling of missing landmarks, out-of-frame limbs, and low visibility.
    - Dynamic skeleton visualization with joint confidence indicators.
    - Decoupled, model-agnostic dataclass output for seamless consumption by Phase 2.
    """

    def __init__(
        self,
        backend: str = config.DEFAULT_BACKEND,
        model_path: Optional[str] = None,
        min_detection_confidence: float = config.MIN_DETECTION_CONFIDENCE,
        min_landmark_confidence: float = config.MIN_LANDMARK_CONFIDENCE,
    ) -> None:
        """
        Initializes the detector with the chosen backend and confidence thresholds.

        Args:
            backend: "yolo" or "mediapipe". Defaults to config.DEFAULT_BACKEND.
            model_path: Path to weights file. If None, resolves from config.
            min_detection_confidence: Minimum score for overall person detection.
            min_landmark_confidence: Minimum confidence for individual landmark validity.
        """
        self.backend = backend.lower()
        self.min_detection_conf = min_detection_confidence
        self.min_landmark_conf = min_landmark_confidence
        self.model_path = model_path
        self._model = None
        self._is_initialized = False

        self._init_backend()

    def _init_backend(self) -> None:
        """Loads and initializes the underlying deep learning pose model."""
        if self.backend == "yolo":
            self._init_yolo()
        elif self.backend == "mediapipe":
            self._init_mediapipe()
        else:
            raise ValueError(
                f"Unsupported backend '{self.backend}'. Choose 'yolo' or 'mediapipe'."
            )
        self._is_initialized = True
        logger.info("Pose detector successfully initialized with backend: %s", self.backend)

    def _init_yolo(self) -> None:
        """Initializes Ultralytics YOLO-Pose model."""
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "Ultralytics package is not installed. Run 'pip install ultralytics'."
            ) from e

        resolved_path = self.model_path or config.YOLO_MODEL_PATH
        # If local model path doesn't exist, fall back to pretrained weights name
        if not Path(resolved_path).exists():
            logger.warning("Local YOLO model not found at %s. Using 'yolov8n-pose.pt'", resolved_path)
            resolved_path = "yolov8n-pose.pt"

        logger.info("Loading YOLO-Pose model from: %s", resolved_path)
        self._model = YOLO(resolved_path)

    def _init_mediapipe(self) -> None:
        """Initializes MediaPipe Tasks PoseLandmarker."""
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ImportError as e:
            raise ImportError(
                "MediaPipe package is not installed. Run 'pip install mediapipe'."
            ) from e

        resolved_path = self.model_path or config.MEDIAPIPE_MODEL_PATH
        if not Path(resolved_path).exists():
            raise FileNotFoundError(
                f"MediaPipe pose task asset not found at {resolved_path}. "
                "Please verify the model file exists."
            )

        logger.info("Loading MediaPipe PoseLandmarker from: %s", resolved_path)
        base_options = python.BaseOptions(model_asset_path=resolved_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=self.min_detection_conf,
            min_pose_presence_confidence=self.min_landmark_conf,
            min_tracking_confidence=self.min_landmark_conf,
            output_segmentation_masks=False,
        )
        self._model = vision.PoseLandmarker.create_from_options(options)

    # --------------------------------------------------------------------------
    # FRAME PROCESSING
    # --------------------------------------------------------------------------

    def process_frame(
        self,
        frame: np.ndarray,
        draw: bool = True,
    ) -> PoseDetectionResult:
        """
        Processes a single video frame and extracts full-body landmarks.

        Args:
            frame: Input video frame (BGR numpy array from OpenCV).
            draw: Whether to render skeleton and status overlays on output frame.

        Returns:
            PoseDetectionResult containing annotated frame, landmarks, status, and warnings.
        """
        start_time = time.perf_counter()

        # 1. Input validation
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            logger.warning("Received invalid or empty frame.")
            return PoseDetectionResult(
                annotated_frame=np.zeros((480, 640, 3), dtype=np.uint8) if frame is None else frame,
                pose_detected=False,
                status=config.STATUS_MODEL_ERROR,
                warnings=["Input frame is invalid or empty."],
            )

        h, w = frame.shape[:2]
        display_frame = frame.copy() if draw else frame

        # 2. Inference by backend
        try:
            if self.backend == "yolo":
                landmarks_dict, pose_detected = self._process_yolo(frame, w, h)
            else:
                landmarks_dict, pose_detected = self._process_mediapipe(frame, w, h)
        except Exception as e:
            logger.error("Inference failed with exception: %s", str(e), exc_info=True)
            return PoseDetectionResult(
                annotated_frame=display_frame,
                pose_detected=False,
                status=config.STATUS_MODEL_ERROR,
                warnings=[f"Inference error: {str(e)}"],
            )

        # 3. Posture completeness & yoga readiness check
        status, warnings, missing_critical = self._evaluate_posture_readiness(
            landmarks_dict, pose_detected
        )

        # 4. Measure execution time
        proc_time = (time.perf_counter() - start_time) * 1000.0  # ms
        current_fps = 1000.0 / proc_time if proc_time > 0 else 0.0

        # 5. Render visualization if requested
        if draw:
            self._draw_skeleton_and_hud(
                display_frame,
                landmarks_dict,
                status,
                warnings,
                current_fps,
                w,
                h,
            )

        return PoseDetectionResult(
            annotated_frame=display_frame,
            pose_detected=pose_detected,
            landmarks=landmarks_dict,
            status=status,
            warnings=warnings,
            missing_critical_landmarks=missing_critical,
            processing_time_ms=proc_time,
            fps=current_fps,
        )

    # --------------------------------------------------------------------------
    # BACKEND PROCESSORS
    # --------------------------------------------------------------------------

    def _process_yolo(
        self, frame: np.ndarray, w: int, h: int
    ) -> Tuple[Dict[str, Landmark], bool]:
        """Performs pose inference using Ultralytics YOLO-Pose."""
        # Run inference (YOLO handles BGR internally)
        results = self._model(frame, conf=self.min_detection_conf, verbose=False)
        landmarks: Dict[str, Landmark] = {}

        if not results or len(results) == 0:
            return landmarks, False

        result = results[0]
        if result.keypoints is None or len(result.keypoints) == 0:
            return landmarks, False

        # Select primary detected person (highest confidence detection)
        # xy shape: (N, 17, 2), conf shape: (N, 17)
        keypoints = result.keypoints
        if keypoints.xy.shape[0] == 0:
            return landmarks, False

        # Extract coordinates and confidence for the most prominent person (index 0)
        kp_xy = keypoints.xy[0].cpu().numpy()  # Pixel coordinates (17, 2)
        kp_conf = keypoints.conf[0].cpu().numpy() if keypoints.conf is not None else np.ones(17)

        for idx, name in config.COCO_KEYPOINTS.items():
            if idx >= len(kp_xy):
                continue
            x_px, y_px = float(kp_xy[idx][0]), float(kp_xy[idx][1])
            conf = float(kp_conf[idx]) if idx < len(kp_conf) else 0.0

            x_norm = max(0.0, min(1.0, x_px / float(w))) if w > 0 else 0.0
            y_norm = max(0.0, min(1.0, y_px / float(h))) if h > 0 else 0.0
            int_x = int(round(max(0, min(w - 1, x_px))))
            int_y = int(round(max(0, min(h - 1, y_px))))

            # A landmark is valid ONLY if confidence meets threshold AND coordinates are within frame
            is_valid = (
                conf >= self.min_landmark_conf
                and 0 <= int_x < w
                and 0 <= int_y < h
                and not (x_px == 0.0 and y_px == 0.0)  # Ignore unassigned (0,0) points
            )

            landmarks[name] = Landmark(
                name=name,
                x_norm=x_norm,
                y_norm=y_norm,
                x_px=int_x,
                y_px=int_y,
                z=0.0,  # 3D depth not supported in standard COCO 2D pose
                confidence=conf,
                is_valid=is_valid,
            )

        # Extended foot landmarks are not available in standard COCO 17-keypoint models
        for foot_name in config.EXTENDED_FOOT_LANDMARKS:
            landmarks[foot_name] = Landmark(
                name=foot_name,
                x_norm=0.0,
                y_norm=0.0,
                x_px=0,
                y_px=0,
                z=0.0,
                confidence=0.0,
                is_valid=False,
            )

        return landmarks, True

    def _process_mediapipe(
        self, frame: np.ndarray, w: int, h: int
    ) -> Tuple[Dict[str, Landmark], bool]:
        """Performs pose inference using MediaPipe Tasks PoseLandmarker."""
        import mediapipe as mp

        # Convert BGR frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = self._model.detect(mp_image)
        landmarks: Dict[str, Landmark] = {}

        if not detection_result or not detection_result.pose_landmarks:
            return landmarks, False

        # First detected pose
        first_pose = detection_result.pose_landmarks[0]

        for idx, mp_lm in enumerate(first_pose):
            if idx not in config.MEDIAPIPE_KEYPOINTS:
                continue
            name = config.MEDIAPIPE_KEYPOINTS[idx]

            x_norm = float(mp_lm.x)
            y_norm = float(mp_lm.y)
            z_coord = float(mp_lm.z) if hasattr(mp_lm, "z") else 0.0
            visibility = float(mp_lm.visibility) if hasattr(mp_lm, "visibility") else 1.0

            int_x = int(round(max(0, min(w - 1, x_norm * w))))
            int_y = int(round(max(0, min(h - 1, y_norm * h))))

            is_valid = (
                visibility >= self.min_landmark_conf
                and 0.0 <= x_norm <= 1.0
                and 0.0 <= y_norm <= 1.0
            )

            landmarks[name] = Landmark(
                name=name,
                x_norm=max(0.0, min(1.0, x_norm)),
                y_norm=max(0.0, min(1.0, y_norm)),
                x_px=int_x,
                y_px=int_y,
                z=z_coord,
                confidence=visibility,
                is_valid=is_valid,
            )

        return landmarks, True

    # --------------------------------------------------------------------------
    # POSTURE READINESS EVALUATION
    # --------------------------------------------------------------------------

    def _evaluate_posture_readiness(
        self,
        landmarks: Dict[str, Landmark],
        pose_detected: bool,
    ) -> Tuple[str, List[str], List[str]]:
        """
        Determines whether the visible body landmarks are sufficient for yoga posture analysis.
        Prevents falsely reporting valid full-body poses when critical limbs are hidden.
        """
        if not pose_detected or not landmarks:
            return (
                config.STATUS_NO_PERSON,
                ["No person detected. Please stand in front of the camera."],
                config.YOGA_CRITICAL_LANDMARKS.copy(),
            )

        # Check critical landmarks
        missing_critical: List[str] = []
        valid_critical_count = 0

        for crit in config.YOGA_CRITICAL_LANDMARKS:
            lm = landmarks.get(crit)
            if lm and lm.is_valid:
                valid_critical_count += 1
            else:
                missing_critical.append(crit)

        total_critical = len(config.YOGA_CRITICAL_LANDMARKS)
        critical_ratio = valid_critical_count / total_critical if total_critical > 0 else 0.0

        warnings: List[str] = []

        # Analyze anatomical visibility groups
        upper_missing = [k for k in config.UPPER_BODY_LANDMARKS if not landmarks.get(k, Landmark(k, 0, 0, 0, 0)).is_valid]
        lower_missing = [k for k in config.LOWER_BODY_LANDMARKS if not landmarks.get(k, Landmark(k, 0, 0, 0, 0)).is_valid]

        if lower_missing and len(lower_missing) >= 4:
            warnings.append("Lower body not visible. Please step back so your entire body is in frame.")
        elif "left_ankle" in lower_missing or "right_ankle" in lower_missing:
            warnings.append("Feet/ankles out of frame. Step back slightly for accurate stance detection.")

        if "left_wrist" in upper_missing or "right_wrist" in upper_missing:
            warnings.append("Hands/wrists partially obscured.")

        # Classify status
        if critical_ratio >= config.CRITICAL_LANDMARK_RATIO_THRESHOLD and len(missing_critical) <= 2:
            status = config.STATUS_VALID_FULL_BODY
            if not warnings:
                warnings.append("Full body clearly visible. Ready for posture guidance.")
        else:
            status = config.STATUS_PARTIAL_BODY
            if not warnings:
                warnings.append(f"Partial body detected ({valid_critical_count}/{total_critical} key joints visible).")

        return status, warnings, missing_critical

    # --------------------------------------------------------------------------
    # SKELETON VISUALIZATION & HUD
    # --------------------------------------------------------------------------

    def _draw_skeleton_and_hud(
        self,
        frame: np.ndarray,
        landmarks: Dict[str, Landmark],
        status: str,
        warnings: List[str],
        fps: float,
        w: int,
        h: int,
    ) -> None:
        """Draws the skeleton lines, joint keypoints, and diagnostic HUD overlay."""
        # 1. Draw Bones (Only if BOTH endpoints exist and are valid)
        for lm1_name, lm2_name, group in config.SKELETON_CONNECTIONS:
            lm1 = landmarks.get(lm1_name)
            lm2 = landmarks.get(lm2_name)

            if lm1 and lm2 and lm1.is_valid and lm2.is_valid:
                color_key = f"bone_{group}"
                bone_color = config.COLORS.get(color_key, config.COLORS["bone_default"])
                cv2.line(
                    frame,
                    lm1.as_pixel_tuple(),
                    lm2.as_pixel_tuple(),
                    bone_color,
                    config.BONE_THICKNESS,
                    lineType=cv2.LINE_AA,
                )

        # 2. Draw Joints
        for name, lm in landmarks.items():
            # Skip drawing invalid/unassigned foot landmarks in YOLO
            if name in config.EXTENDED_FOOT_LANDMARKS and not lm.is_valid:
                continue

            pt = lm.as_pixel_tuple()
            if lm.is_valid:
                joint_color = config.COLORS["joint_valid"]
                radius = config.JOINT_RADIUS
                # White outer border for contrast
                cv2.circle(frame, pt, radius + 2, config.COLORS["joint_border"], 1, lineType=cv2.LINE_AA)
                cv2.circle(frame, pt, radius, joint_color, config.JOINT_THICKNESS, lineType=cv2.LINE_AA)
            elif lm.confidence >= 0.20:
                # Low confidence (visible but below threshold)
                joint_color = config.COLORS["joint_low_conf"]
                cv2.circle(frame, pt, config.JOINT_RADIUS - 1, joint_color, config.JOINT_THICKNESS, lineType=cv2.LINE_AA)

        # 3. Draw Diagnostic HUD Banner (Top of frame)
        self._render_hud_banner(frame, status, warnings, fps, w)

    def _render_hud_banner(
        self,
        frame: np.ndarray,
        status: str,
        warnings: List[str],
        fps: float,
        w: int,
    ) -> None:
        """Renders an overlay header with status badge, warning message, and FPS."""
        hud_height = 80
        overlay = frame.copy()

        # Semi-transparent dark banner
        cv2.rectangle(overlay, (0, 0), (w, hud_height), config.COLORS["overlay_bg"], -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Status badge color
        if status == config.STATUS_VALID_FULL_BODY:
            status_color = config.COLORS["status_valid"]
            status_text = "FULL BODY DETECTED (READY)"
        elif status == config.STATUS_PARTIAL_BODY:
            status_color = config.COLORS["status_warning"]
            status_text = "PARTIAL BODY (STEP BACK)"
        else:
            status_color = config.COLORS["status_error"]
            status_text = "NO PERSON DETECTED"

        # Draw status circle pill & text
        cv2.circle(frame, (20, 26), 8, status_color, -1, lineType=cv2.LINE_AA)
        cv2.putText(
            frame,
            f"YogaCare AI  |  Status: {status_text}",
            (38, 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            config.COLORS["text_white"],
            2,
            lineType=cv2.LINE_AA,
        )

        # Primary warning or guidance message
        main_msg = warnings[0] if warnings else "System active."
        cv2.putText(
            frame,
            main_msg,
            (38, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            status_color if status != config.STATUS_VALID_FULL_BODY else config.COLORS["text_gray"],
            1,
            lineType=cv2.LINE_AA,
        )

        # FPS & Backend counter (top right)
        fps_text = f"FPS: {fps:4.1f} | Model: {self.backend.upper()}"
        text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
        cv2.putText(
            frame,
            fps_text,
            (w - text_size[0] - 20, 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            config.COLORS["bone_torso"],
            1,
            lineType=cv2.LINE_AA,
        )

    # --------------------------------------------------------------------------
    # RESOURCE CLEANUP & CONTEXT MANAGER
    # --------------------------------------------------------------------------

    def release(self) -> None:
        """Releases deep learning model resources and frees memory."""
        if hasattr(self, "_model") and self._model is not None:
            if self.backend == "mediapipe" and hasattr(self._model, "close"):
                try:
                    self._model.close()
                except Exception as e:
                    logger.debug("Error closing MediaPipe model: %s", e)
            self._model = None
            self._is_initialized = False
            logger.info("Pose detector resources released.")

    def __enter__(self) -> "YogaPoseDetector":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
