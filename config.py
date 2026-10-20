"""
YogaCare AI - Phase 1: Configuration Module
Defines settings, landmark mappings, thresholds, and visualization parameters
for real-time human pose estimation and yoga posture tracking.
"""

from pathlib import Path
from typing import Dict, List, Tuple

# ==============================================================================
# 1. MODEL & BACKEND CONFIGURATION
# ==============================================================================

# Backend choice: "yolo" (Ultralytics YOLOv8-Pose) or "mediapipe" (MediaPipe Tasks)
# Default is "yolo", which provides real-time CPU performance and matches the
# AI Workshop Ultralytics environment.
DEFAULT_BACKEND: str = "yolo"

# Base directory for relative paths
BASE_DIR: Path = Path(__file__).resolve().parent

# Model weights paths
YOLO_MODEL_PATH: str = str(BASE_DIR / "models" / "yolov8n-pose.pt")
MEDIAPIPE_MODEL_PATH: str = str(BASE_DIR / "models" / "pose_landmarker_lite.task")

# ==============================================================================
# 2. CONFIDENCE & VALIDITY THRESHOLDS
# ==============================================================================

# Minimum confidence required for overall person detection (bounding box / pose)
MIN_DETECTION_CONFIDENCE: float = 0.50

# Minimum confidence required for an individual landmark to be considered valid
MIN_LANDMARK_CONFIDENCE: float = 0.40

# Fraction of critical landmarks that must be valid to declare a valid full body pose
CRITICAL_LANDMARK_RATIO_THRESHOLD: float = 0.80

# ==============================================================================
# 3. CAMERA & STREAM SETTINGS
# ==============================================================================

DEFAULT_CAMERA_INDEX: int = 0
FRAME_WIDTH: int = 1280
FRAME_HEIGHT: int = 720
TARGET_FPS: int = 30

# ==============================================================================
# 4. LANDMARK DEFINITIONS & MAPPINGS
# ==============================================================================

# COCO 17-Keypoint Index Mapping (Standard for YOLO Pose models)
COCO_KEYPOINTS: Dict[int, str] = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}

# MediaPipe 33-Keypoint Index Mapping (Extended keypoints including feet & hands)
MEDIAPIPE_KEYPOINTS: Dict[int, str] = {
    0: "nose",
    1: "left_eye_inner",
    2: "left_eye",
    3: "left_eye_outer",
    4: "right_eye_inner",
    5: "right_eye",
    6: "right_eye_outer",
    7: "left_ear",
    8: "right_ear",
    9: "mouth_left",
    10: "mouth_right",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    17: "left_pinky",
    18: "right_pinky",
    19: "left_index",
    20: "right_index",
    21: "left_thumb",
    22: "right_thumb",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index",
}

# Critical landmarks essential for yoga posture analysis (e.g. Tree Pose, Warrior II)
# Phase 2 posture evaluators depend on these joints
YOGA_CRITICAL_LANDMARKS: List[str] = [
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# Upper body joints (used for partial-body classification)
UPPER_BODY_LANDMARKS: List[str] = [
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]

# Lower body joints (critical for standing yoga poses)
LOWER_BODY_LANDMARKS: List[str] = [
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# Extended foot landmarks (supported in MediaPipe, optional in YOLO)
EXTENDED_FOOT_LANDMARKS: List[str] = [
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]

# Skeleton connections to draw between landmark pairs (landmark_1, landmark_2, connection_group)
SKELETON_CONNECTIONS: List[Tuple[str, str, str]] = [
    # Head / Face
    ("nose", "left_eye", "head"),
    ("left_eye", "left_ear", "head"),
    ("nose", "right_eye", "head"),
    ("right_eye", "right_ear", "head"),
    # Torso
    ("left_shoulder", "right_shoulder", "torso"),
    ("left_shoulder", "left_hip", "torso"),
    ("right_shoulder", "right_hip", "torso"),
    ("left_hip", "right_hip", "torso"),
    # Left Arm
    ("left_shoulder", "left_elbow", "left_arm"),
    ("left_elbow", "left_wrist", "left_arm"),
    # Right Arm
    ("right_shoulder", "right_elbow", "right_arm"),
    ("right_elbow", "right_wrist", "right_arm"),
    # Left Leg
    ("left_hip", "left_knee", "left_leg"),
    ("left_knee", "left_ankle", "left_leg"),
    ("left_ankle", "left_heel", "left_leg"),
    ("left_ankle", "left_foot_index", "left_leg"),
    # Right Leg
    ("right_hip", "right_knee", "right_leg"),
    ("right_knee", "right_ankle", "right_leg"),
    ("right_ankle", "right_heel", "right_leg"),
    ("right_ankle", "right_foot_index", "right_leg"),
]

# ==============================================================================
# 5. VISUALIZATION & UI PALETTE (BGR Format for OpenCV)
# ==============================================================================

COLORS: Dict[str, Tuple[int, int, int]] = {
    # Bones by anatomical group
    "bone_head": (255, 220, 100),       # Cyan-blue
    "bone_torso": (0, 215, 255),        # Gold / Amber
    "bone_left_arm": (255, 140, 0),     # Deep Sky Blue
    "bone_right_arm": (100, 230, 255),  # Light Yellow
    "bone_left_leg": (255, 0, 180),     # Violet-Pink
    "bone_right_leg": (0, 255, 180),    # Spring Green
    "bone_default": (200, 200, 200),    # Light Gray
    # Keypoint circles
    "joint_valid": (0, 255, 120),       # Vibrant Green
    "joint_low_conf": (0, 165, 255),    # Orange
    "joint_invalid": (100, 100, 100),   # Dim Gray
    "joint_border": (255, 255, 255),    # Pure White
    # Status badges
    "status_valid": (46, 204, 113),     # Emerald Green
    "status_warning": (0, 140, 255),    # Amber
    "status_error": (50, 50, 230),      # Crimson
    "overlay_bg": (25, 25, 30),         # Dark Slate
    "text_white": (255, 255, 255),
    "text_gray": (180, 180, 180),
}

# Drawing dimensions
JOINT_RADIUS: int = 5
JOINT_THICKNESS: int = -1  # Filled
BONE_THICKNESS: int = 2

# ==============================================================================
# 6. SYSTEM STATUS CONSTANTS
# ==============================================================================

STATUS_VALID_FULL_BODY: str = "VALID_FULL_BODY"
STATUS_PARTIAL_BODY: str = "PARTIAL_BODY"
STATUS_NO_PERSON: str = "NO_PERSON_DETECTED"
STATUS_MODEL_ERROR: str = "MODEL_ERROR"
