"""
YogaCare AI - Phase 2: Angle Calculator Module
Provides robust vector geometry, 3-point joint angle calculations, orientation checks
(relative to horizontal/vertical planes), and temporal smoothing for posture analysis.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np

from pose_detector import Landmark, PoseDetectionResult


def calculate_3point_angle(
    a: Union[Tuple[float, float], np.ndarray],
    b: Union[Tuple[float, float], np.ndarray],
    c: Union[Tuple[float, float], np.ndarray],
) -> float:
    """
    Calculates the 2D planar angle (in degrees) at vertex B formed by segments BA and BC.

    Args:
        a: Coordinates of first end-point (x, y).
        b: Coordinates of the vertex joint (x, y) where the angle is measured.
        c: Coordinates of second end-point (x, y).

    Returns:
        Angle in degrees in the range [0.0, 180.0]. Returns 0.0 if any vector is degenerate.
    """
    v1 = np.array([a[0] - b[0], a[1] - b[1]], dtype=np.float64)
    v2 = np.array([c[0] - b[0], c[1] - b[1]], dtype=np.float64)

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    # Prevent division by zero if joints coincide
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0

    cosine = np.dot(v1, v2) / (norm1 * norm2)
    # Clamp cosine to valid [-1.0, 1.0] range to avoid floating point math errors
    cosine = np.clip(cosine, -1.0, 1.0)

    angle_rad = np.arccos(cosine)
    return float(np.degrees(angle_rad))


def calculate_angle_with_horizontal(
    p1: Union[Tuple[float, float], np.ndarray],
    p2: Union[Tuple[float, float], np.ndarray],
) -> float:
    """
    Calculates the absolute acute angle (in degrees) of vector p1 -> p2 relative to the horizontal axis.

    Useful for checking if arms are parallel to the ground (ideal angle close to 0.0°).

    Returns:
        Angle in degrees in range [0.0, 90.0].
    """
    dx = float(p2[0] - p1[0])
    dy = float(p2[1] - p1[1])

    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0.0

    # Angle relative to horizontal
    angle_rad = np.arctan2(abs(dy), abs(dx))
    return float(np.degrees(angle_rad))


def calculate_angle_with_vertical(
    p1: Union[Tuple[float, float], np.ndarray],
    p2: Union[Tuple[float, float], np.ndarray],
) -> float:
    """
    Calculates the absolute acute angle (in degrees) of vector p1 -> p2 relative to the vertical axis.

    Useful for checking spine / torso upright alignment (ideal angle close to 0.0°).

    Returns:
        Angle in degrees in range [0.0, 90.0].
    """
    dx = float(p2[0] - p1[0])
    dy = float(p2[1] - p1[1])

    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0.0

    # Angle relative to vertical
    angle_rad = np.arctan2(abs(dx), abs(dy))
    return float(np.degrees(angle_rad))


def calculate_midpoint(
    p1: Union[Tuple[float, float], np.ndarray],
    p2: Union[Tuple[float, float], np.ndarray],
) -> Tuple[float, float]:
    """Calculates the 2D midpoint between two points."""
    return (float(p1[0] + p2[0]) / 2.0, float(p1[1] + p2[1]) / 2.0)


def calculate_distance(
    p1: Union[Tuple[float, float], np.ndarray],
    p2: Union[Tuple[float, float], np.ndarray],
) -> float:
    """Calculates Euclidean distance between two points."""
    return float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))


def get_joint_angle_safe(
    result: PoseDetectionResult,
    joint_a: str,
    joint_b: str,
    joint_c: str,
    use_pixel: bool = True,
) -> Optional[float]:
    """
    Convenience wrapper that verifies all three joints are present and valid
    before computing the 3-point angle.

    Args:
        result: PoseDetectionResult from pose_detector.
        joint_a: Name of landmark A.
        joint_b: Name of vertex landmark B.
        joint_c: Name of landmark C.
        use_pixel: If True, uses pixel coordinates (default); if False, uses normalized coordinates.

    Returns:
        Angle in degrees [0.0, 180.0] if all joints are valid; otherwise None.
    """
    if not result.are_joints_valid(joint_a, joint_b, joint_c):
        return None

    lm_a = result.landmarks[joint_a]
    lm_b = result.landmarks[joint_b]
    lm_c = result.landmarks[joint_c]

    pt_a = lm_a.as_pixel_tuple() if use_pixel else lm_a.as_norm_tuple()
    pt_b = lm_b.as_pixel_tuple() if use_pixel else lm_b.as_norm_tuple()
    pt_c = lm_c.as_pixel_tuple() if use_pixel else lm_c.as_norm_tuple()

    return calculate_3point_angle(pt_a, pt_b, pt_c)


class AngleSmoother:
    """
    Exponential Moving Average (EMA) filter for joint angles to eliminate jitter
    caused by webcam frame noise while preserving responsive motion.
    """

    def __init__(self, alpha: float = 0.35) -> None:
        """
        Args:
            alpha: Smoothing factor in range (0.0, 1.0].
                   Higher = more responsive, Lower = smoother.
        """
        self.alpha = float(np.clip(alpha, 0.05, 1.0))
        self._smoothed_angles: Dict[str, float] = {}

    def update(self, key: str, raw_angle: Optional[float]) -> Optional[float]:
        """
        Updates the smoothed angle for the given joint key.

        Args:
            key: Unique identifier for the angle (e.g. "left_knee", "right_elbow").
            raw_angle: Measured angle in degrees, or None if joint is invalid.

        Returns:
            Smoothed angle in degrees, or None if raw_angle is None.
        """
        if raw_angle is None:
            # Reset on tracking loss to avoid carry-over
            self._smoothed_angles.pop(key, None)
            return None

        if key not in self._smoothed_angles:
            self._smoothed_angles[key] = raw_angle
        else:
            self._smoothed_angles[key] = (
                self.alpha * raw_angle + (1.0 - self.alpha) * self._smoothed_angles[key]
            )

        return self._smoothed_angles[key]

    def reset(self) -> None:
        """Clears all stored filter states."""
        self._smoothed_angles.clear()
