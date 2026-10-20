"""
YogaCare AI - Phase 2: Posture Analyzer Module
Orchestrates yoga posture evaluation, real-time angle annotation, posture scoring (0-100%),
hold duration timing, and prioritized corrective feedback rendering.
"""

import time
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

import config
from poses.base_pose import AlignmentStatus, BasePoseEvaluator, JointFeedback, PoseEvaluationResult
from poses.cobra_pose import CobraPoseEvaluator
from poses.downward_dog import DownwardDogPoseEvaluator
from poses.mountain_pose import MountainPoseEvaluator
from poses.plank_pose import PlankPoseEvaluator
from poses.tree_pose import TreePoseEvaluator
from poses.warrior_ii import WarriorIIPoseEvaluator
from pose_detector import Landmark, PoseDetectionResult, YogaPoseDetector


class PostureAnalyzer:
    """
    Main Posture Analysis Engine for YogaCare AI.

    Features:
    - Modular posture evaluation for 6 classical yoga postures:
      1. Warrior II (Virabhadrasana II)
      2. Tree Pose (Vrikshasana)
      3. Downward-Facing Dog (Adho Mukha Svanasana)
      4. Plank Pose (Phalakasana)
      5. Cobra Pose (Bhujangasana)
      6. Mountain Pose (Tadasana)
    - Multi-pose auto-detection mode or manual pose selection.
    - Posture hold duration tracking (seconds held in correct alignment).
    - Rich visual HUD with real-time angle badges, score meters, and corrective cues.
    """

    SUPPORTED_POSES: Dict[str, str] = {
        "warrior_ii": "Warrior II (Virabhadrasana II)",
        "tree_pose": "Tree Pose (Vrikshasana)",
        "downward_dog": "Downward-Facing Dog (Adho Mukha Svanasana)",
        "plank_pose": "Plank Pose (Phalakasana)",
        "cobra_pose": "Cobra Pose (Bhujangasana)",
        "mountain_pose": "Mountain Pose (Tadasana)",
    }

    def __init__(
        self,
        default_pose: str = "warrior_ii",
        hold_score_threshold: float = 75.0,
    ) -> None:
        """
        Args:
            default_pose: Initial active pose ('warrior_ii', 'tree_pose', 'downward_dog', 'plank_pose', 'cobra_pose', 'mountain_pose', or 'auto').
            hold_score_threshold: Minimum posture score required to increment hold timer.
        """
        self.evaluators: Dict[str, BasePoseEvaluator] = {
            "warrior_ii": WarriorIIPoseEvaluator(),
            "tree_pose": TreePoseEvaluator(),
            "downward_dog": DownwardDogPoseEvaluator(),
            "plank_pose": PlankPoseEvaluator(),
            "cobra_pose": CobraPoseEvaluator(),
            "mountain_pose": MountainPoseEvaluator(),
        }
        self.active_pose = default_pose.lower()
        self.hold_threshold = hold_score_threshold

        # Posture hold tracking
        self._hold_start_time: Optional[float] = None
        self._current_hold_duration: float = 0.0
        self._best_hold_duration: float = 0.0

    def set_active_pose(self, pose_name: str) -> None:
        """Sets the active posture evaluator."""
        key = pose_name.lower()
        if key != "auto" and key not in self.evaluators:
            raise ValueError(
                f"Unknown pose '{pose_name}'. Supported poses: {list(self.evaluators.keys())} or 'auto'"
            )
        self.active_pose = key
        self.reset_hold_timer()

    def reset_hold_timer(self) -> None:
        """Resets the continuous posture hold timer."""
        self._hold_start_time = None
        self._current_hold_duration = 0.0

    # --------------------------------------------------------------------------
    # EVALUATION PIPELINE
    # --------------------------------------------------------------------------

    def evaluate_result(
        self,
        detection_result: PoseDetectionResult,
        pose_name: Optional[str] = None,
    ) -> PoseEvaluationResult:
        """
        Evaluates posture on a pre-computed PoseDetectionResult from Phase 1.

        Args:
            detection_result: Landmark detection outcome from YogaPoseDetector.
            pose_name: Specific pose to evaluate, or None to use active_pose.

        Returns:
            PoseEvaluationResult containing score, joint feedback, and cues.
        """
        target_pose = (pose_name or self.active_pose).lower()

        if target_pose == "auto":
            # Auto-detect: evaluate all registered postures and pick the highest score
            eval_results = [ev.evaluate(detection_result) for ev in self.evaluators.values()]
            eval_res = max(eval_results, key=lambda r: r.overall_score)
        elif target_pose in self.evaluators:
            eval_res = self.evaluators[target_pose].evaluate(detection_result)
        else:
            raise ValueError(f"Unsupported pose: {target_pose}")

        # Update hold timer
        self._update_hold_timer(eval_res.overall_score)

        return eval_res

    def _update_hold_timer(self, current_score: float) -> None:
        """Tracks continuous seconds spent with score >= threshold."""
        now = time.time()
        if current_score >= self.hold_threshold:
            if self._hold_start_time is None:
                self._hold_start_time = now
            self._current_hold_duration = now - self._hold_start_time
            if self._current_hold_duration > self._best_hold_duration:
                self._best_hold_duration = self._current_hold_duration
        else:
            # Drop below threshold resets continuous hold
            self._hold_start_time = None
            self._current_hold_duration = 0.0

    @property
    def current_hold_duration(self) -> float:
        """Seconds currently held in good posture."""
        return self._current_hold_duration

    @property
    def best_hold_duration(self) -> float:
        """Best continuous hold record in seconds."""
        return self._best_hold_duration

    # --------------------------------------------------------------------------
    # VISUAL FEEDBACK RENDERING
    # --------------------------------------------------------------------------

    def draw_posture_feedback(
        self,
        frame: np.ndarray,
        detection: PoseDetectionResult,
        evaluation: PoseEvaluationResult,
    ) -> np.ndarray:
        """
        Renders Phase 2 posture feedback overlays:
        - Dynamic joint angle tags on body limbs.
        - Posture accuracy gauge and hold timer card.
        - Actionable coaching cues.
        """
        h, w = frame.shape[:2]

        # 1. Draw Joint Angle Badges
        self._draw_joint_angle_badges(frame, detection, evaluation)

        # 2. Draw Posture Score & Hold Timer Card (Bottom Left)
        self._draw_posture_card(frame, evaluation, w, h)

        return frame

    def _draw_joint_angle_badges(
        self,
        frame: np.ndarray,
        detection: PoseDetectionResult,
        evaluation: PoseEvaluationResult,
    ) -> None:
        """Draws small floating angle badges next to evaluated joints."""
        # Mapping joint feedbacks to target landmark position
        joint_landmark_map = {
            "front_knee": ["left_knee", "right_knee"],
            "back_knee": ["left_knee", "right_knee"],
            "standing_knee": ["left_knee", "right_knee"],
            "lifted_knee": ["left_knee", "right_knee"],
            "front_arm": ["left_elbow", "right_elbow"],
            "back_arm": ["left_elbow", "right_elbow"],
            "torso_spine": ["left_shoulder", "right_shoulder"],
            "torso_alignment": ["left_shoulder", "right_shoulder"],
        }

        for j_key, feedback in evaluation.joint_feedbacks.items():
            if feedback.measured_angle is None:
                continue

            # Pick matching landmark coordinate
            candidate_names = joint_landmark_map.get(j_key, [])
            target_pt: Optional[Tuple[int, int]] = None

            for name in candidate_names:
                lm = detection.landmarks.get(name)
                if lm and lm.is_valid:
                    target_pt = lm.as_pixel_tuple()
                    break

            if target_pt is None:
                continue

            # Color by alignment status
            if feedback.status == AlignmentStatus.CORRECT:
                badge_bg = (46, 204, 113)   # Green
                text_col = (255, 255, 255)
            elif feedback.status == AlignmentStatus.WARNING:
                badge_bg = (0, 165, 255)    # Orange
                text_col = (255, 255, 255)
            else:
                badge_bg = (50, 50, 230)    # Red
                text_col = (255, 255, 255)

            # Draw small pill badge
            bx, by = target_pt[0] + 12, target_pt[1] - 8
            label = f"{feedback.measured_angle:.0f}°"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

            cv2.rectangle(frame, (bx - 4, by - th - 4), (bx + tw + 4, by + 4), badge_bg, -1)
            cv2.rectangle(frame, (bx - 4, by - th - 4), (bx + tw + 4, by + 4), (255, 255, 255), 1)
            cv2.putText(
                frame,
                label,
                (bx, by),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                text_col,
                1,
                lineType=cv2.LINE_AA,
            )

    def _draw_posture_card(
        self,
        frame: np.ndarray,
        evaluation: PoseEvaluationResult,
        w: int,
        h: int,
    ) -> None:
        """Renders bottom posture coaching card with score meter, hold timer, and cues."""
        card_w = min(420, w - 20)
        card_h = 135
        card_x = 15
        card_y = h - card_h - 15

        # Card background overlay
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (card_x, card_y),
            (card_x + card_w, card_y + card_h),
            (25, 25, 30),
            -1,
        )
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(
            frame,
            (card_x, card_y),
            (card_x + card_w, card_y + card_h),
            (80, 80, 90),
            1,
        )

        # Pose title & Stance
        title = f"{evaluation.pose_name.split(' (')[0]}"
        side_tag = f"({evaluation.detected_side})" if evaluation.detected_side != "unknown" else ""
        cv2.putText(
            frame,
            f"{title} {side_tag}",
            (card_x + 12, card_y + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )

        # Posture Score Pill
        score = evaluation.overall_score
        if score >= 75.0:
            meter_col = (46, 204, 113)  # Green
            score_label = f"Score: {score:.0f}% (EXCELLENT)"
        elif score >= 50.0:
            meter_col = (0, 165, 255)   # Orange
            score_label = f"Score: {score:.0f}% (ADJUST)"
        else:
            meter_col = (50, 50, 230)   # Red
            score_label = f"Score: {score:.0f}% (POOR)"

        # Score bar
        bar_x = card_x + 12
        bar_y = card_y + 36
        bar_w = card_w - 24
        bar_h = 10
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 55), -1)
        fill_w = int(bar_w * (score / 100.0))
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), meter_col, -1)

        # Hold Duration Timer
        hold_txt = f"Hold: {self._current_hold_duration:4.1f}s (Best: {self._best_hold_duration:4.1f}s)"
        cv2.putText(
            frame,
            f"{score_label}  |  {hold_txt}",
            (card_x + 12, card_y + 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (200, 200, 200),
            1,
            lineType=cv2.LINE_AA,
        )

        # Actionable Coaching Cue (Top cue)
        cue_msg = evaluation.actionable_cues[0] if evaluation.actionable_cues else "Maintain posture stability."
        cv2.putText(
            frame,
            f"> {cue_msg}",
            (card_x + 12, card_y + 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            meter_col if score >= 75.0 else (0, 200, 255),
            1,
            lineType=cv2.LINE_AA,
        )

        # Secondary cue if present
        if len(evaluation.actionable_cues) > 1:
            sec_cue = evaluation.actionable_cues[1]
            cv2.putText(
                frame,
                f"> {sec_cue}",
                (card_x + 12, card_y + 116),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (170, 170, 170),
                1,
                lineType=cv2.LINE_AA,
            )
