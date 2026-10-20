"""
YogaCare AI - Guided Yoga Routine Manager
Manages sequential flow, side tracking (Left vs Right), Cat-Cow repetition counting,
countdown timers, auto-advance logic, and performance metrics for the 22-posture routine.
"""

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RoutinePose:
    """Configuration for a single yoga posture in the guided routine."""
    pose_id: str
    sequence_number: int
    name: str
    sanskrit_name: str
    phase: str
    phase_number: int
    image_filename: str
    local_image_path: str
    suggested_duration_seconds: int
    requires_sides: bool = False
    is_repetition: bool = False
    has_ai_evaluation: bool = False
    applicable_evaluator_key: Optional[str] = None
    instructions: List[str] = field(default_factory=list)
    alignment_tips: List[str] = field(default_factory=list)
    safety_reminder: str = "Breathe naturally and listen to your body."


@dataclass
class PoseSessionState:
    """Live state tracking for a posture during an active routine session."""
    pose_id: str
    sequence_number: int
    name: str
    suggested_duration: int
    current_side: Optional[str] = None  # "Left", "Right", or None
    sides_completed: Dict[str, bool] = field(default_factory=lambda: {"Left": False, "Right": False})
    repetition_count: int = 0
    elapsed_seconds: float = 0.0
    correct_seconds: float = 0.0
    best_hold_seconds: float = 0.0
    is_completed: bool = False
    is_skipped: bool = False


class RoutineManager:
    """
    Coordinates the 22-posture guided yoga routine.
    Maintains current index, bilateral side switches, reps, pauses, and summary statistics.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        if config_path:
            self.config_path = Path(config_path)
        else:
            base_dir = Path(__file__).resolve().parent
            self.config_path = base_dir / "data" / "yoga_routine.json"

        self.poses: List[RoutinePose] = self._load_poses()
        self.current_index: int = 0
        self.is_active: bool = False
        self.is_paused: bool = False
        self.auto_advance: bool = True
        self.start_timestamp: Optional[str] = None
        self.states: Dict[int, PoseSessionState] = {}
        self._init_states()

    def _load_poses(self) -> List[RoutinePose]:
        """Loads all postures from yoga_routine.json."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Routine configuration not found at: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        poses: List[RoutinePose] = []
        for item in data:
            poses.append(
                RoutinePose(
                    pose_id=item["pose_id"],
                    sequence_number=item["sequence_number"],
                    name=item["name"],
                    sanskrit_name=item["sanskrit_name"],
                    phase=item["phase"],
                    phase_number=item["phase_number"],
                    image_filename=item["image_filename"],
                    local_image_path=item["local_image_path"],
                    suggested_duration_seconds=item["suggested_duration_seconds"],
                    requires_sides=item.get("requires_sides", False),
                    is_repetition=item.get("is_repetition", False),
                    has_ai_evaluation=item.get("has_ai_evaluation", False),
                    applicable_evaluator_key=item.get("applicable_evaluator_key"),
                    instructions=item.get("instructions", []),
                    alignment_tips=item.get("alignment_tips", []),
                    safety_reminder=item.get("safety_reminder", "Move mindfully and breathe smoothly."),
                )
            )
        # Ensure ordered by sequence_number
        poses.sort(key=lambda p: p.sequence_number)
        return poses

    def _init_states(self) -> None:
        """Initializes empty state tracking for each posture."""
        self.states = {}
        for idx, p in enumerate(self.poses):
            init_side = "Left" if p.requires_sides else None
            self.states[idx] = PoseSessionState(
                pose_id=p.pose_id,
                sequence_number=p.sequence_number,
                name=p.name,
                suggested_duration=p.suggested_duration_seconds,
                current_side=init_side,
                sides_completed={"Left": False, "Right": False} if p.requires_sides else {},
            )

    def start_routine(self) -> None:
        """Begins or resumes the guided routine."""
        self.is_active = True
        self.is_paused = False
        if not self.start_timestamp:
            from datetime import datetime
            self.start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def toggle_pause(self) -> bool:
        """Toggles between paused and active states."""
        self.is_paused = not self.is_paused
        return self.is_paused

    def get_current_pose(self) -> RoutinePose:
        """Returns the currently active RoutinePose."""
        if not self.poses:
            raise ValueError("No poses loaded in routine.")
        idx = max(0, min(self.current_index, len(self.poses) - 1))
        return self.poses[idx]

    def get_current_state(self) -> PoseSessionState:
        """Returns the active state tracking object for current pose."""
        if self.current_index not in self.states:
            self._init_states()
        return self.states[self.current_index]

    def update_timing(self, delta_seconds: float, is_correct: bool = False) -> Tuple[bool, float]:
        """
        Increments elapsed time and correct time if pose criteria met.
        Returns:
            (has_reached_target, remaining_seconds)
        """
        if not self.is_active or self.is_paused:
            rem = max(0.0, self.get_current_pose().suggested_duration_seconds - self.get_current_state().elapsed_seconds)
            return False, rem

        state = self.get_current_state()
        state.elapsed_seconds += delta_seconds

        if is_correct:
            state.correct_seconds += delta_seconds
            if state.correct_seconds > state.best_hold_seconds:
                state.best_hold_seconds = state.correct_seconds

        target = float(self.get_current_pose().suggested_duration_seconds)
        remaining = max(0.0, target - state.elapsed_seconds)
        reached = (state.elapsed_seconds >= target)

        if reached:
            if self.get_current_pose().requires_sides:
                if state.current_side:
                    state.sides_completed[state.current_side] = True
                    # Check if all required sides completed
                    if all(state.sides_completed.values()):
                        state.is_completed = True
            else:
                state.is_completed = True

        return reached, remaining

    def switch_side(self) -> Optional[str]:
        """
        Toggles between 'Left' and 'Right' for bilateral postures.
        Marks the previous side as completed.
        """
        pose = self.get_current_pose()
        if not pose.requires_sides:
            return None

        state = self.get_current_state()
        if state.current_side == "Left":
            state.sides_completed["Left"] = True
            state.current_side = "Right"
        elif state.current_side == "Right":
            state.sides_completed["Right"] = True
            state.current_side = "Left"
        else:
            state.current_side = "Left"

        # Reset timer counter for new side
        state.elapsed_seconds = 0.0
        return state.current_side

    def record_rep(self) -> int:
        """Increments repetition count for dynamic/alternating poses (e.g. Cat-Cow)."""
        state = self.get_current_state()
        state.repetition_count += 1
        if state.repetition_count >= 5:
            state.is_completed = True
        return state.repetition_count

    def advance(self, force: bool = False) -> bool:
        """
        Advances to the next posture.
        If current pose requires sides and user hasn't switched sides yet,
        automatically switches to the other side unless force=True.
        Returns True if moved to next pose or side, False if already at end.
        """
        pose = self.get_current_pose()
        state = self.get_current_state()

        if not force and pose.requires_sides:
            # Check if one side is unfinished
            if state.current_side == "Left" and not state.sides_completed.get("Right", False):
                self.switch_side()
                return True

        state.is_completed = True

        if self.current_index < len(self.poses) - 1:
            self.current_index += 1
            return True
        else:
            # Reached end of routine
            self.is_active = False
            return False

    def previous(self) -> bool:
        """Moves to the previous posture in routine."""
        if self.current_index > 0:
            self.current_index -= 1
            return True
        return False

    def jump_to_pose(self, sequence_number: int) -> bool:
        """Jumps directly to a pose by its 1-based sequence number."""
        target_idx = sequence_number - 1
        if 0 <= target_idx < len(self.poses):
            self.current_index = target_idx
            return True
        return False

    def get_progress(self) -> Dict[str, Any]:
        """Returns overall routine progress information."""
        total = len(self.poses)
        completed_count = sum(1 for s in self.states.values() if s.is_completed)
        percent = int(round((completed_count / total) * 100)) if total > 0 else 0
        curr_pose = self.get_current_pose()

        return {
            "current_index": self.current_index,
            "current_pose_number": self.current_index + 1,
            "total_poses": total,
            "completed_count": completed_count,
            "percent_completed": percent,
            "current_pose_name": curr_pose.name,
            "current_sanskrit_name": curr_pose.sanskrit_name,
            "current_phase": curr_pose.phase,
            "is_finished": self.current_index >= total - 1 and self.get_current_state().is_completed,
        }

    def get_summary(self) -> Dict[str, Any]:
        """
        Compiles the complete workout routine performance summary.
        Does NOT fabricate scores for un-evaluated postures.
        """
        from datetime import datetime
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        total_practice_sec = sum(s.elapsed_seconds for s in self.states.values())
        poses_attempted = sum(1 for s in self.states.values() if s.elapsed_seconds >= 2.0 or s.repetition_count > 0)
        poses_completed = sum(1 for s in self.states.values() if s.is_completed)

        # Only sum correct pose time for poses with AI evaluation
        evaluated_states = [
            s for idx, s in self.states.items()
            if self.poses[idx].has_ai_evaluation and s.elapsed_seconds >= 2.0
        ]
        if evaluated_states:
            total_correct_sec = sum(s.correct_seconds for s in evaluated_states)
            best_overall_hold = max((s.best_hold_seconds for s in evaluated_states), default=0.0)
        else:
            total_correct_sec = None
            best_overall_hold = 0.0

        details = []
        for idx, p in enumerate(self.poses):
            st = self.states[idx]
            details.append({
                "sequence": p.sequence_number,
                "name": p.name,
                "sanskrit": p.sanskrit_name,
                "phase": p.phase,
                "duration_seconds": round(st.elapsed_seconds, 1),
                "correct_seconds": round(st.correct_seconds, 1) if p.has_ai_evaluation else None,
                "completed": st.is_completed,
                "reps": st.repetition_count if p.is_repetition else None,
                "has_ai_evaluation": p.has_ai_evaluation,
            })

        return {
            "routine_name": "22-Pose Guided Yoga Routine",
            "start_time": self.start_timestamp or end_time,
            "end_time": end_time,
            "total_duration_seconds": round(total_practice_sec, 1),
            "poses_attempted": poses_attempted,
            "poses_completed": poses_completed,
            "completed_all": (poses_completed >= len(self.poses)),
            "correct_pose_seconds": round(total_correct_sec, 1) if total_correct_sec is not None else None,
            "best_hold_seconds": round(best_overall_hold, 1),
            "details": details,
        }

    def reset_routine(self) -> None:
        """Resets routine back to Pose 1 with fresh timers."""
        self.current_index = 0
        self.is_active = False
        self.is_paused = False
        self.start_timestamp = None
        self._init_states()
