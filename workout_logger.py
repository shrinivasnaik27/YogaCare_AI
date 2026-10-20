"""
YogaCare AI - Phase 4: Workout Session Logger
Tracks practice history, session durations, posture stability scores,
and common alignment corrections for analytics and progress tracking.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PoseRecord:
    timestamp: str
    pose_name: str
    stance: str
    duration_seconds: float
    average_score: float
    best_hold_seconds: float
    common_cues: List[str] = field(default_factory=list)
    correct_pose_seconds: Optional[float] = None


class WorkoutLogger:
    """Manages persistent workout logs and performance statistics."""

    def __init__(self, storage_path: Optional[str] = None) -> None:
        if storage_path:
            self.file_path = Path(storage_path)
        else:
            base_dir = Path(__file__).resolve().parent
            self.file_path = base_dir / "data" / "workout_history.json"

        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._history: List[Dict[str, Any]] = self._load_history()

    def _load_history(self) -> List[Dict[str, Any]]:
        """Loads existing session logs from JSON file."""
        if not self.file_path.exists():
            return []
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _save_history(self) -> None:
        """Saves current session logs to JSON file."""
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self._history, f, indent=2)
        except Exception as e:
            print(f"[Warning] Failed to save workout history: {e}")

    def log_session(
        self,
        pose_name: str,
        duration_seconds: float,
        target_duration: Optional[int] = None,
        correct_pose_seconds: Optional[float] = None,
        max_continuous_hold: Optional[float] = None,
        average_score: Optional[float] = 75.0,
        target_achieved: bool = False,
        cues_provided: Optional[List[str]] = None,
        stance: str = "Standard",
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        """
        Logs a general pose practice session (from 40-practice library, single drill, or guided pose).
        Accepts flexible parameters, prevents trivial flickers (< 2.0s), and persists directly to JSON.
        """
        if duration_seconds < 2.0:
            # Ignore trivial flickers < 2 seconds
            return None

        best_hold = max_continuous_hold if max_continuous_hold is not None else (
            correct_pose_seconds if correct_pose_seconds is not None else 0.0
        )
        avg_score = average_score if average_score is not None else 75.0

        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "pose_name": pose_name,
            "stance": stance,
            "duration_seconds": round(float(duration_seconds), 1),
            "average_score": round(float(avg_score), 1),
            "best_hold_seconds": round(float(best_hold), 1),
            "common_cues": cues_provided or [],
            "correct_pose_seconds": round(float(correct_pose_seconds), 1) if correct_pose_seconds is not None else None,
            "target_duration": target_duration,
            "target_achieved": target_achieved,
        }

        self._history.append(record)
        self._save_history()
        return record

    def log_pose_session(
        self,
        pose_name: str,
        stance: str,
        duration_seconds: float,
        average_score: float,
        best_hold_seconds: float,
        common_cues: Optional[List[str]] = None,
        correct_pose_seconds: Optional[float] = None,
    ) -> None:
        """Appends a new practice record to history."""
        self.log_session(
            pose_name=pose_name,
            duration_seconds=duration_seconds,
            stance=stance,
            average_score=average_score,
            max_continuous_hold=best_hold_seconds,
            cues_provided=common_cues,
            correct_pose_seconds=correct_pose_seconds,
        )

    def log_routine_session(
        self,
        routine_summary: Dict[str, Any],
    ) -> None:
        """
        Logs a completed or ended guided routine session into practice history.
        Includes session duration, poses completed, correct pose hold duration,
        and whether it was completed or ended early.
        """
        total_duration = float(routine_summary.get("total_duration_seconds", 0.0))
        if total_duration < 2.0 and routine_summary.get("poses_attempted", 0) == 0:
            return

        poses_completed = routine_summary.get("poses_completed", 0)
        poses_attempted = routine_summary.get("poses_attempted", 0)
        completed_all = routine_summary.get("completed_all", False)
        status_label = "Completed" if completed_all else "Ended Early"
        routine_name = routine_summary.get("routine_name", "22-Pose Guided Yoga Routine")

        pose_label = f"{routine_name} ({poses_completed}/{poses_attempted} poses, {status_label})"
        correct_sec = routine_summary.get("correct_pose_seconds")
        best_hold = float(routine_summary.get("best_hold_seconds", 0.0))

        record = {
            "timestamp": routine_summary.get("end_time") or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "pose_name": pose_label,
            "stance": "Routine Flow",
            "duration_seconds": round(total_duration, 1),
            "average_score": 100.0 if completed_all else round((poses_completed / max(1, poses_attempted)) * 100.0, 1),
            "best_hold_seconds": round(best_hold, 1),
            "common_cues": [f"Completed {poses_completed} of {poses_attempted} poses", f"Status: {status_label}"],
            "correct_pose_seconds": round(float(correct_sec), 1) if correct_sec is not None else None,
            "is_routine": True,
            "poses_completed": poses_completed,
            "poses_attempted": poses_attempted,
            "completed_all": completed_all,
            "routine_details": routine_summary.get("details", []),
        }

        self._history.append(record)
        self._save_history()

    def get_history(self) -> List[Dict[str, Any]]:
        """Returns all historical practice sessions, freshly reloaded from disk."""
        self._history = self._load_history()
        return self._history

    def get_summary_stats(self) -> Dict[str, Any]:
        """Calculates cumulative workout statistics, freshly reloaded from disk."""
        self._history = self._load_history()
        if not self._history:
            return {
                "total_sessions": 0,
                "total_practice_time_seconds": 0.0,
                "total_correct_pose_seconds": None,
                "average_score": 0.0,
                "best_overall_hold": 0.0,
                "favorite_pose": "None",
            }

        total_sessions = len(self._history)
        total_time = sum(float(r.get("duration_seconds", 0.0)) for r in self._history)
        avg_score = sum(float(r.get("average_score", 0.0)) for r in self._history) / total_sessions
        best_hold = max((float(r.get("best_hold_seconds", 0.0)) for r in self._history), default=0.0)

        # Sum recorded correct pose time where available
        correct_vals = [
            float(r["correct_pose_seconds"])
            for r in self._history
            if r.get("correct_pose_seconds") is not None
        ]
        total_correct = sum(correct_vals) if correct_vals else None

        # Count favorite pose
        pose_counts: Dict[str, int] = {}
        for r in self._history:
            name = r.get("pose_name", "Unknown").split(" (")[0]
            pose_counts[name] = pose_counts.get(name, 0) + 1

        fav_pose = max(pose_counts.items(), key=lambda x: x[1])[0] if pose_counts else "None"

        return {
            "total_sessions": total_sessions,
            "total_practice_time_seconds": round(total_time, 1),
            "total_correct_pose_seconds": round(total_correct, 1) if total_correct is not None else None,
            "average_score": round(avg_score, 1),
            "best_overall_hold": round(best_hold, 1),
            "favorite_pose": fav_pose,
        }

    def clear_history(self) -> None:
        """Resets workout logs."""
        self._history = []
        self._save_history()


def format_seconds_readable(seconds: Optional[float]) -> str:
    """
    Converts raw seconds into readable 'X min Y sec' format.
    Returns 'Not recorded' if seconds is None.
    """
    if seconds is None:
        return "Not recorded"
    sec_int = int(round(seconds))
    if sec_int < 0:
        return "0 sec"
    mins, secs = divmod(sec_int, 60)
    if mins > 0 and secs > 0:
        return f"{mins} min {secs} sec"
    elif mins > 0:
        return f"{mins} min"
    else:
        return f"{secs} sec"


def format_datetime_readable(ts_str: str) -> str:
    """
    Converts 'YYYY-MM-DD HH:MM:SS' into user-friendly 'Oct 19, 2026 12:08 PM' format.
    """
    if not ts_str:
        return "N/A"
    try:
        dt = datetime.strptime(ts_str.strip(), "%Y-%m-%d %H:%M:%S")
        hour = dt.strftime("%I").lstrip("0") or "12"
        return f"{dt.strftime('%b %d, %Y')} {hour}:{dt.strftime('%M %p')}"
    except Exception:
        return ts_str


def format_pose_display_name(raw_name: str) -> str:
    """
    Maps internal or abbreviated pose names to clear user-facing titles.
    """
    clean = raw_name.strip()
    if "routine" in clean.lower():
        return clean
    mapping = {
        "warrior_ii": "Warrior II (Virabhadrasana II)",
        "tree_pose": "Tree Pose (Vrikshasana)",
        "downward_dog": "Downward-Facing Dog (Adho Mukha Svanasana)",
        "plank_pose": "Plank Pose (Phalakasana)",
        "cobra_pose": "Cobra Pose (Bhujangasana)",
        "mountain_pose": "Mountain Pose (Tadasana)",
    }
    if clean in mapping:
        return mapping[clean]
    for key, full in mapping.items():
        if key in clean.lower():
            return full
    return clean
