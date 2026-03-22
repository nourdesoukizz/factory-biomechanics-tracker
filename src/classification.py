"""Task classification module — rule-based activity classifier with confidence scoring."""

import logging
from collections import defaultdict
from enum import Enum
from typing import Dict, List, Optional, Tuple

from src.biomechanics import RULA_MEDIUM
from src.models import PersonFrame, TaskEvent

logger = logging.getLogger(__name__)


class TaskState(Enum):
    """State machine states for task detection."""
    IDLE = "idle"
    CANDIDATE = "candidate"
    ACTIVE = "active"


class TaskClassifier:
    """Rule-based classifier for factory floor tasks."""

    def __init__(self, confidence_threshold: float = 0.6,
                 min_task_duration_frames: int = 8,
                 task_cooldown_frames: int = 15,
                 pick_velocity_threshold: float = 30.0,
                 stationary_threshold: float = 20.0,
                 movement_velocity_threshold: float = 60.0) -> None:
        """Initialize the task classifier.

        Args:
            confidence_threshold: Minimum confidence to trigger a task.
            min_task_duration_frames: Minimum frames before a task event is logged.
            task_cooldown_frames: Frames before the same task can re-trigger.
            pick_velocity_threshold: Velocity threshold for pick_and_place.
            stationary_threshold: Centroid movement threshold for stationary.
            movement_velocity_threshold: Velocity threshold for move_rack.
        """
        self._conf_threshold = confidence_threshold
        self._min_duration = min_task_duration_frames
        self._cooldown = task_cooldown_frames
        self._pick_vel = pick_velocity_threshold
        self._stationary_thresh = stationary_threshold
        self._movement_vel = movement_velocity_threshold

        # Per-person state machine tracking
        # track_id -> {task_name -> {state, start_frame, consecutive_frames, last_end_frame}}
        self._person_states: Dict[int, Dict[str, Dict]] = defaultdict(
            lambda: defaultdict(lambda: {
                "state": TaskState.IDLE,
                "start_frame": 0,
                "consecutive_frames": 0,
                "last_end_frame": -9999,
                "confidence_sum": 0.0,
            })
        )

    def classify_frame(self, person: PersonFrame, joint_angles: Dict[str, float],
                       velocity: float, rula_score: float, rula_label: str,
                       fps: float) -> Tuple[str, float, List[TaskEvent]]:
        """Classify the task for a single person in a single frame.

        Args:
            person: PersonFrame data.
            joint_angles: Computed joint angles.
            velocity: Centroid velocity in px/sec.
            rula_score: RULA score for this frame.
            rula_label: RULA risk label.
            fps: Video FPS for duration computation.

        Returns:
            Tuple of (task_name, confidence, list of newly completed TaskEvents).
        """
        # Compute confidence for each task
        task_confidences = {
            "pick_and_place": self._pick_and_place_confidence(joint_angles, velocity),
            "lift_and_place": self._lift_and_place_confidence(joint_angles, velocity, rula_label),
            "move_rack": self._move_rack_confidence(joint_angles, velocity),
        }

        # Find best task above threshold
        best_task = "idle"
        best_conf = 0.0
        for task, conf in task_confidences.items():
            if conf > best_conf:
                best_conf = conf
                best_task = task

        if best_conf < self._conf_threshold:
            best_task = "idle"
            best_conf = 1.0 - max(task_confidences.values()) if task_confidences else 1.0

        # Update state machines and collect completed events
        completed_events = self._update_state_machines(
            person.track_id, person.frame_index, task_confidences, fps
        )

        return best_task, best_conf, completed_events

    def _pick_and_place_confidence(self, angles: Dict[str, float], velocity: float) -> float:
        """Compute confidence for pick_and_place task.

        Conditions:
        1. At least one wrist below shoulder (wrist_height < 0)
        2. Elbow angle 60-130 degrees
        3. Wrist velocity low
        4. Trunk flex < 30 degrees
        5. Person centroid stationary
        """
        conditions_met = 0
        total_conditions = 5

        # Condition 1: wrist below shoulder
        wh_left = angles.get("wrist_height_left", 0.0)
        wh_right = angles.get("wrist_height_right", 0.0)
        if wh_left < 0 or wh_right < 0:
            conditions_met += 1

        # Condition 2: elbow angle 60-130
        el_left = angles.get("elbow_left", 0.0)
        el_right = angles.get("elbow_right", 0.0)
        if (60 <= el_left <= 130) or (60 <= el_right <= 130):
            conditions_met += 1

        # Condition 3: low velocity (precise movement)
        if velocity < self._pick_vel:
            conditions_met += 1

        # Condition 4: trunk flex < 30
        if angles.get("trunk_flex", 0.0) < 30:
            conditions_met += 1

        # Condition 5: stationary centroid
        if velocity < self._stationary_thresh:
            conditions_met += 1

        return conditions_met / total_conditions

    def _lift_and_place_confidence(self, angles: Dict[str, float], velocity: float,
                                   rula_label: str) -> float:
        """Compute confidence for lift_and_place task.

        Conditions:
        1. Both wrists above waist (wrist_height > -0.5) or rising
        2. Shoulder angle > 45
        3. Trunk flex > threshold
        4. Centroid velocity low
        5. RULA score >= MEDIUM
        """
        conditions_met = 0
        total_conditions = 5

        # Condition 1: wrists above waist level
        wh_left = angles.get("wrist_height_left", 0.0)
        wh_right = angles.get("wrist_height_right", 0.0)
        if wh_left > -0.5 and wh_right > -0.5:
            conditions_met += 1

        # Condition 2: shoulder angle > 45
        sh_left = angles.get("shoulder_left", 0.0)
        sh_right = angles.get("shoulder_right", 0.0)
        if sh_left > 45 or sh_right > 45:
            conditions_met += 1

        # Condition 3: trunk flexion (forward lean)
        if angles.get("trunk_flex", 0.0) > 20:
            conditions_met += 1

        # Condition 4: low centroid velocity
        if velocity < self._stationary_thresh:
            conditions_met += 1

        # Condition 5: RULA >= MEDIUM
        if rula_label in (RULA_MEDIUM, "HIGH", "VERY_HIGH"):
            conditions_met += 1

        return conditions_met / total_conditions

    def _move_rack_confidence(self, angles: Dict[str, float], velocity: float) -> float:
        """Compute confidence for move_rack task.

        Conditions:
        1. Both arms extended (elbow angle > 140)
        2. Centroid velocity high (walking)
        3. Trunk flex < 15 (upright)
        4. Wrists at hip height
        """
        conditions_met = 0
        total_conditions = 4

        # Condition 1: arms extended
        el_left = angles.get("elbow_left", 0.0)
        el_right = angles.get("elbow_right", 0.0)
        if el_left > 140 or el_right > 140:
            conditions_met += 1

        # Condition 2: high centroid velocity
        if velocity > self._movement_vel:
            conditions_met += 1

        # Condition 3: upright posture
        if angles.get("trunk_flex", 0.0) < 15:
            conditions_met += 1

        # Condition 4: wrists near hip height
        wh_left = angles.get("wrist_height_left", 0.0)
        wh_right = angles.get("wrist_height_right", 0.0)
        if abs(wh_left) < 0.3 and abs(wh_right) < 0.3:
            conditions_met += 1

        return conditions_met / total_conditions

    def _update_state_machines(self, track_id: int, frame_index: int,
                               task_confidences: Dict[str, float],
                               fps: float) -> List[TaskEvent]:
        """Update per-person state machines and return completed events.

        Args:
            track_id: Person track ID.
            frame_index: Current frame index.
            task_confidences: Confidence scores per task.
            fps: Video FPS.

        Returns:
            List of newly completed TaskEvents.
        """
        completed: List[TaskEvent] = []
        states = self._person_states[track_id]

        for task_name, conf in task_confidences.items():
            state_info = states[task_name]

            # Check cooldown
            if frame_index - state_info["last_end_frame"] < self._cooldown:
                continue

            if conf >= self._conf_threshold:
                if state_info["state"] == TaskState.IDLE:
                    # Transition to CANDIDATE
                    state_info["state"] = TaskState.CANDIDATE
                    state_info["start_frame"] = frame_index
                    state_info["consecutive_frames"] = 1
                    state_info["confidence_sum"] = conf
                elif state_info["state"] == TaskState.CANDIDATE:
                    state_info["consecutive_frames"] += 1
                    state_info["confidence_sum"] += conf
                    if state_info["consecutive_frames"] >= self._min_duration:
                        state_info["state"] = TaskState.ACTIVE
                elif state_info["state"] == TaskState.ACTIVE:
                    state_info["consecutive_frames"] += 1
                    state_info["confidence_sum"] += conf
            else:
                # Confidence dropped — close active task or reset candidate
                if state_info["state"] == TaskState.ACTIVE:
                    avg_conf = state_info["confidence_sum"] / max(state_info["consecutive_frames"], 1)
                    duration = state_info["consecutive_frames"] / fps if fps > 0 else 0.0
                    completed.append(TaskEvent(
                        track_id=track_id,
                        task=task_name,
                        start_frame=state_info["start_frame"],
                        end_frame=frame_index,
                        confidence=round(avg_conf, 3),
                        duration_sec=round(duration, 2),
                    ))
                    state_info["last_end_frame"] = frame_index
                    logger.debug("Task completed: %s for person %d (frames %d-%d, conf=%.2f)",
                                 task_name, track_id, state_info["start_frame"], frame_index, avg_conf)

                state_info["state"] = TaskState.IDLE
                state_info["consecutive_frames"] = 0
                state_info["confidence_sum"] = 0.0

        return completed

    def finalize(self, track_id: int, final_frame: int, fps: float) -> List[TaskEvent]:
        """Close any open task events at end of video.

        Args:
            track_id: Person track ID.
            final_frame: Last frame index.
            fps: Video FPS.

        Returns:
            List of any remaining active TaskEvents.
        """
        completed: List[TaskEvent] = []
        if track_id not in self._person_states:
            return completed

        for task_name, state_info in self._person_states[track_id].items():
            if state_info["state"] == TaskState.ACTIVE:
                avg_conf = state_info["confidence_sum"] / max(state_info["consecutive_frames"], 1)
                duration = state_info["consecutive_frames"] / fps if fps > 0 else 0.0
                completed.append(TaskEvent(
                    track_id=track_id,
                    task=task_name,
                    start_frame=state_info["start_frame"],
                    end_frame=final_frame,
                    confidence=round(avg_conf, 3),
                    duration_sec=round(duration, 2),
                ))

        return completed
