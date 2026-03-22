"""Shared data contracts for the factory biomechanics tracker pipeline."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class PersonFrame:
    """Per-person detection data for a single frame."""

    track_id: int
    bbox: List[float]  # [x1, y1, x2, y2]
    keypoints: Dict[str, Tuple[float, float, float]]  # name -> (x, y, confidence)
    centroid: Tuple[float, float]
    frame_index: int
    timestamp_sec: float


@dataclass
class TaskEvent:
    """A detected task event with start/end boundaries."""

    track_id: int
    task: str
    start_frame: int
    end_frame: int
    confidence: float
    duration_sec: float


@dataclass
class ZoneEvent:
    """A zone occupancy record."""

    track_id: int
    zone_id: str
    frame_index: int
    timestamp_sec: float


@dataclass
class PersonState:
    """Accumulated per-person state across frames."""

    track_id: int
    frame_history: List[PersonFrame] = field(default_factory=list)
    joint_angle_history: List[Dict[str, float]] = field(default_factory=list)
    velocity_history: List[float] = field(default_factory=list)
    task_log: List[TaskEvent] = field(default_factory=list)
    zone_log: List[ZoneEvent] = field(default_factory=list)
    is_active: bool = False


@dataclass
class PersonMetrics:
    """Final aggregated metrics for a single person."""

    track_id: int
    total_active_frames: int
    total_idle_frames: int
    active_ratio: float
    total_movement_px: float
    zone_dwell_times: Dict[str, float]  # zone_id -> seconds
    task_counts: Dict[str, int]  # task_name -> count
    task_events: List[TaskEvent]
    avg_rula_score: float
    peak_rula_score: float
