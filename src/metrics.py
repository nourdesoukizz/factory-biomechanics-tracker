"""Metrics engine — per-person metric aggregation, zone tracking, and output export."""

import csv
import json
import logging
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.models import PersonFrame, PersonMetrics, PersonState, TaskEvent, ZoneEvent

logger = logging.getLogger(__name__)


class MetricsEngine:
    """Aggregates per-person metrics across all frames."""

    def __init__(self, fps: float, frame_width: int, frame_height: int,
                 grid_cols: int = 3, grid_rows: int = 2,
                 idle_velocity_threshold: float = 15.0) -> None:
        """Initialize the metrics engine.

        Args:
            fps: Video frames per second.
            frame_width: Video frame width in pixels.
            frame_height: Video frame height in pixels.
            grid_cols: Number of zone grid columns.
            grid_rows: Number of zone grid rows.
            idle_velocity_threshold: Velocity below which a person is idle.
        """
        self._fps = fps
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._grid_cols = grid_cols
        self._grid_rows = grid_rows
        self._idle_threshold = idle_velocity_threshold

        # Zone label grid
        self._zone_labels = self._build_zone_labels()

        # Per-person accumulated state
        self._states: Dict[int, PersonState] = {}

        # Per-person RULA scores
        self._rula_scores: Dict[int, List[float]] = defaultdict(list)

        # Per-person active/idle frame counters
        self._active_frames: Dict[int, int] = defaultdict(int)
        self._idle_frames: Dict[int, int] = defaultdict(int)

        # Per-person total movement
        self._total_movement: Dict[int, float] = defaultdict(float)

        # Per-person zone dwell counts (zone_id -> frame_count)
        self._zone_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def _build_zone_labels(self) -> List[List[str]]:
        """Build zone label grid (e.g., A1-C2 for 3x2).

        Returns:
            2D list of zone labels.
        """
        labels = []
        for row in range(self._grid_rows):
            row_labels = []
            for col in range(self._grid_cols):
                col_letter = chr(ord('A') + col)
                row_number = row + 1
                row_labels.append(f"{col_letter}{row_number}")
            labels.append(row_labels)
        return labels

    def get_zone_for_point(self, x: float, y: float) -> str:
        """Determine which zone a point falls into.

        Args:
            x: X coordinate in pixels.
            y: Y coordinate in pixels.

        Returns:
            Zone label string.
        """
        col = int(x / (self._frame_width / self._grid_cols))
        row = int(y / (self._frame_height / self._grid_rows))
        col = max(0, min(col, self._grid_cols - 1))
        row = max(0, min(row, self._grid_rows - 1))
        return self._zone_labels[row][col]

    def get_zone_labels(self) -> List[List[str]]:
        """Return the zone label grid."""
        return self._zone_labels

    def get_person_state(self, track_id: int) -> PersonState:
        """Get or create PersonState for a track_id.

        Args:
            track_id: Person track ID.

        Returns:
            PersonState object.
        """
        if track_id not in self._states:
            self._states[track_id] = PersonState(track_id=track_id)
        return self._states[track_id]

    def update(self, person: PersonFrame, velocity: float, is_active: bool,
               joint_angles: Dict[str, float], rula_score: float,
               task_events: List[TaskEvent]) -> None:
        """Update metrics for a person in a single frame.

        Args:
            person: PersonFrame data.
            velocity: Centroid velocity in px/sec.
            is_active: Whether the person is active this frame.
            joint_angles: Computed joint angles.
            rula_score: RULA score for this frame.
            task_events: Any newly completed task events.
        """
        state = self.get_person_state(person.track_id)
        state.frame_history.append(person)
        state.joint_angle_history.append(joint_angles)
        state.velocity_history.append(velocity)
        state.is_active = is_active

        # Active/idle tracking
        if is_active:
            self._active_frames[person.track_id] += 1
        else:
            self._idle_frames[person.track_id] += 1

        # Movement tracking (cumulative displacement)
        if len(state.frame_history) >= 2:
            prev = state.frame_history[-2]
            dx = person.centroid[0] - prev.centroid[0]
            dy = person.centroid[1] - prev.centroid[1]
            self._total_movement[person.track_id] += np.sqrt(dx ** 2 + dy ** 2)

        # Zone tracking
        zone_id = self.get_zone_for_point(person.centroid[0], person.centroid[1])
        self._zone_counts[person.track_id][zone_id] += 1
        state.zone_log.append(ZoneEvent(
            track_id=person.track_id,
            zone_id=zone_id,
            frame_index=person.frame_index,
            timestamp_sec=person.timestamp_sec,
        ))

        # RULA tracking
        self._rula_scores[person.track_id].append(rula_score)

        # Task events
        for event in task_events:
            state.task_log.append(event)

    def add_task_events(self, events: List[TaskEvent]) -> None:
        """Add task events from finalization.

        Args:
            events: List of TaskEvents to add.
        """
        for event in events:
            state = self.get_person_state(event.track_id)
            state.task_log.append(event)

    def build_metrics(self, track_id: int) -> PersonMetrics:
        """Build final PersonMetrics for a tracked person.

        Args:
            track_id: Person track ID.

        Returns:
            PersonMetrics object with all computed metrics.
        """
        state = self.get_person_state(track_id)
        active = self._active_frames.get(track_id, 0)
        idle = self._idle_frames.get(track_id, 0)
        total = active + idle

        # Zone dwell times in seconds
        zone_dwell: Dict[str, float] = {}
        for zone_id, count in self._zone_counts.get(track_id, {}).items():
            zone_dwell[zone_id] = count / self._fps if self._fps > 0 else 0.0

        # Task counts
        task_counts: Dict[str, int] = defaultdict(int)
        for event in state.task_log:
            task_counts[event.task] += 1

        # RULA stats
        rula_scores = self._rula_scores.get(track_id, [0.0])
        avg_rula = float(np.mean(rula_scores)) if rula_scores else 0.0
        peak_rula = float(max(rula_scores)) if rula_scores else 0.0

        return PersonMetrics(
            track_id=track_id,
            total_active_frames=active,
            total_idle_frames=idle,
            active_ratio=active / total if total > 0 else 0.0,
            total_movement_px=self._total_movement.get(track_id, 0.0),
            zone_dwell_times=zone_dwell,
            task_counts=dict(task_counts),
            task_events=list(state.task_log),
            avg_rula_score=round(avg_rula, 2),
            peak_rula_score=round(peak_rula, 2),
        )

    def get_all_track_ids(self) -> List[int]:
        """Return all tracked person IDs."""
        return list(self._states.keys())

    def export_csv(self, output_path: str) -> None:
        """Export metrics to CSV — one row per person.

        Args:
            output_path: Path to write CSV file.
        """
        track_ids = sorted(self.get_all_track_ids())
        if not track_ids:
            logger.warning("No tracked persons to export")
            return

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        all_zones = set()
        all_tasks = set()
        metrics_list = []
        for tid in track_ids:
            m = self.build_metrics(tid)
            metrics_list.append(m)
            all_zones.update(m.zone_dwell_times.keys())
            all_tasks.update(m.task_counts.keys())

        sorted_zones = sorted(all_zones)
        sorted_tasks = sorted(all_tasks)

        fieldnames = [
            "track_id", "total_active_frames", "total_idle_frames", "active_ratio",
            "total_movement_px", "avg_rula_score", "peak_rula_score",
        ]
        fieldnames += [f"zone_{z}_sec" for z in sorted_zones]
        fieldnames += [f"task_{t}_count" for t in sorted_tasks]

        try:
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for m in metrics_list:
                    row: Dict[str, Any] = {
                        "track_id": m.track_id,
                        "total_active_frames": m.total_active_frames,
                        "total_idle_frames": m.total_idle_frames,
                        "active_ratio": round(m.active_ratio, 4),
                        "total_movement_px": round(m.total_movement_px, 2),
                        "avg_rula_score": m.avg_rula_score,
                        "peak_rula_score": m.peak_rula_score,
                    }
                    for z in sorted_zones:
                        row[f"zone_{z}_sec"] = round(m.zone_dwell_times.get(z, 0.0), 2)
                    for t in sorted_tasks:
                        row[f"task_{t}_count"] = m.task_counts.get(t, 0)
                    writer.writerow(row)

            logger.info("Exported metrics CSV: %s (%d persons)", output_path, len(metrics_list))
        except Exception as e:
            logger.error("Failed to export CSV: %s", e)
            raise

    def export_json(self, output_path: str) -> None:
        """Export metrics to JSON with full task event detail.

        Args:
            output_path: Path to write JSON file.
        """
        track_ids = sorted(self.get_all_track_ids())
        if not track_ids:
            logger.warning("No tracked persons to export")
            return

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        output = {
            "metadata": {
                "fps": self._fps,
                "frame_width": self._frame_width,
                "frame_height": self._frame_height,
                "grid_cols": self._grid_cols,
                "grid_rows": self._grid_rows,
                "total_persons": len(track_ids),
            },
            "persons": []
        }

        for tid in track_ids:
            m = self.build_metrics(tid)
            person_data = {
                "track_id": m.track_id,
                "total_active_frames": m.total_active_frames,
                "total_idle_frames": m.total_idle_frames,
                "active_ratio": round(m.active_ratio, 4),
                "total_movement_px": round(m.total_movement_px, 2),
                "zone_dwell_times": {k: round(v, 2) for k, v in m.zone_dwell_times.items()},
                "task_counts": m.task_counts,
                "task_events": [asdict(e) for e in m.task_events],
                "avg_rula_score": m.avg_rula_score,
                "peak_rula_score": m.peak_rula_score,
            }
            output["persons"].append(person_data)

        try:
            with open(output_path, "w") as f:
                json.dump(output, f, indent=2)
            logger.info("Exported metrics JSON: %s (%d persons)", output_path, len(track_ids))
        except Exception as e:
            logger.error("Failed to export JSON: %s", e)
            raise
