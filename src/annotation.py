"""Frame annotation module — draws all overlay layers on video frames."""

import logging
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.models import PersonFrame, PersonMetrics

logger = logging.getLogger(__name__)

# Track ID colors — cycle through these
TRACK_COLORS: List[Tuple[int, int, int]] = [
    (60, 76, 231),    # #E74C3C (BGR)
    (219, 152, 52),   # #3498DB
    (113, 204, 46),   # #2ECC71
    (18, 156, 243),   # #F39C12
    (182, 89, 155),   # #9B59B6
    (156, 188, 26),   # #1ABC9C
]

# Task label colors (BGR)
TASK_COLORS: Dict[str, Tuple[int, int, int]] = {
    "pick_and_place": (219, 152, 52),   # #3498DB
    "lift_and_place": (34, 126, 230),   # #E67E22
    "move_rack": (182, 89, 155),        # #9B59B6
    "idle": (166, 165, 149),            # #95A5A6
    "other": (128, 128, 128),
}

# RULA badge colors (BGR)
RULA_COLORS: Dict[str, Tuple[int, int, int]] = {
    "LOW": (113, 204, 46),       # #2ECC71
    "MEDIUM": (15, 196, 241),    # #F1C40F
    "HIGH": (34, 126, 230),      # #E67E22
    "VERY_HIGH": (60, 76, 231),  # #E74C3C
}

# COCO skeleton connections
SKELETON_CONNECTIONS: List[Tuple[str, str]] = [
    ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow"),
    ("left_elbow", "left_wrist"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("right_hip", "right_knee"),
    ("left_knee", "left_ankle"), ("right_knee", "right_ankle"),
]


class FrameAnnotator:
    """Draws all overlay layers on video frames."""

    def __init__(self, frame_width: int, frame_height: int,
                 grid_cols: int = 3, grid_rows: int = 2,
                 zone_labels: Optional[List[List[str]]] = None) -> None:
        """Initialize frame annotator.

        Args:
            frame_width: Video frame width.
            frame_height: Video frame height.
            grid_cols: Number of zone grid columns.
            grid_rows: Number of zone grid rows.
            zone_labels: Optional zone label grid.
        """
        self._width = frame_width
        self._height = frame_height
        self._grid_cols = grid_cols
        self._grid_rows = grid_rows
        self._zone_labels = zone_labels

        # Pre-compute zone grid overlay
        self._zone_overlay = self._create_zone_overlay()

    def _create_zone_overlay(self) -> np.ndarray:
        """Create semi-transparent zone grid overlay.

        Returns:
            BGRA overlay image.
        """
        overlay = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        cell_w = self._width / self._grid_cols
        cell_h = self._height / self._grid_rows

        zone_colors = [
            (200, 200, 200), (180, 200, 180), (200, 180, 180),
            (180, 180, 200), (200, 200, 180), (180, 200, 200),
        ]

        for row in range(self._grid_rows):
            for col in range(self._grid_cols):
                x1 = int(col * cell_w)
                y1 = int(row * cell_h)
                x2 = int((col + 1) * cell_w)
                y2 = int((row + 1) * cell_h)

                color_idx = (row * self._grid_cols + col) % len(zone_colors)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), zone_colors[color_idx], -1)

                # Draw grid lines
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (100, 100, 100), 1)

                # Zone label
                if self._zone_labels and row < len(self._zone_labels) and col < len(self._zone_labels[row]):
                    label = self._zone_labels[row][col]
                    cv2.putText(overlay, label, (x1 + 5, y1 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)

        return overlay

    def _get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """Get color for a track ID.

        Args:
            track_id: Person track ID.

        Returns:
            BGR color tuple.
        """
        return TRACK_COLORS[track_id % len(TRACK_COLORS)]

    def annotate_frame(self, frame: np.ndarray, persons: List[PersonFrame],
                       task_labels: Dict[int, Tuple[str, float]],
                       rula_labels: Dict[int, Tuple[float, str]],
                       track_histories: Dict[int, deque],
                       current_metrics: Optional[Dict[int, Dict]] = None,
                       frame_index: int = 0, timestamp_sec: float = 0.0) -> np.ndarray:
        """Annotate a frame with all overlay layers.

        Args:
            frame: BGR image to annotate.
            persons: List of PersonFrame objects for this frame.
            task_labels: Dict of track_id -> (task_name, confidence).
            rula_labels: Dict of track_id -> (rula_score, rula_label).
            track_histories: Dict of track_id -> deque of centroid positions.
            current_metrics: Optional dict of track_id -> live metric data.
            frame_index: Current frame index.
            timestamp_sec: Current timestamp.

        Returns:
            Annotated frame as numpy array.
        """
        output = frame.copy()

        # Layer 1: Zone grid (semi-transparent)
        cv2.addWeighted(self._zone_overlay, 0.15, output, 1.0, 0, output)

        for person in persons:
            color = self._get_track_color(person.track_id)
            x1, y1, x2, y2 = [int(v) for v in person.bbox]

            # Layer 2: Bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # Layer 3: Skeleton
            self._draw_skeleton(output, person.keypoints, color)

            # Layer 4: Centroid trajectory trail
            if person.track_id in track_histories:
                self._draw_trajectory(output, track_histories[person.track_id], color)

            # Layer 5: Track ID label + task label
            task_name, task_conf = task_labels.get(person.track_id, ("idle", 0.0))
            self._draw_labels(output, x1, y1, person.track_id, task_name, task_conf)

            # Layer 6: RULA badge
            rula_score, rula_label = rula_labels.get(person.track_id, (1.0, "LOW"))
            self._draw_rula_badge(output, x2, y1, rula_label)

        # Layer 7: Metric panel (bottom-left)
        if current_metrics:
            self._draw_metric_panel(output, current_metrics)

        # Layer 8: Frame counter + timestamp (top-right)
        self._draw_frame_info(output, frame_index, timestamp_sec)

        return output

    def _draw_skeleton(self, frame: np.ndarray, keypoints: Dict[str, Tuple[float, float, float]],
                       color: Tuple[int, int, int]) -> None:
        """Draw skeleton overlay connecting keypoints.

        Args:
            frame: Image to draw on.
            keypoints: Named keypoints with confidence.
            color: BGR color for the skeleton.
        """
        if not keypoints:
            return

        for kp1_name, kp2_name in SKELETON_CONNECTIONS:
            if kp1_name not in keypoints or kp2_name not in keypoints:
                continue

            x1, y1, c1 = keypoints[kp1_name]
            x2, y2, c2 = keypoints[kp2_name]

            min_conf = min(c1, c2)
            if min_conf < 0.3:
                continue

            # Opacity based on confidence
            alpha = max(0.3, min(1.0, min_conf))
            line_color = tuple(int(c * alpha) for c in color)

            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), line_color, 2)

        # Draw keypoint dots
        for name, (x, y, c) in keypoints.items():
            if c >= 0.3:
                cv2.circle(frame, (int(x), int(y)), 3, color, -1)

    def _draw_trajectory(self, frame: np.ndarray, history: deque,
                         color: Tuple[int, int, int]) -> None:
        """Draw fading centroid trajectory trail.

        Args:
            frame: Image to draw on.
            history: Deque of (x, y) centroid positions.
            color: BGR color.
        """
        points = list(history)
        num_points = len(points)
        if num_points < 2:
            return

        for i in range(1, num_points):
            alpha = i / num_points  # Fades in
            line_color = tuple(int(c * alpha) for c in color)
            thickness = max(1, int(2 * alpha))
            pt1 = (int(points[i - 1][0]), int(points[i - 1][1]))
            pt2 = (int(points[i][0]), int(points[i][1]))
            cv2.line(frame, pt1, pt2, line_color, thickness)

    def _draw_labels(self, frame: np.ndarray, x: int, y: int,
                     track_id: int, task_name: str, confidence: float) -> None:
        """Draw track ID and task label above bounding box.

        Args:
            frame: Image to draw on.
            x: Left edge of bounding box.
            y: Top edge of bounding box.
            track_id: Person track ID.
            task_name: Current task name.
            confidence: Task confidence.
        """
        # Track ID label
        id_text = f"Person {track_id}"
        (tw, th), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x, y - th - 10), (x + tw + 4, y - 2), (0, 0, 0), -1)
        cv2.putText(frame, id_text, (x + 2, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Task label with color
        task_color = TASK_COLORS.get(task_name, TASK_COLORS["idle"])
        task_text = f"{task_name} ({confidence:.0%})"
        (tw2, th2), _ = cv2.getTextSize(task_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        label_y = y - th - 15
        cv2.rectangle(frame, (x, label_y - th2 - 6), (x + tw2 + 4, label_y), task_color, -1)
        cv2.putText(frame, task_text, (x + 2, label_y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Confidence bar
        bar_y = y - th - 16 - th2 - 10
        bar_width = int(100 * confidence)
        cv2.rectangle(frame, (x, bar_y), (x + 100, bar_y + 5), (50, 50, 50), -1)
        cv2.rectangle(frame, (x, bar_y), (x + bar_width, bar_y + 5), task_color, -1)

    def _draw_rula_badge(self, frame: np.ndarray, x: int, y: int,
                         rula_label: str) -> None:
        """Draw RULA risk badge as colored circle.

        Args:
            frame: Image to draw on.
            x: Right edge of bounding box.
            y: Top edge of bounding box.
            rula_label: RULA risk label.
        """
        badge_color = RULA_COLORS.get(rula_label, RULA_COLORS["LOW"])
        center = (x - 10, y + 10)
        cv2.circle(frame, center, 8, badge_color, -1)
        cv2.circle(frame, center, 8, (255, 255, 255), 1)

    def _draw_metric_panel(self, frame: np.ndarray,
                           metrics: Dict[int, Dict]) -> None:
        """Draw mini metric panel at bottom-left.

        Args:
            frame: Image to draw on.
            metrics: Dict of track_id -> metric data dict.
        """
        num_persons = len(metrics)
        panel_h = 20 + num_persons * 18
        panel_w = 280
        x, y = 10, self._height - panel_h - 10

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Header
        cv2.putText(frame, "Live Metrics", (x + 5, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Per-person metrics
        for i, (tid, data) in enumerate(sorted(metrics.items())):
            line_y = y + 30 + i * 18
            color = self._get_track_color(tid)
            active_pct = data.get("active_ratio", 0.0) * 100
            movement = data.get("total_movement", 0.0)
            text = f"P{tid}: Active {active_pct:.0f}% | Move {movement:.0f}px"
            cv2.putText(frame, text, (x + 5, line_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    def _draw_frame_info(self, frame: np.ndarray, frame_index: int,
                         timestamp_sec: float) -> None:
        """Draw frame counter and timestamp at top-right.

        Args:
            frame: Image to draw on.
            frame_index: Current frame index.
            timestamp_sec: Current timestamp.
        """
        minutes = int(timestamp_sec // 60)
        seconds = timestamp_sec % 60
        text = f"Frame {frame_index} | {minutes:02d}:{seconds:05.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        x = self._width - tw - 10
        y = 25
        cv2.rectangle(frame, (x - 5, y - th - 5), (x + tw + 5, y + 5), (0, 0, 0), -1)
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
