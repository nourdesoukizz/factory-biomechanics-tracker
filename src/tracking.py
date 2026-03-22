"""Person tracking module — assigns consistent IDs across frames using ByteTrack + ReID."""

import logging
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.models import PersonFrame

logger = logging.getLogger(__name__)


class KeypointKalmanFilter:
    """2D Kalman filter tracking position and velocity for a keypoint."""

    def __init__(self, process_noise: float = 0.1, measurement_noise: float = 1.0) -> None:
        """Initialize 2D Kalman filter with state [x, y, vx, vy].

        Args:
            process_noise: Process noise covariance scale.
            measurement_noise: Measurement noise covariance scale.
        """
        self._state = np.zeros(4)  # [x, y, vx, vy]
        self._P = np.eye(4) * 1000.0
        self._F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        self._H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)
        self._Q = np.eye(4) * process_noise
        self._R = np.eye(2) * measurement_noise
        self._initialized = False

    def predict(self) -> Tuple[float, float]:
        """Predict next state. Useful during brief occlusions.

        Returns:
            Predicted (x, y) position.
        """
        if not self._initialized:
            return (0.0, 0.0)
        self._state = self._F @ self._state
        self._P = self._F @ self._P @ self._F.T + self._Q
        return (float(self._state[0]), float(self._state[1]))

    def update(self, x: float, y: float) -> Tuple[float, float]:
        """Update with measurement and return smoothed position.

        Args:
            x: Measured x coordinate.
            y: Measured y coordinate.

        Returns:
            Smoothed (x, y) position.
        """
        if not self._initialized:
            self._state = np.array([x, y, 0.0, 0.0])
            self._initialized = True
            return (x, y)

        # Predict
        self._state = self._F @ self._state
        self._P = self._F @ self._P @ self._F.T + self._Q

        # Update
        z = np.array([x, y])
        y_residual = z - self._H @ self._state
        S = self._H @ self._P @ self._H.T + self._R
        K = self._P @ self._H.T @ np.linalg.inv(S)
        self._state += K @ y_residual
        self._P = (np.eye(4) - K @ self._H) @ self._P

        return (float(self._state[0]), float(self._state[1]))

    def get_velocity(self) -> Tuple[float, float]:
        """Get current estimated velocity.

        Returns:
            (vx, vy) velocity components.
        """
        return (float(self._state[2]), float(self._state[3]))


# Keep for backward compatibility
class SimpleKalmanFilter:
    """Simple 1D Kalman filter for smoothing keypoint coordinates."""

    def __init__(self) -> None:
        """Initialize Kalman filter state."""
        self._x = 0.0
        self._p = 1000.0
        self._q = 0.1
        self._r = 1.0
        self._initialized = False

    def update(self, measurement: float) -> float:
        """Update filter with new measurement and return smoothed value."""
        if not self._initialized:
            self._x = measurement
            self._initialized = True
            return self._x
        self._p += self._q
        k = self._p / (self._p + self._r)
        self._x += k * (measurement - self._x)
        self._p *= (1.0 - k)
        return self._x


class TrackReIdentifier:
    """Post-tracking re-identification to merge fragmented tracks."""

    def __init__(self, appearance_weight: float = 0.4,
                 spatial_weight: float = 0.3,
                 temporal_weight: float = 0.3,
                 merge_threshold: float = 0.6,
                 max_gap_frames: int = 150) -> None:
        """Initialize ReID system.

        Args:
            appearance_weight: Weight for color histogram similarity.
            spatial_weight: Weight for predicted spatial proximity.
            temporal_weight: Weight for temporal gap penalty.
            merge_threshold: Minimum similarity to merge tracks.
            max_gap_frames: Maximum frame gap to consider merging.
        """
        self._app_weight = appearance_weight
        self._spatial_weight = spatial_weight
        self._temporal_weight = temporal_weight
        self._merge_threshold = merge_threshold
        self._max_gap_frames = max_gap_frames

        # Per-track appearance histograms (running average)
        self._appearances: Dict[int, np.ndarray] = {}
        # Per-track last known state
        self._last_seen: Dict[int, dict] = {}
        # Track ID mapping: new_id -> original_id
        self._id_map: Dict[int, int] = {}
        # Active track IDs this frame
        self._active_ids: set = set()
        # Recently disappeared tracks
        self._disappeared: Dict[int, dict] = {}

    def _compute_appearance(self, frame_bgr: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        """Compute HSV color histogram of person bbox region.

        Args:
            frame_bgr: Full BGR frame.
            bbox: Bounding box [x1, y1, x2, y2].

        Returns:
            Normalized histogram or None if bbox invalid.
        """
        x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
        h, w = frame_bgr.shape[:2]
        x1, x2 = min(x1, w - 1), min(x2, w)
        y1, y2 = min(y1, h - 1), min(y2, h)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        try:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 4],
                                [0, 180, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            return hist
        except Exception:
            return None

    def _appearance_similarity(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        """Compute appearance similarity between two histograms.

        Args:
            hist1: First histogram.
            hist2: Second histogram.

        Returns:
            Similarity score (0-1).
        """
        return float(cv2.compareHist(
            hist1.astype(np.float32), hist2.astype(np.float32),
            cv2.HISTCMP_CORREL
        ))

    def process(self, frame_bgr: np.ndarray, persons: List[PersonFrame],
                frame_index: int) -> List[PersonFrame]:
        """Process tracked persons and attempt ReID merging.

        Args:
            frame_bgr: Current BGR frame.
            persons: Tracked PersonFrame list from ByteTrack.
            frame_index: Current frame index.

        Returns:
            PersonFrame list with potentially remapped track_ids.
        """
        current_ids = {p.track_id for p in persons}

        # Update disappeared tracks
        for tid in list(self._active_ids):
            if tid not in current_ids and tid in self._last_seen:
                self._disappeared[tid] = {
                    "last_frame": self._last_seen[tid]["frame_index"],
                    "last_centroid": self._last_seen[tid]["centroid"],
                    "last_velocity": self._last_seen[tid].get("velocity", (0, 0)),
                    "appearance": self._appearances.get(tid),
                }

        # Clean up old disappeared tracks
        for tid in list(self._disappeared.keys()):
            if frame_index - self._disappeared[tid]["last_frame"] > self._max_gap_frames:
                del self._disappeared[tid]

        self._active_ids = set()
        result = []

        for person in persons:
            tid = person.track_id

            # Compute appearance
            appearance = self._compute_appearance(frame_bgr, person.bbox)

            # Check if this is a new track that might be a re-appeared person
            if tid not in self._last_seen and self._disappeared:
                best_match = None
                best_score = 0.0

                for old_tid, old_info in self._disappeared.items():
                    score = self._compute_match_score(
                        person, appearance, old_info, frame_index
                    )
                    if score > best_score:
                        best_score = score
                        best_match = old_tid

                if best_match is not None and best_score >= self._merge_threshold:
                    # Merge: remap this track to the old track
                    self._id_map[tid] = best_match
                    del self._disappeared[best_match]
                    logger.debug("ReID: merged track %d → %d (score=%.2f)",
                                 tid, best_match, best_score)

            # Apply ID mapping
            mapped_id = self._id_map.get(tid, tid)

            # Update appearance (running average)
            if appearance is not None:
                if mapped_id in self._appearances:
                    self._appearances[mapped_id] = (
                        0.9 * self._appearances[mapped_id] + 0.1 * appearance
                    )
                else:
                    self._appearances[mapped_id] = appearance

            # Update last seen
            self._last_seen[mapped_id] = {
                "frame_index": frame_index,
                "centroid": person.centroid,
                "velocity": (0, 0),  # Will be refined by Kalman
            }
            self._active_ids.add(mapped_id)

            # Create remapped PersonFrame
            if mapped_id != tid:
                person = PersonFrame(
                    track_id=mapped_id,
                    bbox=person.bbox,
                    keypoints=person.keypoints,
                    centroid=person.centroid,
                    frame_index=person.frame_index,
                    timestamp_sec=person.timestamp_sec,
                )

            result.append(person)

        return result

    def _compute_match_score(self, person: PersonFrame,
                             appearance: Optional[np.ndarray],
                             old_info: dict, frame_index: int) -> float:
        """Compute match score between a new track and a disappeared track.

        Args:
            person: New PersonFrame.
            appearance: New track appearance histogram.
            old_info: Disappeared track info dict.
            frame_index: Current frame index.

        Returns:
            Match score (0-1).
        """
        scores = []
        weights = []

        # Appearance similarity
        if appearance is not None and old_info.get("appearance") is not None:
            app_sim = self._appearance_similarity(appearance, old_info["appearance"])
            # Clamp to 0-1
            app_sim = max(0.0, min(1.0, app_sim))
            scores.append(app_sim)
            weights.append(self._app_weight)

        # Spatial proximity (based on predicted position)
        frame_gap = frame_index - old_info["last_frame"]
        old_cx, old_cy = old_info["last_centroid"]
        vx, vy = old_info.get("last_velocity", (0, 0))
        predicted_x = old_cx + vx * frame_gap
        predicted_y = old_cy + vy * frame_gap

        dist = np.sqrt((person.centroid[0] - predicted_x) ** 2 +
                        (person.centroid[1] - predicted_y) ** 2)
        # Normalize: 0 distance = 1.0, 500px = 0.0
        spatial_sim = max(0.0, 1.0 - dist / 500.0)
        scores.append(spatial_sim)
        weights.append(self._spatial_weight)

        # Temporal penalty (exponential decay)
        temporal_sim = np.exp(-frame_gap / (self._max_gap_frames / 3.0))
        scores.append(float(temporal_sim))
        weights.append(self._temporal_weight)

        if not scores:
            return 0.0

        total_weight = sum(weights)
        return sum(s * w for s, w in zip(scores, weights)) / total_weight if total_weight > 0 else 0.0


class PersonTracker:
    """Tracks persons across frames using ByteTrack via ultralytics."""

    def __init__(self, conf_threshold: float = 0.4, iou_threshold: float = 0.5,
                 keypoint_conf_threshold: float = 0.3, kalman_window: int = 5,
                 max_disappeared_frames: int = 90, device: str = "mps",
                 reid_enabled: bool = False,
                 reid_appearance_weight: float = 0.4,
                 reid_spatial_weight: float = 0.3,
                 reid_temporal_weight: float = 0.3,
                 reid_merge_threshold: float = 0.6,
                 reid_max_gap_frames: int = 150) -> None:
        """Initialize the person tracker.

        Args:
            conf_threshold: Minimum detection confidence for tracking.
            iou_threshold: IoU threshold for matching.
            keypoint_conf_threshold: Minimum keypoint confidence.
            kalman_window: Window size for Kalman smoothing.
            max_disappeared_frames: Frames before a track is dropped.
            device: Inference device.
            reid_enabled: Whether to enable ReID post-processing.
            reid_appearance_weight: Appearance weight for ReID scoring.
            reid_spatial_weight: Spatial weight for ReID scoring.
            reid_temporal_weight: Temporal weight for ReID scoring.
            reid_merge_threshold: Minimum score to merge tracks.
            reid_max_gap_frames: Max frame gap for ReID matching.
        """
        self._conf_threshold = conf_threshold
        self._iou_threshold = iou_threshold
        self._keypoint_conf_threshold = keypoint_conf_threshold
        self._kalman_window = kalman_window
        self._max_disappeared_frames = max_disappeared_frames
        self._device = device
        self._model = None

        # Track history for trajectory drawing
        self.track_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=30))

        # 2D Kalman filters per track_id per keypoint
        self._kalman_filters: Dict[int, Dict[str, KeypointKalmanFilter]] = {}

        # ReID system
        self._reid: Optional[TrackReIdentifier] = None
        if reid_enabled:
            self._reid = TrackReIdentifier(
                appearance_weight=reid_appearance_weight,
                spatial_weight=reid_spatial_weight,
                temporal_weight=reid_temporal_weight,
                merge_threshold=reid_merge_threshold,
                max_gap_frames=reid_max_gap_frames,
            )
            logger.info("ReID post-processing enabled")

    def _load_model(self) -> None:
        """Lazy-load the YOLO model for tracking."""
        if self._model is None:
            try:
                from ultralytics import YOLO
                self._model = YOLO("yolov8m-pose.pt")
                logger.info("Loaded tracking model on device: %s", self._device)
            except Exception as e:
                logger.error("Failed to load tracking model: %s", e)
                raise

    def _get_kalman_filter(self, track_id: int, keypoint_name: str) -> KeypointKalmanFilter:
        """Get or create 2D Kalman filter for a keypoint.

        Args:
            track_id: Person track ID.
            keypoint_name: Name of the keypoint.

        Returns:
            KeypointKalmanFilter instance.
        """
        if track_id not in self._kalman_filters:
            self._kalman_filters[track_id] = {}
        if keypoint_name not in self._kalman_filters[track_id]:
            self._kalman_filters[track_id][keypoint_name] = KeypointKalmanFilter()
        return self._kalman_filters[track_id][keypoint_name]

    def _smooth_keypoints(self, track_id: int,
                          keypoints: Dict[str, Tuple[float, float, float]]) -> Dict[str, Tuple[float, float, float]]:
        """Apply 2D Kalman smoothing to keypoints.

        Args:
            track_id: Person track ID.
            keypoints: Raw keypoints dict.

        Returns:
            Smoothed keypoints dict.
        """
        smoothed = {}
        for name, (x, y, conf) in keypoints.items():
            if conf < self._keypoint_conf_threshold:
                smoothed[name] = (x, y, conf)
                continue

            kf = self._get_kalman_filter(track_id, name)
            sx, sy = kf.update(x, y)
            smoothed[name] = (sx, sy, conf)

        return smoothed

    def update(self, frame_bgr: np.ndarray, frame_index: int = 0,
               timestamp_sec: float = 0.0) -> List[PersonFrame]:
        """Run detection + tracking on a frame, returning tracked PersonFrames.

        Args:
            frame_bgr: BGR image.
            frame_index: Current frame index.
            timestamp_sec: Current timestamp in seconds.

        Returns:
            List of PersonFrame with track_id populated.
        """
        self._load_model()

        try:
            results = self._model.track(
                frame_bgr,
                persist=True,
                conf=self._conf_threshold,
                iou=self._iou_threshold,
                tracker="bytetrack.yaml",
                device=self._device,
                verbose=False
            )
        except Exception as e:
            logger.error("Tracking failed on frame %d: %s", frame_index, e)
            return []

        tracked: List[PersonFrame] = []

        for result in results:
            if result.boxes is None or result.boxes.id is None or result.keypoints is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().numpy()
            kpts = result.keypoints.data.cpu().numpy()

            for i in range(len(boxes)):
                tid = int(track_ids[i])
                bbox = boxes[i].tolist()
                centroid = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

                # Build keypoints dict
                from src.detection import KEYPOINT_NAMES
                keypoints: Dict[str, Tuple[float, float, float]] = {}
                for j, name in enumerate(KEYPOINT_NAMES):
                    x, y, c = float(kpts[i][j][0]), float(kpts[i][j][1]), float(kpts[i][j][2])
                    keypoints[name] = (x, y, c)

                # Apply 2D Kalman smoothing
                keypoints = self._smooth_keypoints(tid, keypoints)

                # Update trajectory history
                self.track_history[tid].append(centroid)

                tracked.append(PersonFrame(
                    track_id=tid,
                    bbox=bbox,
                    keypoints=keypoints,
                    centroid=centroid,
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                ))

        # Apply ReID post-processing
        if self._reid is not None:
            tracked = self._reid.process(frame_bgr, tracked, frame_index)
            # Update track_history for remapped IDs
            for person in tracked:
                self.track_history[person.track_id].append(person.centroid)

        return tracked
