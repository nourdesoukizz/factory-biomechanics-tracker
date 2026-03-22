"""Person detection module — wraps YOLOv8 pose estimation model."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.models import PersonFrame

logger = logging.getLogger(__name__)

# COCO 17-keypoint names in order
KEYPOINT_NAMES: List[str] = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


class PoseDetector:
    """Detects persons and their pose keypoints using YOLOv8-pose."""

    def __init__(self, conf_threshold: float = 0.4, keypoint_conf_threshold: float = 0.3,
                 device: str = "mps") -> None:
        """Initialize the pose detector.

        Args:
            conf_threshold: Minimum detection confidence.
            keypoint_conf_threshold: Minimum keypoint confidence to include.
            device: Inference device (mps for Apple Silicon).
        """
        self._conf_threshold = conf_threshold
        self._keypoint_conf_threshold = keypoint_conf_threshold
        self._device = device
        self._model = None

    def _load_model(self) -> None:
        """Lazy-load the YOLO model on first use."""
        if self._model is None:
            try:
                from ultralytics import YOLO
                # DECISION: Using yolov8m-pose.pt per CLAUDE.md (not yolov8x-pose.pt from DESIGN.md)
                self._model = YOLO("yolov8m-pose.pt")
                logger.info("Loaded YOLOv8m-pose model on device: %s", self._device)
            except Exception as e:
                logger.error("Failed to load YOLO model: %s", e)
                raise

    def detect(self, frame_bgr: np.ndarray, frame_index: int = 0,
               timestamp_sec: float = 0.0) -> List[PersonFrame]:
        """Detect persons and keypoints in a single frame.

        Args:
            frame_bgr: BGR image as numpy array.
            frame_index: Current frame index.
            timestamp_sec: Current timestamp in seconds.

        Returns:
            List of PersonFrame objects (without track_id assigned yet).
        """
        self._load_model()

        try:
            results = self._model(
                frame_bgr,
                conf=self._conf_threshold,
                device=self._device,
                verbose=False
            )
        except Exception as e:
            logger.error("Detection failed on frame %d: %s", frame_index, e)
            return []

        detections: List[PersonFrame] = []

        for result in results:
            if result.boxes is None or result.keypoints is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            kpts = result.keypoints.data.cpu().numpy()  # (N, 17, 3) — x, y, conf

            for i in range(len(boxes)):
                if confs[i] < self._conf_threshold:
                    continue

                bbox = boxes[i].tolist()
                centroid = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

                keypoints: Dict[str, Tuple[float, float, float]] = {}
                for j, name in enumerate(KEYPOINT_NAMES):
                    x, y, c = float(kpts[i][j][0]), float(kpts[i][j][1]), float(kpts[i][j][2])
                    keypoints[name] = (x, y, c)

                detections.append(PersonFrame(
                    track_id=-1,  # Not yet assigned
                    bbox=bbox,
                    keypoints=keypoints,
                    centroid=centroid,
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                ))

        return detections
