"""Object detection module — detects trays, racks, and items for task confirmation."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# COCO class mappings to factory object categories
# These map standard COCO classes to our factory-relevant categories
OBJECT_CATEGORY_MAP = {
    "bowl": "small_item",
    "cup": "small_item",
    "bottle": "small_item",
    "cell phone": "small_item",
    "book": "small_item",
    "remote": "small_item",
    "dining table": "tray_or_surface",
    "bench": "tray_or_surface",
    "suitcase": "rack_or_cart",
    "backpack": "rack_or_cart",
    "refrigerator": "rack_or_cart",
    "oven": "rack_or_cart",
    "handbag": "tray_or_surface",
    "box": "small_item",
}

# Factory category descriptions for logging
FACTORY_CATEGORIES = {
    "small_item": "Small object suitable for pick and place",
    "tray_or_surface": "Flat surface or tray for lift and place",
    "rack_or_cart": "Large structure like a rack or cart",
}


class ObjectDetector:
    """Detects objects in frames using YOLOv8 for task confirmation."""

    def __init__(self, conf_threshold: float = 0.3, device: str = "mps") -> None:
        """Initialize the object detector.

        Args:
            conf_threshold: Minimum detection confidence.
            device: Inference device.
        """
        self._conf_threshold = conf_threshold
        self._device = device
        self._model = None

    def _load_model(self) -> None:
        """Lazy-load the YOLOv8 model on first use."""
        if self._model is None:
            try:
                from ultralytics import YOLO
                self._model = YOLO("yolov8m.pt")
                logger.info("Loaded YOLOv8m object detection model on device: %s", self._device)
            except Exception as e:
                logger.error("Failed to load object detection model: %s", e)
                raise

    def detect_objects(self, frame_bgr: np.ndarray) -> List[Dict]:
        """Detect objects in a frame.

        Args:
            frame_bgr: BGR image as numpy array.

        Returns:
            List of dicts with keys: class_name, category, bbox, confidence.
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
            logger.error("Object detection failed: %s", e)
            return []

        detections = []

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for i in range(len(boxes)):
                class_id = int(classes[i])
                class_name = result.names.get(class_id, "unknown")
                category = OBJECT_CATEGORY_MAP.get(class_name)

                if category is None:
                    continue

                bbox = boxes[i].tolist()
                detections.append({
                    "class_name": class_name,
                    "category": category,
                    "bbox": bbox,
                    "confidence": float(confs[i]),
                    "center": ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0),
                })

        return detections

    @staticmethod
    def compute_hand_object_proximity(
        keypoints: Dict[str, Tuple[float, float, float]],
        objects: List[Dict],
        keypoint_conf_threshold: float = 0.3
    ) -> Dict[str, Optional[Dict]]:
        """Compute proximity between wrists and detected objects.

        Args:
            keypoints: Person keypoints dict.
            objects: Detected objects list from detect_objects.
            keypoint_conf_threshold: Minimum keypoint confidence.

        Returns:
            Dict with keys: nearest_left_wrist, nearest_right_wrist.
            Each value is a dict with object info and distance, or None.
        """
        result = {
            "nearest_left_wrist": None,
            "nearest_right_wrist": None,
        }

        if not objects:
            return result

        for side in ["left", "right"]:
            wrist_key = f"{side}_wrist"
            if wrist_key not in keypoints:
                continue

            wx, wy, wc = keypoints[wrist_key]
            if wc < keypoint_conf_threshold:
                continue

            best_dist = float("inf")
            best_obj = None

            for obj in objects:
                ox, oy = obj["center"]
                dist = np.sqrt((wx - ox) ** 2 + (wy - oy) ** 2)

                if dist < best_dist:
                    best_dist = dist
                    best_obj = {
                        "category": obj["category"],
                        "class_name": obj["class_name"],
                        "distance_px": round(dist, 1),
                        "confidence": obj["confidence"],
                    }

            result[f"nearest_{side}_wrist"] = best_obj

        return result
