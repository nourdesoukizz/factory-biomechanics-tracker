"""Biomechanics analysis module — joint angles, velocity, and RULA scoring."""

import logging
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.models import PersonFrame

logger = logging.getLogger(__name__)

# RULA risk level labels
RULA_LOW = "LOW"
RULA_MEDIUM = "MEDIUM"
RULA_HIGH = "HIGH"
RULA_VERY_HIGH = "VERY_HIGH"

# Standard RULA Table A: upper_arm(1-6) x lower_arm(1-3) x wrist(1-4), wrist_twist variant
# Dimensions: [upper_arm_score][lower_arm_score][wrist_score][wrist_twist-1]
# Source: McAtamney & Corlett (1993) RULA worksheet
RULA_TABLE_A = np.array([
    # upper_arm = 1
    [[[1, 2], [2, 2], [2, 3], [3, 3]],
     [[2, 2], [2, 2], [3, 3], [3, 3]],
     [[2, 3], [3, 3], [3, 3], [4, 4]]],
    # upper_arm = 2
    [[[2, 3], [3, 3], [3, 4], [4, 4]],
     [[3, 3], [3, 3], [3, 4], [4, 4]],
     [[3, 4], [4, 4], [4, 4], [5, 5]]],
    # upper_arm = 3
    [[[3, 3], [4, 4], [4, 4], [5, 5]],
     [[3, 4], [4, 4], [4, 4], [5, 5]],
     [[4, 4], [4, 4], [4, 5], [5, 5]]],
    # upper_arm = 4
    [[[4, 4], [4, 4], [4, 5], [5, 5]],
     [[4, 4], [4, 4], [4, 5], [5, 5]],
     [[4, 4], [4, 5], [5, 5], [6, 6]]],
    # upper_arm = 5
    [[[5, 5], [5, 5], [5, 6], [6, 7]],
     [[5, 6], [6, 6], [6, 7], [7, 7]],
     [[6, 6], [6, 7], [7, 7], [7, 8]]],
    # upper_arm = 6
    [[[7, 7], [7, 7], [7, 8], [8, 9]],
     [[8, 8], [8, 8], [8, 9], [9, 9]],
     [[9, 9], [9, 9], [9, 9], [9, 9]]],
])

# Standard RULA Table B: neck(1-6) x trunk(1-6) x legs(1-2)
RULA_TABLE_B = np.array([
    # neck = 1
    [[1, 3], [2, 3], [3, 4], [5, 5], [6, 6], [7, 7]],
    # neck = 2
    [[2, 3], [2, 3], [4, 5], [5, 5], [6, 7], [7, 7]],
    # neck = 3
    [[3, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 7]],
    # neck = 4
    [[5, 5], [5, 6], [6, 7], [7, 7], [7, 7], [8, 8]],
    # neck = 5
    [[7, 7], [7, 7], [7, 8], [8, 8], [8, 8], [8, 8]],
    # neck = 6
    [[8, 8], [8, 8], [8, 8], [8, 9], [9, 9], [9, 9]],
])

# Standard RULA Table C: score_a(1-8+) x score_b(1-7+) → final score
RULA_TABLE_C = np.array([
    [1, 2, 3, 3, 4, 5, 5],
    [2, 2, 3, 4, 4, 5, 5],
    [3, 3, 3, 4, 4, 5, 6],
    [3, 3, 3, 4, 5, 6, 6],
    [4, 4, 4, 5, 6, 7, 7],
    [4, 4, 5, 6, 6, 7, 7],
    [5, 5, 6, 6, 7, 7, 7],
    [5, 5, 6, 7, 7, 7, 7],
])

# Expected limb length ratios relative to person height (for foreshortening detection)
LIMB_RATIOS = {
    "upper_arm": 0.186,   # shoulder to elbow
    "forearm": 0.146,     # elbow to wrist
    "thigh": 0.245,       # hip to knee
    "shin": 0.246,        # knee to ankle
    "torso": 0.288,       # shoulder to hip
}


def _angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute angle in degrees between two 2D vectors using arctan2.

    Args:
        v1: First vector as numpy array.
        v2: Second vector as numpy array.

    Returns:
        Angle in degrees (0-180).
    """
    dot = np.dot(v1, v2)
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    angle = np.abs(np.degrees(np.arctan2(cross, dot)))
    return float(angle)


def _get_keypoint_xy(keypoints: Dict[str, Tuple[float, float, float]],
                     name: str, min_conf: float = 0.3) -> Optional[np.ndarray]:
    """Extract (x, y) from keypoints if confidence is sufficient.

    Args:
        keypoints: Keypoints dictionary.
        name: Keypoint name.
        min_conf: Minimum confidence threshold.

    Returns:
        Numpy array [x, y] or None if below threshold.
    """
    if name not in keypoints:
        return None
    x, y, c = keypoints[name]
    if c < min_conf:
        return None
    return np.array([x, y])


def _estimate_person_height(keypoints: Dict[str, Tuple[float, float, float]],
                            min_conf: float = 0.3) -> float:
    """Estimate person height in pixels from keypoints.

    Args:
        keypoints: Named keypoints.
        min_conf: Minimum confidence.

    Returns:
        Estimated height in pixels, or 0 if insufficient keypoints.
    """
    nose = _get_keypoint_xy(keypoints, "nose", min_conf)
    l_ankle = _get_keypoint_xy(keypoints, "left_ankle", min_conf)
    r_ankle = _get_keypoint_xy(keypoints, "right_ankle", min_conf)

    if nose is None:
        return 0.0

    ankle = None
    if l_ankle is not None and r_ankle is not None:
        ankle = (l_ankle + r_ankle) / 2.0
    elif l_ankle is not None:
        ankle = l_ankle
    elif r_ankle is not None:
        ankle = r_ankle

    if ankle is None:
        return 0.0

    return float(np.linalg.norm(nose - ankle))


class BiomechanicsAnalyzer:
    """Computes joint angles, velocity, and RULA ergonomic risk scores."""

    def __init__(self, idle_velocity_threshold: float = 15.0,
                 trunk_flex_threshold: float = 20.0,
                 wrist_above_shoulder_threshold: float = 0.0,
                 keypoint_conf_threshold: float = 0.3,
                 full_rula: bool = False,
                 force_load_score: int = 0,
                 angle_smoothing_window: int = 5) -> None:
        """Initialize the biomechanics analyzer.

        Args:
            idle_velocity_threshold: Velocity below this (px/sec) is idle.
            trunk_flex_threshold: Trunk flexion threshold in degrees.
            wrist_above_shoulder_threshold: Threshold for wrist above shoulder.
            keypoint_conf_threshold: Minimum keypoint confidence.
            full_rula: If True, use full RULA scoring with Tables A/B/C.
            force_load_score: Force/load adjustment for RULA (0-3).
            angle_smoothing_window: Number of frames for temporal angle smoothing.
        """
        self._idle_threshold = idle_velocity_threshold
        self._trunk_flex_threshold = trunk_flex_threshold
        self._wrist_above_shoulder_threshold = wrist_above_shoulder_threshold
        self._keypoint_conf = keypoint_conf_threshold
        self._full_rula = full_rula
        self._force_load = min(3, max(0, force_load_score))
        self._smoothing_window = angle_smoothing_window

        # Per-track angle history for temporal smoothing
        self._angle_history: Dict[int, deque] = {}

    def _get_smoothed_angles(self, track_id: int, angles: Dict[str, float]) -> Dict[str, float]:
        """Apply temporal smoothing to joint angles.

        Args:
            track_id: Person track ID.
            angles: Current frame angles.

        Returns:
            Smoothed angles.
        """
        if track_id not in self._angle_history:
            self._angle_history[track_id] = deque(maxlen=self._smoothing_window)

        self._angle_history[track_id].append(angles)
        history = self._angle_history[track_id]

        if len(history) < 2:
            return angles

        smoothed = {}
        all_keys = set()
        for h in history:
            all_keys.update(h.keys())

        for key in all_keys:
            values = [h.get(key, 0.0) for h in history]
            # Weighted average — more recent frames get more weight
            weights = np.linspace(0.5, 1.0, len(values))
            smoothed[key] = float(np.average(values, weights=weights))

        return smoothed

    def compute_foreshortening_confidence(self, keypoints: Dict[str, Tuple[float, float, float]]) -> Dict[str, float]:
        """Detect foreshortening in limb segments.

        Compares observed 2D limb length to expected proportion of person height.
        Low confidence means the limb is likely pointing toward/away from camera.

        Args:
            keypoints: Named keypoints.

        Returns:
            Dict of limb_name -> confidence (0-1).
        """
        conf = self._keypoint_conf
        person_height = _estimate_person_height(keypoints, conf)

        if person_height <= 0:
            return {"left_arm": 1.0, "right_arm": 1.0, "trunk": 1.0}

        result = {}

        # Check each limb segment
        limb_checks = {
            "left_arm": [("left_shoulder", "left_elbow", "upper_arm"),
                         ("left_elbow", "left_wrist", "forearm")],
            "right_arm": [("right_shoulder", "right_elbow", "upper_arm"),
                          ("right_elbow", "right_wrist", "forearm")],
            "trunk": [("left_shoulder", "left_hip", "torso")],
        }

        for limb_name, segments in limb_checks.items():
            segment_confs = []
            for kp1_name, kp2_name, ratio_key in segments:
                kp1 = _get_keypoint_xy(keypoints, kp1_name, conf)
                kp2 = _get_keypoint_xy(keypoints, kp2_name, conf)

                if kp1 is None or kp2 is None:
                    segment_confs.append(0.5)  # Unknown — neutral confidence
                    continue

                observed_len = float(np.linalg.norm(kp2 - kp1))
                expected_len = person_height * LIMB_RATIOS.get(ratio_key, 0.2)

                if expected_len <= 0:
                    segment_confs.append(0.5)
                    continue

                ratio = observed_len / expected_len
                # Confidence is high when ratio is near 1.0, drops when much shorter
                if ratio >= 0.6:
                    segment_confs.append(min(1.0, ratio))
                else:
                    segment_confs.append(max(0.1, ratio / 0.6))

            result[limb_name] = float(np.mean(segment_confs)) if segment_confs else 1.0

        return result

    def compute_joint_angles(self, keypoints: Dict[str, Tuple[float, float, float]],
                             track_id: Optional[int] = None) -> Dict[str, float]:
        """Compute joint angles from keypoints.

        Args:
            keypoints: Named keypoints with (x, y, confidence).
            track_id: Optional track ID for temporal smoothing.

        Returns:
            Dictionary of angle names to angle values in degrees.
        """
        angles: Dict[str, float] = {}
        conf = self._keypoint_conf

        # Elbow angles
        for side in ["left", "right"]:
            shoulder = _get_keypoint_xy(keypoints, f"{side}_shoulder", conf)
            elbow = _get_keypoint_xy(keypoints, f"{side}_elbow", conf)
            wrist = _get_keypoint_xy(keypoints, f"{side}_wrist", conf)

            if shoulder is not None and elbow is not None and wrist is not None:
                v1 = shoulder - elbow
                v2 = wrist - elbow
                angles[f"elbow_{side}"] = _angle_between_vectors(v1, v2)
            else:
                angles[f"elbow_{side}"] = 0.0

        # Shoulder angles
        for side in ["left", "right"]:
            hip = _get_keypoint_xy(keypoints, f"{side}_hip", conf)
            shoulder = _get_keypoint_xy(keypoints, f"{side}_shoulder", conf)
            elbow = _get_keypoint_xy(keypoints, f"{side}_elbow", conf)

            if hip is not None and shoulder is not None and elbow is not None:
                v1 = hip - shoulder
                v2 = elbow - shoulder
                angles[f"shoulder_{side}"] = _angle_between_vectors(v1, v2)
            else:
                angles[f"shoulder_{side}"] = 0.0

        # Trunk flexion
        l_hip = _get_keypoint_xy(keypoints, "left_hip", conf)
        r_hip = _get_keypoint_xy(keypoints, "right_hip", conf)
        l_shoulder = _get_keypoint_xy(keypoints, "left_shoulder", conf)
        r_shoulder = _get_keypoint_xy(keypoints, "right_shoulder", conf)

        if l_hip is not None and r_hip is not None and l_shoulder is not None and r_shoulder is not None:
            mid_hip = (l_hip + r_hip) / 2.0
            mid_shoulder = (l_shoulder + r_shoulder) / 2.0
            spine_vec = mid_shoulder - mid_hip
            vertical = np.array([0.0, -1.0])
            angles["trunk_flex"] = _angle_between_vectors(spine_vec, vertical)

            # Trunk side-bend: difference in shoulder line angle vs hip line angle
            shoulder_vec = r_shoulder - l_shoulder
            hip_vec = r_hip - l_hip
            shoulder_angle = np.degrees(np.arctan2(shoulder_vec[1], shoulder_vec[0]))
            hip_angle = np.degrees(np.arctan2(hip_vec[1], hip_vec[0]))
            angles["trunk_side_bend"] = abs(shoulder_angle - hip_angle)
        else:
            angles["trunk_flex"] = 0.0
            angles["trunk_side_bend"] = 0.0

        # Neck flexion: angle between mid_shoulder → nose and vertical
        nose = _get_keypoint_xy(keypoints, "nose", conf)
        if nose is not None and l_shoulder is not None and r_shoulder is not None:
            mid_shoulder = (l_shoulder + r_shoulder) / 2.0
            neck_vec = nose - mid_shoulder
            vertical = np.array([0.0, -1.0])
            angles["neck_flex"] = _angle_between_vectors(neck_vec, vertical)

            # Neck side-bend: from ear positions
            l_ear = _get_keypoint_xy(keypoints, "left_ear", conf)
            r_ear = _get_keypoint_xy(keypoints, "right_ear", conf)
            if l_ear is not None and r_ear is not None:
                ear_diff = abs(l_ear[1] - r_ear[1])
                ear_dist = np.linalg.norm(l_ear - r_ear)
                angles["neck_side_bend"] = float(np.degrees(np.arcsin(
                    min(1.0, ear_diff / ear_dist)))) if ear_dist > 0 else 0.0
            else:
                angles["neck_side_bend"] = 0.0
        else:
            angles["neck_flex"] = 0.0
            angles["neck_side_bend"] = 0.0

        # Wrist height relative to shoulder
        for side in ["left", "right"]:
            wrist = _get_keypoint_xy(keypoints, f"{side}_wrist", conf)
            shoulder = _get_keypoint_xy(keypoints, f"{side}_shoulder", conf)

            if wrist is not None and shoulder is not None:
                height_diff = shoulder[1] - wrist[1]
                hip = _get_keypoint_xy(keypoints, f"{side}_hip", conf)
                if hip is not None:
                    torso_len = np.linalg.norm(shoulder - hip)
                    if torso_len > 0:
                        height_diff = height_diff / torso_len
                angles[f"wrist_height_{side}"] = float(height_diff)
            else:
                angles[f"wrist_height_{side}"] = 0.0

        # Foreshortening confidence
        fconf = self.compute_foreshortening_confidence(keypoints)
        angles["foreshortening_conf_left_arm"] = fconf.get("left_arm", 1.0)
        angles["foreshortening_conf_right_arm"] = fconf.get("right_arm", 1.0)

        # Apply temporal smoothing if track_id provided
        if track_id is not None:
            angles = self._get_smoothed_angles(track_id, angles)

        return angles

    def compute_velocity(self, current: PersonFrame, previous: Optional[PersonFrame],
                         fps: float) -> float:
        """Compute centroid velocity in pixels per second.

        Args:
            current: Current frame PersonFrame.
            previous: Previous frame PersonFrame (or None for first frame).
            fps: Video frames per second.

        Returns:
            Velocity in pixels/second.
        """
        if previous is None or fps <= 0:
            return 0.0

        dx = current.centroid[0] - previous.centroid[0]
        dy = current.centroid[1] - previous.centroid[1]
        displacement = np.sqrt(dx ** 2 + dy ** 2)

        frame_diff = current.frame_index - previous.frame_index
        if frame_diff <= 0:
            return 0.0

        time_diff = frame_diff / fps
        return float(displacement / time_diff)

    def compute_rula_score(self, joint_angles: Dict[str, float]) -> Tuple[float, str]:
        """Compute RULA ergonomic risk score.

        Uses full RULA if self._full_rula is True, otherwise simplified.

        Args:
            joint_angles: Dictionary of computed joint angles.

        Returns:
            Tuple of (score 1-7, risk label string).
        """
        if self._full_rula:
            return self._compute_full_rula(joint_angles)
        return self._compute_simplified_rula(joint_angles)

    def _compute_full_rula(self, joint_angles: Dict[str, float]) -> Tuple[float, str]:
        """Compute full RULA score with Groups A and B and lookup tables.

        Args:
            joint_angles: Dictionary of computed joint angles.

        Returns:
            Tuple of (score 1-7, risk label string).
        """
        # === GROUP A: Upper extremity ===

        # Step 1: Upper arm score (1-6)
        shoulder_angle = max(
            joint_angles.get("shoulder_left", 0.0),
            joint_angles.get("shoulder_right", 0.0)
        )
        if shoulder_angle <= 20:
            upper_arm = 1
        elif shoulder_angle <= 45:
            upper_arm = 2
        elif shoulder_angle <= 90:
            upper_arm = 3
        else:
            upper_arm = 4

        # Adjustment: +1 if shoulder raised (wrist above shoulder significantly)
        wh = max(joint_angles.get("wrist_height_left", 0.0),
                 joint_angles.get("wrist_height_right", 0.0))
        if wh > 0.8:
            upper_arm = min(6, upper_arm + 1)

        # Step 2: Lower arm score (1-3)
        elbow_left = joint_angles.get("elbow_left", 90.0)
        elbow_right = joint_angles.get("elbow_right", 90.0)
        # RULA: 60-100 degrees flexion = score 1; else = score 2
        elbow_ok_left = 60 <= elbow_left <= 100
        elbow_ok_right = 60 <= elbow_right <= 100
        if elbow_ok_left or elbow_ok_right:
            lower_arm = 1
        else:
            lower_arm = 2
        # Adjustment: +1 if arm crosses midline (not detectable from 2D — skip)
        lower_arm = min(3, lower_arm)

        # Step 3: Wrist score (1-4)
        wrist_h_left = abs(joint_angles.get("wrist_height_left", 0.0))
        wrist_h_right = abs(joint_angles.get("wrist_height_right", 0.0))
        max_wrist_dev = max(wrist_h_left, wrist_h_right)
        if max_wrist_dev <= 0.1:
            wrist_score = 1  # Neutral
        elif max_wrist_dev <= 0.3:
            wrist_score = 2  # 0-15 degrees
        elif max_wrist_dev <= 0.6:
            wrist_score = 3  # 15+ degrees
        else:
            wrist_score = 4  # Extreme
        wrist_score = min(4, wrist_score)

        # Step 4: Wrist twist (1-2) — default to 1 (midrange, not detectable in 2D)
        wrist_twist = 1

        # Step 5: Table A lookup
        ua_idx = min(5, max(0, upper_arm - 1))
        la_idx = min(2, max(0, lower_arm - 1))
        ws_idx = min(3, max(0, wrist_score - 1))
        wt_idx = min(1, max(0, wrist_twist - 1))
        score_a = int(RULA_TABLE_A[ua_idx][la_idx][ws_idx][wt_idx])

        # Step 6-7: Muscle use + force/load
        score_a += 1  # +1 for repetitive work (assumed in factory)
        score_a += self._force_load

        # === GROUP B: Neck, trunk, legs ===

        # Step 8: Neck score (1-6)
        neck_flex = joint_angles.get("neck_flex", 0.0)
        if neck_flex <= 10:
            neck_score = 1
        elif neck_flex <= 20:
            neck_score = 2
        elif neck_flex <= 30:
            neck_score = 3
        else:
            neck_score = 4
        # Adjustment: +1 if neck side-bent
        if joint_angles.get("neck_side_bend", 0.0) > 10:
            neck_score = min(6, neck_score + 1)

        # Step 9: Trunk score (1-6)
        trunk_flex = joint_angles.get("trunk_flex", 0.0)
        if trunk_flex <= 5:
            trunk_score = 1  # Upright
        elif trunk_flex <= 20:
            trunk_score = 2
        elif trunk_flex <= 60:
            trunk_score = 3
        else:
            trunk_score = 4
        # Adjustment: +1 if side-bent
        if joint_angles.get("trunk_side_bend", 0.0) > 10:
            trunk_score = min(6, trunk_score + 1)

        # Step 10: Legs (1-2) — score 1 if stable (both ankles visible = balanced)
        legs_score = 1  # Default: balanced bilateral support

        # Step 11: Table B lookup
        nk_idx = min(5, max(0, neck_score - 1))
        tr_idx = min(5, max(0, trunk_score - 1))
        lg_idx = min(1, max(0, legs_score - 1))
        score_b = int(RULA_TABLE_B[nk_idx][tr_idx][lg_idx])

        # Step 12-13: Muscle use + force/load for Group B
        score_b += 1  # +1 for repetitive work
        score_b += self._force_load

        # === FINAL: Table C lookup ===
        sa_idx = min(7, max(0, score_a - 1))
        sb_idx = min(6, max(0, score_b - 1))
        final_score = int(RULA_TABLE_C[sa_idx][sb_idx])
        final_score = max(1, min(7, final_score))

        # Risk label
        if final_score <= 2:
            label = RULA_LOW
        elif final_score <= 4:
            label = RULA_MEDIUM
        elif final_score <= 6:
            label = RULA_HIGH
        else:
            label = RULA_VERY_HIGH

        return float(final_score), label

    def _compute_simplified_rula(self, joint_angles: Dict[str, float]) -> Tuple[float, str]:
        """Compute simplified RULA score (backward compatible).

        Args:
            joint_angles: Dictionary of computed joint angles.

        Returns:
            Tuple of (score 1-7, risk label string).
        """
        shoulder_angle = max(
            joint_angles.get("shoulder_left", 0.0),
            joint_angles.get("shoulder_right", 0.0)
        )
        if shoulder_angle <= 20:
            upper_arm = 1
        elif shoulder_angle <= 45:
            upper_arm = 2
        elif shoulder_angle <= 90:
            upper_arm = 3
        else:
            upper_arm = 4

        elbow_left = joint_angles.get("elbow_left", 90.0)
        elbow_right = joint_angles.get("elbow_right", 90.0)
        elbow_worst = max(abs(elbow_left - 80), abs(elbow_right - 80))
        if elbow_worst <= 40:
            lower_arm = 1
        else:
            lower_arm = 2

        wrist_h_left = joint_angles.get("wrist_height_left", 0.0)
        wrist_h_right = joint_angles.get("wrist_height_right", 0.0)
        max_wrist_h = max(abs(wrist_h_left), abs(wrist_h_right))
        if max_wrist_h <= 0.2:
            wrist_score = 1
        elif max_wrist_h <= 0.5:
            wrist_score = 2
        else:
            wrist_score = 3

        trunk = joint_angles.get("trunk_flex", 0.0)
        if trunk <= 10:
            trunk_mod = 0
        elif trunk <= 20:
            trunk_mod = 1
        elif trunk <= 45:
            trunk_mod = 2
        else:
            trunk_mod = 3

        raw = upper_arm + lower_arm + wrist_score + trunk_mod
        score = max(1.0, min(7.0, 1.0 + (raw - 3) * (6.0 / 9.0)))
        score = round(score, 1)

        if score <= 2:
            label = RULA_LOW
        elif score <= 4:
            label = RULA_MEDIUM
        elif score <= 6:
            label = RULA_HIGH
        else:
            label = RULA_VERY_HIGH

        return score, label

    def is_idle(self, velocity: float) -> bool:
        """Determine if a person is idle based on velocity.

        Args:
            velocity: Centroid velocity in px/sec.

        Returns:
            True if idle.
        """
        return velocity < self._idle_threshold
