"""Shared test fixtures for the factory biomechanics tracker test suite."""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import PersonFrame, TaskEvent


def _make_keypoints(**overrides):
    """Create a full set of 17 COCO keypoints with defaults.

    Default: standing upright person centered at (300, 300) in a 640x480 frame.
    All keypoints have confidence 0.9 by default.
    """
    # Default upright standing posture
    defaults = {
        "nose": (300, 150, 0.9),
        "left_eye": (290, 145, 0.9),
        "right_eye": (310, 145, 0.9),
        "left_ear": (280, 150, 0.9),
        "right_ear": (320, 150, 0.9),
        "left_shoulder": (270, 200, 0.9),
        "right_shoulder": (330, 200, 0.9),
        "left_elbow": (260, 270, 0.9),
        "right_elbow": (340, 270, 0.9),
        "left_wrist": (255, 340, 0.9),
        "right_wrist": (345, 340, 0.9),
        "left_hip": (280, 350, 0.9),
        "right_hip": (320, 350, 0.9),
        "left_knee": (275, 420, 0.9),
        "right_knee": (325, 420, 0.9),
        "left_ankle": (270, 470, 0.9),
        "right_ankle": (330, 470, 0.9),
    }
    defaults.update(overrides)
    return defaults


@pytest.fixture
def person_frame_standing():
    """PersonFrame with upright standing posture — arms at sides, elbows nearly straight."""
    kps = _make_keypoints()
    return PersonFrame(
        track_id=1,
        bbox=[240.0, 130.0, 360.0, 480.0],
        keypoints=kps,
        centroid=(300.0, 300.0),
        frame_index=0,
        timestamp_sec=0.0,
    )


@pytest.fixture
def person_frame_reaching():
    """PersonFrame with reaching posture — one wrist extended, elbow ~90 degrees."""
    kps = _make_keypoints(
        left_elbow=(240, 230, 0.9),      # Bent elbow
        left_wrist=(200, 230, 0.9),      # Wrist extended forward, below shoulder
        right_elbow=(340, 270, 0.9),
        right_wrist=(345, 340, 0.9),
    )
    return PersonFrame(
        track_id=1,
        bbox=[180.0, 130.0, 360.0, 480.0],
        keypoints=kps,
        centroid=(270.0, 300.0),
        frame_index=10,
        timestamp_sec=0.33,
    )


@pytest.fixture
def person_frame_lifting():
    """PersonFrame with lifting posture — wrists above shoulders, trunk leaned forward."""
    kps = _make_keypoints(
        left_shoulder=(270, 220, 0.9),
        right_shoulder=(330, 220, 0.9),
        left_elbow=(260, 180, 0.9),      # Arms raised
        right_elbow=(340, 180, 0.9),
        left_wrist=(255, 150, 0.9),      # Wrists above shoulders
        right_wrist=(345, 150, 0.9),
        left_hip=(280, 350, 0.9),
        right_hip=(320, 350, 0.9),
        # Trunk leaned forward: shoulders shifted forward relative to hips
        nose=(320, 170, 0.9),
    )
    # Move shoulders forward to create trunk flexion
    kps["left_shoulder"] = (260, 230, 0.9)
    kps["right_shoulder"] = (320, 230, 0.9)
    return PersonFrame(
        track_id=1,
        bbox=[230.0, 130.0, 370.0, 480.0],
        keypoints=kps,
        centroid=(300.0, 300.0),
        frame_index=20,
        timestamp_sec=0.67,
    )


@pytest.fixture
def person_frame_walking():
    """PersonFrame with walking posture — upright, arms slightly extended, high velocity."""
    kps = _make_keypoints(
        left_elbow=(240, 260, 0.9),
        right_elbow=(360, 260, 0.9),
        left_wrist=(220, 320, 0.9),
        right_wrist=(380, 320, 0.9),  # Arms extended
    )
    return PersonFrame(
        track_id=1,
        bbox=[200.0, 130.0, 400.0, 480.0],
        keypoints=kps,
        centroid=(300.0, 300.0),
        frame_index=30,
        timestamp_sec=1.0,
    )


@pytest.fixture
def person_frame_walking_prev():
    """Previous frame for walking person — shifted centroid for high velocity."""
    kps = _make_keypoints()
    return PersonFrame(
        track_id=1,
        bbox=[150.0, 130.0, 350.0, 480.0],
        keypoints=kps,
        centroid=(200.0, 300.0),  # 100px to the left of walking frame
        frame_index=29,
        timestamp_sec=0.967,
    )


@pytest.fixture
def mock_video_path(tmp_path):
    """Create a 3-second synthetic MP4 video (black frames, 30fps)."""
    video_path = str(tmp_path / "test_video.mp4")
    fps = 30
    duration = 3  # seconds
    width, height = 640, 480

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for i in range(fps * duration):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Add some variation so frames aren't identical
        cv2.putText(frame, f"Frame {i}", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        writer.write(frame)

    writer.release()
    return video_path


@pytest.fixture
def default_config():
    """Load config/default.yaml and return as dict."""
    config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_frame():
    """Create a sample BGR frame (640x480)."""
    return np.zeros((480, 640, 3), dtype=np.uint8)
