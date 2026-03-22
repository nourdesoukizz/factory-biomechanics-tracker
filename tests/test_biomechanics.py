"""Tests for the biomechanics analysis module."""

import numpy as np
import pytest

from src.biomechanics import BiomechanicsAnalyzer
from src.models import PersonFrame


@pytest.fixture
def analyzer():
    """Create a BiomechanicsAnalyzer with default settings."""
    return BiomechanicsAnalyzer()


class TestJointAngles:
    """Tests for joint angle computation."""

    def test_elbow_angle_straight_arm(self, analyzer):
        """Nearly straight arm should produce ~170+ degree elbow angle."""
        # Straight arm: shoulder, elbow, wrist roughly in a line
        kps = {
            "left_shoulder": (200, 200, 0.9),
            "left_elbow": (200, 300, 0.9),
            "left_wrist": (200, 400, 0.9),
            "right_shoulder": (400, 200, 0.9),
            "right_elbow": (400, 300, 0.9),
            "right_wrist": (400, 400, 0.9),
            "left_hip": (250, 400, 0.9),
            "right_hip": (350, 400, 0.9),
        }
        angles = analyzer.compute_joint_angles(kps)
        assert angles["elbow_left"] > 160

    def test_elbow_angle_bent_arm(self, analyzer):
        """Bent arm at ~90 degrees."""
        kps = {
            "left_shoulder": (200, 200, 0.9),
            "left_elbow": (200, 300, 0.9),
            "left_wrist": (300, 300, 0.9),  # 90-degree bend
            "right_shoulder": (400, 200, 0.9),
            "right_elbow": (400, 300, 0.9),
            "right_wrist": (400, 400, 0.9),
            "left_hip": (250, 400, 0.9),
            "right_hip": (350, 400, 0.9),
        }
        angles = analyzer.compute_joint_angles(kps)
        assert 80 <= angles["elbow_left"] <= 100

    def test_trunk_flex_upright(self, analyzer):
        """Upright spine should have ~0 degree trunk flexion."""
        kps = {
            "left_shoulder": (270, 200, 0.9),
            "right_shoulder": (330, 200, 0.9),
            "left_hip": (280, 400, 0.9),
            "right_hip": (320, 400, 0.9),
            "left_elbow": (260, 280, 0.9),
            "right_elbow": (340, 280, 0.9),
            "left_wrist": (255, 350, 0.9),
            "right_wrist": (345, 350, 0.9),
        }
        angles = analyzer.compute_joint_angles(kps)
        assert angles["trunk_flex"] < 10

    def test_trunk_flex_leaning(self, analyzer):
        """Forward-leaning torso should have >20 degree trunk flexion."""
        kps = {
            "left_shoulder": (200, 220, 0.9),   # Shifted forward
            "right_shoulder": (260, 220, 0.9),
            "left_hip": (280, 400, 0.9),
            "right_hip": (320, 400, 0.9),
            "left_elbow": (190, 290, 0.9),
            "right_elbow": (270, 290, 0.9),
            "left_wrist": (185, 350, 0.9),
            "right_wrist": (265, 350, 0.9),
        }
        angles = analyzer.compute_joint_angles(kps)
        assert angles["trunk_flex"] > 20

    def test_wrist_height_above_shoulder(self, analyzer):
        """Wrist above shoulder returns positive value."""
        kps = {
            "left_shoulder": (270, 300, 0.9),
            "left_wrist": (270, 200, 0.9),  # Above shoulder (lower y)
            "left_hip": (280, 450, 0.9),
            "right_shoulder": (330, 300, 0.9),
            "right_wrist": (330, 350, 0.9),
            "right_hip": (320, 450, 0.9),
            "left_elbow": (270, 250, 0.9),
            "right_elbow": (330, 320, 0.9),
        }
        angles = analyzer.compute_joint_angles(kps)
        assert angles["wrist_height_left"] > 0

    def test_wrist_height_below_shoulder(self, analyzer):
        """Wrist below shoulder returns negative value."""
        kps = {
            "left_shoulder": (270, 200, 0.9),
            "left_wrist": (270, 400, 0.9),  # Below shoulder (higher y)
            "left_hip": (280, 350, 0.9),
            "right_shoulder": (330, 200, 0.9),
            "right_wrist": (330, 400, 0.9),
            "right_hip": (320, 350, 0.9),
            "left_elbow": (270, 300, 0.9),
            "right_elbow": (330, 300, 0.9),
        }
        angles = analyzer.compute_joint_angles(kps)
        assert angles["wrist_height_left"] < 0

    def test_compute_joint_angles_returns_all_keys(self, analyzer, person_frame_standing):
        """Output dict contains all expected angle keys."""
        angles = analyzer.compute_joint_angles(person_frame_standing.keypoints)
        expected_keys = [
            "elbow_left", "elbow_right",
            "shoulder_left", "shoulder_right",
            "trunk_flex",
            "wrist_height_left", "wrist_height_right",
        ]
        for key in expected_keys:
            assert key in angles

    def test_rula_missing_keypoints(self, analyzer):
        """Handles low-confidence keypoints gracefully."""
        kps = {
            "left_shoulder": (270, 200, 0.1),  # Low confidence
            "right_shoulder": (330, 200, 0.1),
            "left_elbow": (260, 280, 0.1),
            "right_elbow": (340, 280, 0.1),
            "left_wrist": (255, 350, 0.1),
            "right_wrist": (345, 350, 0.1),
            "left_hip": (280, 350, 0.1),
            "right_hip": (320, 350, 0.1),
        }
        angles = analyzer.compute_joint_angles(kps)
        score, label = analyzer.compute_rula_score(angles)
        assert 1 <= score <= 7


class TestVelocity:
    """Tests for velocity computation."""

    def test_velocity_stationary(self, analyzer):
        """Returns ~0 when centroid unchanged between frames."""
        frame1 = PersonFrame(1, [0, 0, 100, 100], {}, (300, 300), 0, 0.0)
        frame2 = PersonFrame(1, [0, 0, 100, 100], {}, (300, 300), 1, 0.033)
        vel = analyzer.compute_velocity(frame2, frame1, 30.0)
        assert vel < 1.0

    def test_velocity_moving(self, analyzer):
        """Returns correct px/sec for known displacement."""
        frame1 = PersonFrame(1, [0, 0, 100, 100], {}, (100, 100), 0, 0.0)
        frame2 = PersonFrame(1, [0, 0, 100, 100], {}, (200, 100), 1, 0.033)
        vel = analyzer.compute_velocity(frame2, frame1, 30.0)
        # 100 pixels in 1/30 second = 3000 px/sec
        assert abs(vel - 3000.0) < 1.0

    def test_velocity_no_previous(self, analyzer):
        """Returns 0 when no previous frame."""
        frame1 = PersonFrame(1, [0, 0, 100, 100], {}, (300, 300), 0, 0.0)
        vel = analyzer.compute_velocity(frame1, None, 30.0)
        assert vel == 0.0


class TestRULA:
    """Tests for RULA score computation."""

    def test_rula_low_score_upright(self, analyzer, person_frame_standing):
        """Standing upright posture scores LOW (1-2)."""
        angles = analyzer.compute_joint_angles(person_frame_standing.keypoints)
        score, label = analyzer.compute_rula_score(angles)
        assert score <= 3
        assert label in ("LOW", "MEDIUM")

    def test_rula_high_score_lifting(self, analyzer, person_frame_lifting):
        """Overhead reach posture should score higher."""
        angles = analyzer.compute_joint_angles(person_frame_lifting.keypoints)
        score, label = analyzer.compute_rula_score(angles)
        assert score >= 3

    def test_rula_score_range(self, analyzer, person_frame_standing):
        """RULA score is always between 1 and 7."""
        angles = analyzer.compute_joint_angles(person_frame_standing.keypoints)
        score, label = analyzer.compute_rula_score(angles)
        assert 1 <= score <= 7
        assert label in ("LOW", "MEDIUM", "HIGH", "VERY_HIGH")
