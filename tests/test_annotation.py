"""Tests for the frame annotation module."""

from collections import deque

import numpy as np
import pytest

from src.annotation import FrameAnnotator, TRACK_COLORS
from src.models import PersonFrame


@pytest.fixture
def annotator():
    """Create a FrameAnnotator for 640x480 frames."""
    return FrameAnnotator(
        frame_width=640, frame_height=480,
        grid_cols=3, grid_rows=2,
        zone_labels=[["A1", "B1", "C1"], ["A2", "B2", "C2"]],
    )


class TestFrameAnnotation:
    """Tests for frame annotation."""

    def test_annotate_frame_returns_same_shape(self, annotator, sample_frame, person_frame_standing):
        """Output frame shape matches input frame shape."""
        result = annotator.annotate_frame(
            sample_frame, [person_frame_standing],
            {1: ("idle", 0.8)}, {1: (2.0, "LOW")},
            {1: deque([(300, 300)])},
            frame_index=0, timestamp_sec=0.0,
        )
        assert result.shape == sample_frame.shape

    def test_annotate_frame_modifies_pixels(self, annotator, sample_frame, person_frame_standing):
        """Output frame differs from input (something was drawn)."""
        original = sample_frame.copy()
        result = annotator.annotate_frame(
            sample_frame, [person_frame_standing],
            {1: ("idle", 0.8)}, {1: (2.0, "LOW")},
            {1: deque([(300, 300)])},
            frame_index=0, timestamp_sec=0.0,
        )
        assert not np.array_equal(result, original)

    def test_no_crash_on_missing_keypoints(self, annotator, sample_frame):
        """Annotator handles None/empty keypoints without error."""
        person = PersonFrame(
            track_id=1, bbox=[100, 100, 200, 300],
            keypoints={},  # Empty keypoints
            centroid=(150, 200), frame_index=0, timestamp_sec=0.0,
        )
        result = annotator.annotate_frame(
            sample_frame, [person],
            {1: ("idle", 0.5)}, {1: (1.0, "LOW")},
            {1: deque([(150, 200)])},
        )
        assert result.shape == sample_frame.shape

    def test_no_crash_on_single_person(self, annotator, sample_frame, person_frame_standing):
        """Works correctly with exactly one tracked person."""
        result = annotator.annotate_frame(
            sample_frame, [person_frame_standing],
            {1: ("pick_and_place", 0.7)}, {1: (3.0, "MEDIUM")},
            {1: deque([(300, 300)])},
        )
        assert result is not None

    def test_no_crash_on_zero_persons(self, annotator, sample_frame):
        """Works correctly with empty person list."""
        result = annotator.annotate_frame(
            sample_frame, [], {}, {}, {},
        )
        assert result.shape == sample_frame.shape

    def test_track_id_color_cycles(self, annotator):
        """Different track IDs get different colors."""
        color_0 = annotator._get_track_color(0)
        color_1 = annotator._get_track_color(1)
        color_2 = annotator._get_track_color(2)
        assert color_0 != color_1
        assert color_1 != color_2
        # Verify cycling
        color_wrap = annotator._get_track_color(len(TRACK_COLORS))
        assert color_wrap == color_0
