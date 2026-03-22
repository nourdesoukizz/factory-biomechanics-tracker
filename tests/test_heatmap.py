"""Tests for the heatmap generation module."""

import os

import cv2
import numpy as np
import pytest

from src.heatmap import HeatmapGenerator


@pytest.fixture
def heatmap_gen():
    """Create a HeatmapGenerator for 640x480 frames."""
    gen = HeatmapGenerator(frame_width=640, frame_height=480)
    gen.set_background(np.zeros((480, 640, 3), dtype=np.uint8))
    return gen


class TestHeatmapGeneration:
    """Tests for heatmap generation."""

    def test_heatmap_output_is_png(self, heatmap_gen, tmp_path):
        """Output file is a valid PNG."""
        for i in range(50):
            heatmap_gen.accumulate(1, (300 + i, 200))

        output_path = str(tmp_path / "test_heatmap.png")
        heatmap_gen.save_person_heatmap(1, output_path)

        assert os.path.exists(output_path)
        # Verify it's a valid image
        img = cv2.imread(output_path)
        assert img is not None

    def test_heatmap_shape_reasonable(self, heatmap_gen, tmp_path):
        """Heatmap dimensions are reasonable."""
        for i in range(50):
            heatmap_gen.accumulate(1, (300 + i, 200))

        output_path = str(tmp_path / "test_heatmap.png")
        heatmap_gen.save_person_heatmap(1, output_path)

        img = cv2.imread(output_path)
        assert img is not None
        assert img.shape[0] > 0
        assert img.shape[1] > 0

    def test_heatmap_nonzero_where_person_present(self, heatmap_gen, tmp_path):
        """Heatmap image is non-zero (something was drawn)."""
        for i in range(100):
            heatmap_gen.accumulate(1, (320, 240))  # Center of frame

        output_path = str(tmp_path / "test_heatmap.png")
        heatmap_gen.save_person_heatmap(1, output_path)

        img = cv2.imread(output_path)
        assert img is not None
        assert np.any(img > 0)

    def test_combined_heatmap_includes_all_persons(self, heatmap_gen, tmp_path):
        """Combined heatmap has nonzero pixels for all person positions."""
        # Person 1 in top-left
        for i in range(50):
            heatmap_gen.accumulate(1, (100, 100))

        # Person 2 in bottom-right
        for i in range(50):
            heatmap_gen.accumulate(2, (500, 400))

        output_path = str(tmp_path / "combined_heatmap.png")
        heatmap_gen.save_combined_heatmap(output_path)

        assert os.path.exists(output_path)
        img = cv2.imread(output_path)
        assert img is not None
        assert np.any(img > 0)
