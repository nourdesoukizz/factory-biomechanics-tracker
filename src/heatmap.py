"""Heatmap generation module — spatial presence heatmaps per person and combined."""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class HeatmapGenerator:
    """Generates spatial heatmaps from centroid position history."""

    def __init__(self, frame_width: int, frame_height: int,
                 bin_size: int = 10) -> None:
        """Initialize heatmap generator.

        Args:
            frame_width: Video frame width in pixels.
            frame_height: Video frame height in pixels.
            bin_size: Size of histogram bins in pixels.
        """
        self._width = frame_width
        self._height = frame_height
        self._bin_size = bin_size

        # Per-person centroid accumulators
        self._centroids: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        self._background_frame: Optional[np.ndarray] = None

    def set_background(self, frame_bgr: np.ndarray) -> None:
        """Set the background frame for heatmap overlay.

        Args:
            frame_bgr: First video frame (BGR) for grayscale background.
        """
        self._background_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    def accumulate(self, track_id: int, centroid: Tuple[float, float]) -> None:
        """Add a centroid position to the accumulator.

        Args:
            track_id: Person track ID.
            centroid: (x, y) position in pixels.
        """
        self._centroids[track_id].append(centroid)

    def _create_heatmap(self, centroids: List[Tuple[float, float]]) -> np.ndarray:
        """Create a 2D histogram heatmap from centroid positions.

        Args:
            centroids: List of (x, y) positions.

        Returns:
            2D numpy array of the heatmap.
        """
        if not centroids:
            return np.zeros((self._height // self._bin_size, self._width // self._bin_size))

        xs = [c[0] for c in centroids]
        ys = [c[1] for c in centroids]

        bins_x = max(1, self._width // self._bin_size)
        bins_y = max(1, self._height // self._bin_size)

        heatmap, _, _ = np.histogram2d(
            ys, xs,
            bins=[bins_y, bins_x],
            range=[[0, self._height], [0, self._width]]
        )

        return heatmap

    def _render_heatmap_image(self, heatmap: np.ndarray, title: str) -> np.ndarray:
        """Render a heatmap as an image overlaid on the background.

        Args:
            heatmap: 2D histogram array.
            title: Title for the plot.

        Returns:
            RGB image as numpy array.
        """
        fig, ax = plt.subplots(1, 1, figsize=(self._width / 100, self._height / 100), dpi=100)

        # Background
        if self._background_frame is not None:
            ax.imshow(self._background_frame, cmap="gray", alpha=0.5,
                      extent=[0, self._width, self._height, 0])

        # Heatmap overlay
        heatmap_resized = cv2.resize(
            heatmap.astype(np.float32),
            (self._width, self._height),
            interpolation=cv2.INTER_LINEAR
        )

        if heatmap_resized.max() > 0:
            heatmap_resized = heatmap_resized / heatmap_resized.max()

        ax.imshow(heatmap_resized, cmap="hot", alpha=0.6,
                  extent=[0, self._width, self._height, 0])

        ax.set_title(title, fontsize=12, color="white",
                     bbox=dict(boxstyle="round", facecolor="black", alpha=0.7))
        ax.axis("off")

        fig.tight_layout(pad=0)
        fig.canvas.draw()

        # Convert to numpy array
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3]  # Drop alpha channel
        plt.close(fig)

        return img

    def save_person_heatmap(self, track_id: int, output_path: str) -> None:
        """Save heatmap for a single person.

        Args:
            track_id: Person track ID.
            output_path: Path to save PNG.
        """
        centroids = self._centroids.get(track_id, [])
        if not centroids:
            logger.warning("No centroids for person %d, skipping heatmap", track_id)
            return

        heatmap = self._create_heatmap(centroids)
        img = self._render_heatmap_image(heatmap, f"Person {track_id} — Spatial Presence")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            logger.info("Saved heatmap: %s", output_path)
        except Exception as e:
            logger.error("Failed to save heatmap %s: %s", output_path, e)
            raise

    def save_combined_heatmap(self, output_path: str) -> None:
        """Save combined heatmap for all persons.

        Args:
            output_path: Path to save PNG.
        """
        all_centroids: List[Tuple[float, float]] = []
        for centroids in self._centroids.values():
            all_centroids.extend(centroids)

        if not all_centroids:
            logger.warning("No centroids accumulated, skipping combined heatmap")
            return

        heatmap = self._create_heatmap(all_centroids)
        img = self._render_heatmap_image(heatmap, "Combined — All Workers Spatial Presence")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            logger.info("Saved combined heatmap: %s", output_path)
        except Exception as e:
            logger.error("Failed to save combined heatmap %s: %s", output_path, e)
            raise

    def get_all_track_ids(self) -> List[int]:
        """Return all tracked person IDs with centroid data."""
        return list(self._centroids.keys())
