"""Video ingestion module — reads video files and exposes a clean frame iterator."""

import logging
from pathlib import Path
from typing import Generator, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoReader:
    """Wraps cv2.VideoCapture to provide validated video reading with metadata."""

    def __init__(self, video_path: str, skip_frames: int = 0, max_frames: Optional[int] = None) -> None:
        """Initialize VideoReader with path validation and metadata extraction.

        Args:
            video_path: Path to the input video file.
            skip_frames: Process every Nth frame (0 = all frames).
            max_frames: Maximum number of frames to yield (None = all).
        """
        self._cap = None
        self._path = Path(video_path)
        if not self._path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self._cap = cv2.VideoCapture(str(self._path))
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        self._skip_frames = skip_frames
        self._max_frames = max_frames

        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._duration_sec = self._total_frames / self._fps if self._fps > 0 else 0.0

        logger.info(
            "Loaded video: %s | %dx%d | %.1f fps | %d frames | %.1f sec",
            self._path.name, self._width, self._height,
            self._fps, self._total_frames, self._duration_sec
        )

    @property
    def fps(self) -> float:
        """Frames per second of the video."""
        return self._fps

    @property
    def total_frames(self) -> int:
        """Total number of frames in the video."""
        return self._total_frames

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        return self._width

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        return self._height

    @property
    def duration_sec(self) -> float:
        """Video duration in seconds."""
        return self._duration_sec

    def iter_frames(self) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """Yield (frame_index, timestamp_sec, frame_bgr) tuples.

        Respects skip_frames and max_frames settings.
        """
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_index = 0
        yielded_count = 0
        step = self._skip_frames + 1 if self._skip_frames > 0 else 1

        while True:
            if self._max_frames is not None and yielded_count >= self._max_frames:
                break

            ret, frame = self._cap.read()
            if not ret:
                break

            if frame_index % step == 0:
                timestamp_sec = frame_index / self._fps if self._fps > 0 else 0.0
                yield frame_index, timestamp_sec, frame
                yielded_count += 1

            frame_index += 1

        logger.info("Finished reading video: %d frames yielded", yielded_count)

    def release(self) -> None:
        """Release the video capture resource."""
        if self._cap is not None:
            self._cap.release()

    def __del__(self) -> None:
        """Ensure capture is released on garbage collection."""
        self.release()
