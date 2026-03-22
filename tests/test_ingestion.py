"""Tests for the video ingestion module."""

import pytest

from src.ingestion import VideoReader


class TestVideoReader:
    """Tests for VideoReader class."""

    def test_video_reader_opens_valid_file(self, mock_video_path):
        """VideoReader initializes without error on a valid video file."""
        reader = VideoReader(mock_video_path)
        assert reader is not None
        reader.release()

    def test_video_reader_metadata(self, mock_video_path):
        """Video metadata properties are correct."""
        reader = VideoReader(mock_video_path)
        assert reader.fps == 30.0
        assert reader.width == 640
        assert reader.height == 480
        assert reader.total_frames == 90  # 3 seconds * 30 fps
        assert abs(reader.duration_sec - 3.0) < 0.1
        reader.release()

    def test_iter_frames_yields_correct_count(self, mock_video_path):
        """Number of yielded frames matches total_frames."""
        reader = VideoReader(mock_video_path)
        frames = list(reader.iter_frames())
        assert len(frames) == reader.total_frames
        reader.release()

    def test_iter_frames_timestamp_increases(self, mock_video_path):
        """Each frame timestamp is greater than the previous."""
        reader = VideoReader(mock_video_path)
        prev_ts = -1.0
        for _, ts, _ in reader.iter_frames():
            assert ts > prev_ts
            prev_ts = ts
        reader.release()

    def test_video_reader_raises_on_missing_file(self):
        """FileNotFoundError raised for non-existent file."""
        with pytest.raises(FileNotFoundError):
            VideoReader("/nonexistent/video.mp4")

    def test_skip_frames_reduces_output(self, mock_video_path):
        """skip_frames=2 yields approximately 1/3 of total frames."""
        reader = VideoReader(mock_video_path, skip_frames=2)
        frames = list(reader.iter_frames())
        # With skip_frames=2, we process every 3rd frame: 0, 3, 6, ...
        expected = (reader.total_frames + 2) // 3
        assert len(frames) == expected
        reader.release()

    def test_max_frames_limits_output(self, mock_video_path):
        """max_frames caps the number of yielded frames."""
        reader = VideoReader(mock_video_path, max_frames=10)
        frames = list(reader.iter_frames())
        assert len(frames) == 10
        reader.release()

    def test_frame_shape(self, mock_video_path):
        """Frames have correct shape (height, width, 3)."""
        reader = VideoReader(mock_video_path)
        for _, _, frame in reader.iter_frames():
            assert frame.shape == (480, 640, 3)
            break  # Only check first frame
        reader.release()
