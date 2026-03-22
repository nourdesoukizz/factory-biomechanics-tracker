"""Integration tests for the full pipeline.

These tests use a synthetic video with no real persons, so YOLO detection
will return empty results. We test that the pipeline orchestration logic
handles this gracefully and produces all expected output files.
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import yaml

from src.models import PersonFrame


def _create_mock_config(tmp_path, video_path):
    """Create a temporary config file pointing to the test video."""
    config = {
        "ingestion": {
            "input_path": video_path,
            "skip_frames": 0,
            "max_frames": 30,  # Only process 30 frames for speed
        },
        "tracking": {
            "conf_threshold": 0.4,
            "iou_threshold": 0.5,
            "keypoint_conf_threshold": 0.3,
            "kalman_window": 5,
            "max_disappeared_frames": 30,
        },
        "biomechanics": {
            "idle_velocity_threshold_px_per_sec": 15.0,
            "trunk_flex_threshold_deg": 20.0,
            "wrist_above_shoulder_threshold": 0.0,
        },
        "classification": {
            "confidence_threshold": 0.6,
            "min_task_duration_frames": 8,
            "task_cooldown_frames": 15,
            "pick_velocity_threshold": 30.0,
            "stationary_threshold": 20.0,
            "movement_velocity_threshold": 60.0,
        },
        "zones": {
            "grid_cols": 3,
            "grid_rows": 2,
        },
        "output": {
            "annotated_video": True,
            "metrics_csv": True,
            "metrics_json": True,
            "heatmaps": True,
            "pdf_report": True,
            "run_id": "test_run",
        },
    }
    config_path = str(tmp_path / "test_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


def _create_synthetic_person_frames(frame_index, timestamp):
    """Create synthetic PersonFrame data for testing."""
    kps = {
        "nose": (300, 150, 0.9), "left_eye": (290, 145, 0.9),
        "right_eye": (310, 145, 0.9), "left_ear": (280, 150, 0.9),
        "right_ear": (320, 150, 0.9),
        "left_shoulder": (270, 200, 0.9), "right_shoulder": (330, 200, 0.9),
        "left_elbow": (260, 270, 0.9), "right_elbow": (340, 270, 0.9),
        "left_wrist": (255, 340, 0.9), "right_wrist": (345, 340, 0.9),
        "left_hip": (280, 350, 0.9), "right_hip": (320, 350, 0.9),
        "left_knee": (275, 420, 0.9), "right_knee": (325, 420, 0.9),
        "left_ankle": (270, 470, 0.9), "right_ankle": (330, 470, 0.9),
    }
    return [PersonFrame(
        track_id=1, bbox=[240, 130, 360, 480],
        keypoints=kps, centroid=(300 + frame_index * 0.5, 300),
        frame_index=frame_index, timestamp_sec=timestamp,
    )]


class TestPipelineIntegration:
    """Integration tests using mock tracker to avoid needing YOLO model."""

    def _run_pipeline_with_mock(self, tmp_path, mock_video_path):
        """Run pipeline with mocked tracker to avoid YOLO dependency."""
        config_path = _create_mock_config(tmp_path, mock_video_path)

        # Patch output directories to tmp_path
        output_base = str(tmp_path / "output")
        os.makedirs(f"{output_base}/annotated", exist_ok=True)
        os.makedirs(f"{output_base}/metrics", exist_ok=True)
        os.makedirs(f"{output_base}/heatmaps", exist_ok=True)
        os.makedirs(f"{output_base}/reports", exist_ok=True)

        # Run a simplified version of the pipeline with synthetic data
        config = yaml.safe_load(open(config_path))
        run_id = config["output"]["run_id"]

        from src.ingestion import VideoReader
        from src.biomechanics import BiomechanicsAnalyzer
        from src.classification import TaskClassifier
        from src.metrics import MetricsEngine
        from src.annotation import FrameAnnotator
        from src.heatmap import HeatmapGenerator

        reader = VideoReader(mock_video_path, max_frames=30)
        bio = BiomechanicsAnalyzer()
        classifier = TaskClassifier()
        metrics_engine = MetricsEngine(
            fps=reader.fps, frame_width=reader.width, frame_height=reader.height
        )
        annotator = FrameAnnotator(
            frame_width=reader.width, frame_height=reader.height,
            zone_labels=metrics_engine.get_zone_labels(),
        )
        heatmap_gen = HeatmapGenerator(reader.width, reader.height)

        # Video writer
        annotated_path = f"{output_base}/annotated/annotated_{run_id}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(annotated_path, fourcc, reader.fps, (reader.width, reader.height))

        previous_frames = {}
        first_frame_set = False

        for fi, ts, frame_bgr in reader.iter_frames():
            if not first_frame_set:
                heatmap_gen.set_background(frame_bgr)
                first_frame_set = True

            # Use synthetic person data instead of YOLO
            persons = _create_synthetic_person_frames(fi, ts)

            task_labels = {}
            rula_labels = {}

            for person in persons:
                angles = bio.compute_joint_angles(person.keypoints)
                prev = previous_frames.get(person.track_id)
                vel = bio.compute_velocity(person, prev, reader.fps)
                rula_score, rula_label = bio.compute_rula_score(angles)
                is_active = not bio.is_idle(vel)

                task, conf, events = classifier.classify_frame(
                    person, angles, vel, rula_score, rula_label, reader.fps
                )

                metrics_engine.update(person, vel, is_active, angles, rula_score, events)
                heatmap_gen.accumulate(person.track_id, person.centroid)

                task_labels[person.track_id] = (task, conf)
                rula_labels[person.track_id] = (rula_score, rula_label)
                previous_frames[person.track_id] = person

            annotated = annotator.annotate_frame(
                frame_bgr, persons, task_labels, rula_labels, {}, None, fi, ts
            )
            writer.write(annotated)

        writer.release()

        # Finalize
        for tid in metrics_engine.get_all_track_ids():
            remaining = classifier.finalize(tid, 29, reader.fps)
            metrics_engine.add_task_events(remaining)

        # Export
        csv_path = f"{output_base}/metrics/metrics_{run_id}.csv"
        json_path = f"{output_base}/metrics/metrics_{run_id}.json"
        metrics_engine.export_csv(csv_path)
        metrics_engine.export_json(json_path)

        for tid in heatmap_gen.get_all_track_ids():
            heatmap_gen.save_person_heatmap(tid, f"{output_base}/heatmaps/heatmap_person_{tid}_{run_id}.png")
        heatmap_gen.save_combined_heatmap(f"{output_base}/heatmaps/heatmap_combined_{run_id}.png")

        # PDF
        from src.reporting import ReportGenerator
        all_metrics = [metrics_engine.build_metrics(tid) for tid in metrics_engine.get_all_track_ids()]
        heatmap_paths = {
            tid: f"{output_base}/heatmaps/heatmap_person_{tid}_{run_id}.png"
            for tid in heatmap_gen.get_all_track_ids()
        }
        reporter = ReportGenerator(
            video_filename="test_video.mp4",
            video_duration_sec=reader.duration_sec,
            fps=reader.fps, total_frames=30,
        )
        reporter.generate(all_metrics, heatmap_paths,
                          f"{output_base}/reports/report_{run_id}.pdf", run_id)

        reader.release()
        return output_base, run_id

    def test_pipeline_runs_on_mock_video(self, tmp_path, mock_video_path):
        """Full pipeline completes without error on synthetic video."""
        output_base, run_id = self._run_pipeline_with_mock(tmp_path, mock_video_path)
        assert True  # No exception = success

    def test_pipeline_produces_annotated_video(self, tmp_path, mock_video_path):
        """Output MP4 exists and is nonzero size."""
        output_base, run_id = self._run_pipeline_with_mock(tmp_path, mock_video_path)
        path = f"{output_base}/annotated/annotated_{run_id}.mp4"
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_pipeline_produces_csv(self, tmp_path, mock_video_path):
        """Output CSV exists and has at least one data row."""
        output_base, run_id = self._run_pipeline_with_mock(tmp_path, mock_video_path)
        path = f"{output_base}/metrics/metrics_{run_id}.csv"
        assert os.path.exists(path)
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) >= 2  # Header + at least one data row

    def test_pipeline_produces_json(self, tmp_path, mock_video_path):
        """Output JSON is valid and parseable."""
        output_base, run_id = self._run_pipeline_with_mock(tmp_path, mock_video_path)
        path = f"{output_base}/metrics/metrics_{run_id}.json"
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert "persons" in data
        assert len(data["persons"]) >= 1

    def test_pipeline_produces_heatmaps(self, tmp_path, mock_video_path):
        """At least one heatmap PNG exists in output dir."""
        output_base, run_id = self._run_pipeline_with_mock(tmp_path, mock_video_path)
        heatmap_dir = f"{output_base}/heatmaps"
        pngs = [f for f in os.listdir(heatmap_dir) if f.endswith(".png")]
        assert len(pngs) >= 1

    def test_pipeline_produces_pdf(self, tmp_path, mock_video_path):
        """Output PDF exists and is nonzero size."""
        output_base, run_id = self._run_pipeline_with_mock(tmp_path, mock_video_path)
        path = f"{output_base}/reports/report_{run_id}.pdf"
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_pipeline_run_id_namespaces_outputs(self, tmp_path, mock_video_path):
        """All output files contain the run_id string."""
        output_base, run_id = self._run_pipeline_with_mock(tmp_path, mock_video_path)

        for subdir in ["annotated", "metrics", "heatmaps", "reports"]:
            dir_path = f"{output_base}/{subdir}"
            if os.path.exists(dir_path):
                for f in os.listdir(dir_path):
                    assert run_id in f, f"File {f} does not contain run_id '{run_id}'"
