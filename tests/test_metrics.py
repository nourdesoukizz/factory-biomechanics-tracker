"""Tests for the metrics engine module."""

import json
import os

import pytest

from src.metrics import MetricsEngine
from src.models import PersonFrame, TaskEvent


@pytest.fixture
def engine():
    """Create a MetricsEngine with default settings."""
    return MetricsEngine(fps=30.0, frame_width=640, frame_height=480)


class TestActiveRatio:
    """Tests for active/idle ratio computation."""

    def test_active_ratio_all_active(self, engine):
        """active_ratio = 1.0 when all frames have high velocity."""
        for i in range(30):
            pf = PersonFrame(1, [0, 0, 100, 100], {}, (100 + i * 10, 200), i, i / 30.0)
            engine.update(pf, velocity=100.0, is_active=True,
                          joint_angles={}, rula_score=2.0, task_events=[])

        m = engine.build_metrics(1)
        assert m.active_ratio == 1.0

    def test_active_ratio_all_idle(self, engine):
        """active_ratio = 0.0 when all frames have zero velocity."""
        for i in range(30):
            pf = PersonFrame(1, [0, 0, 100, 100], {}, (300, 300), i, i / 30.0)
            engine.update(pf, velocity=0.0, is_active=False,
                          joint_angles={}, rula_score=1.0, task_events=[])

        m = engine.build_metrics(1)
        assert m.active_ratio == 0.0

    def test_active_ratio_mixed(self, engine):
        """active_ratio ~0.5 for half active, half idle."""
        for i in range(30):
            is_active = i < 15
            pf = PersonFrame(1, [0, 0, 100, 100], {}, (300, 300), i, i / 30.0)
            engine.update(pf, velocity=50.0 if is_active else 0.0,
                          is_active=is_active,
                          joint_angles={}, rula_score=2.0, task_events=[])

        m = engine.build_metrics(1)
        assert abs(m.active_ratio - 0.5) < 0.01


class TestTotalMovement:
    """Tests for total movement computation."""

    def test_total_movement_zero_stationary(self, engine):
        """total_movement_px = 0 when centroid never moves."""
        for i in range(10):
            pf = PersonFrame(1, [0, 0, 100, 100], {}, (300, 300), i, i / 30.0)
            engine.update(pf, velocity=0.0, is_active=False,
                          joint_angles={}, rula_score=1.0, task_events=[])

        m = engine.build_metrics(1)
        assert m.total_movement_px == 0.0

    def test_total_movement_known_displacement(self, engine):
        """Returns correct sum for known centroid trajectory."""
        # Move 10px right each frame for 10 frames = 90px total (9 displacements)
        for i in range(10):
            pf = PersonFrame(1, [0, 0, 100, 100], {}, (100 + i * 10, 200), i, i / 30.0)
            engine.update(pf, velocity=300.0, is_active=True,
                          joint_angles={}, rula_score=2.0, task_events=[])

        m = engine.build_metrics(1)
        assert abs(m.total_movement_px - 90.0) < 1.0


class TestZones:
    """Tests for zone assignment and dwell time."""

    def test_zone_assignment_correct(self, engine):
        """Centroid in top-left maps to zone A1."""
        zone = engine.get_zone_for_point(50, 50)
        assert zone == "A1"

    def test_zone_assignment_bottom_right(self, engine):
        """Centroid in bottom-right maps to zone C2."""
        zone = engine.get_zone_for_point(600, 400)
        assert zone == "C2"

    def test_zone_dwell_time_sums_to_total(self, engine):
        """Sum of all zone dwell times approximately equals total time seen."""
        for i in range(30):
            # Move across zones
            x = (i / 30.0) * 640
            pf = PersonFrame(1, [0, 0, 100, 100], {}, (x, 240), i, i / 30.0)
            engine.update(pf, velocity=50.0, is_active=True,
                          joint_angles={}, rula_score=2.0, task_events=[])

        m = engine.build_metrics(1)
        total_dwell = sum(m.zone_dwell_times.values())
        expected = 30 / 30.0  # 30 frames at 30 fps = 1 second
        assert abs(total_dwell - expected) < 0.1


class TestTaskCounts:
    """Tests for task count aggregation."""

    def test_task_counts_correct(self, engine):
        """task_counts matches number of TaskEvents per task type."""
        events = [
            TaskEvent(1, "pick_and_place", 0, 10, 0.8, 0.33),
            TaskEvent(1, "pick_and_place", 25, 35, 0.7, 0.33),
            TaskEvent(1, "lift_and_place", 50, 65, 0.9, 0.5),
        ]

        pf = PersonFrame(1, [0, 0, 100, 100], {}, (300, 300), 0, 0.0)
        engine.update(pf, velocity=50.0, is_active=True,
                      joint_angles={}, rula_score=2.0, task_events=events)

        m = engine.build_metrics(1)
        assert m.task_counts.get("pick_and_place", 0) == 2
        assert m.task_counts.get("lift_and_place", 0) == 1


class TestMetricsOutput:
    """Tests for metrics output fields and export."""

    def test_metrics_output_has_all_fields(self, engine):
        """PersonMetrics contains all required fields."""
        pf = PersonFrame(1, [0, 0, 100, 100], {}, (300, 300), 0, 0.0)
        engine.update(pf, velocity=50.0, is_active=True,
                      joint_angles={}, rula_score=3.0, task_events=[])

        m = engine.build_metrics(1)
        assert hasattr(m, "track_id")
        assert hasattr(m, "total_active_frames")
        assert hasattr(m, "total_idle_frames")
        assert hasattr(m, "active_ratio")
        assert hasattr(m, "total_movement_px")
        assert hasattr(m, "zone_dwell_times")
        assert hasattr(m, "task_counts")
        assert hasattr(m, "task_events")
        assert hasattr(m, "avg_rula_score")
        assert hasattr(m, "peak_rula_score")

    def test_multiple_persons_tracked_independently(self, engine):
        """Two persons with different track_ids produce separate metrics."""
        for i in range(10):
            pf1 = PersonFrame(1, [0, 0, 100, 100], {}, (100, 200), i, i / 30.0)
            pf2 = PersonFrame(2, [200, 0, 300, 100], {}, (400, 200), i, i / 30.0)
            engine.update(pf1, velocity=50.0, is_active=True,
                          joint_angles={}, rula_score=2.0, task_events=[])
            engine.update(pf2, velocity=0.0, is_active=False,
                          joint_angles={}, rula_score=1.0, task_events=[])

        m1 = engine.build_metrics(1)
        m2 = engine.build_metrics(2)
        assert m1.active_ratio == 1.0
        assert m2.active_ratio == 0.0
        assert m1.track_id != m2.track_id

    def test_export_csv(self, engine, tmp_path):
        """CSV export creates valid file."""
        for i in range(10):
            pf = PersonFrame(1, [0, 0, 100, 100], {}, (300, 300), i, i / 30.0)
            engine.update(pf, velocity=50.0, is_active=True,
                          joint_angles={}, rula_score=2.0, task_events=[])

        csv_path = str(tmp_path / "test_metrics.csv")
        engine.export_csv(csv_path)
        assert os.path.exists(csv_path)
        assert os.path.getsize(csv_path) > 0

    def test_export_json(self, engine, tmp_path):
        """JSON export creates valid file."""
        for i in range(10):
            pf = PersonFrame(1, [0, 0, 100, 100], {}, (300, 300), i, i / 30.0)
            engine.update(pf, velocity=50.0, is_active=True,
                          joint_angles={}, rula_score=2.0, task_events=[])

        json_path = str(tmp_path / "test_metrics.json")
        engine.export_json(json_path)
        assert os.path.exists(json_path)

        with open(json_path) as f:
            data = json.load(f)
        assert "persons" in data
        assert len(data["persons"]) == 1
