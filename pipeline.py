"""Main pipeline orchestrator — runs the full factory biomechanics tracker pipeline."""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger("pipeline")


def setup_logging() -> None:
    """Configure logging for the pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Configuration dictionary.
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info("Loaded configuration from %s", config_path)
        return config
    except Exception as e:
        logger.error("Failed to load config: %s", e)
        raise


def run_pipeline(config_path: str, input_path: Optional[str] = None,
                 extract_features: bool = False, train_model: bool = False) -> None:
    """Run the full biomechanics tracker pipeline.

    Args:
        config_path: Path to YAML config file.
        input_path: Optional override for video input path.
        extract_features: If True, save per-frame features for training.
        train_model: If True, train the learned classifier before running.
    """
    config = load_config(config_path)

    # Generate run ID
    run_id = config.get("output", {}).get("run_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("Pipeline run ID: %s", run_id)

    # === Train learned classifier if requested ===
    if train_model:
        _train_learned_classifier(config, run_id)
        if not input_path and not extract_features:
            logger.info("Training complete. Re-run without --train to use the learned model.")

    # Resolve input video path
    video_path = input_path or config.get("ingestion", {}).get("input_path", "data/input/video.mp4")
    logger.info("Input video: %s", video_path)

    # === Phase 1: Ingestion ===
    from src.ingestion import VideoReader

    skip_frames = config.get("ingestion", {}).get("skip_frames", 0)
    max_frames = config.get("ingestion", {}).get("max_frames")

    reader = VideoReader(video_path, skip_frames=skip_frames, max_frames=max_frames)

    # === Phase 2: Detection & Tracking ===
    from src.tracking import PersonTracker

    tracking_cfg = config.get("tracking", {})
    tracker = PersonTracker(
        conf_threshold=tracking_cfg.get("conf_threshold", 0.4),
        iou_threshold=tracking_cfg.get("iou_threshold", 0.5),
        keypoint_conf_threshold=tracking_cfg.get("keypoint_conf_threshold", 0.3),
        kalman_window=tracking_cfg.get("kalman_window", 5),
        max_disappeared_frames=tracking_cfg.get("max_disappeared_frames", 90),
        device="mps",
        reid_enabled=tracking_cfg.get("reid_enabled", False),
        reid_appearance_weight=tracking_cfg.get("reid_appearance_weight", 0.4),
        reid_spatial_weight=tracking_cfg.get("reid_spatial_weight", 0.3),
        reid_temporal_weight=tracking_cfg.get("reid_temporal_weight", 0.3),
        reid_merge_threshold=tracking_cfg.get("reid_merge_threshold", 0.6),
        reid_max_gap_frames=tracking_cfg.get("reid_max_gap_frames", 150),
    )

    # === Phase 3: Biomechanics ===
    from src.biomechanics import BiomechanicsAnalyzer

    bio_cfg = config.get("biomechanics", {})
    bio = BiomechanicsAnalyzer(
        idle_velocity_threshold=bio_cfg.get("idle_velocity_threshold_px_per_sec", 15.0),
        trunk_flex_threshold=bio_cfg.get("trunk_flex_threshold_deg", 20.0),
        wrist_above_shoulder_threshold=bio_cfg.get("wrist_above_shoulder_threshold", 0.0),
        keypoint_conf_threshold=tracking_cfg.get("keypoint_conf_threshold", 0.3),
        full_rula=bio_cfg.get("full_rula", False),
        force_load_score=bio_cfg.get("force_load_score", 0),
        angle_smoothing_window=bio_cfg.get("angle_smoothing_window", 5),
    )

    # === Phase 3b: Object Detection ===
    obj_cfg = config.get("object_detection", {})
    obj_detector = None
    if obj_cfg.get("enabled", False):
        from src.object_detection import ObjectDetector
        obj_detector = ObjectDetector(
            conf_threshold=obj_cfg.get("conf_threshold", 0.3),
            device="mps",
        )
        logger.info("Object detection enabled for task confirmation")

    # === Phase 4: Task Classification ===
    learned_cfg = config.get("learned_classification", {})
    if learned_cfg.get("enabled", False):
        try:
            from src.learned_classifier import LearnedTaskClassifier
            class_cfg = config.get("classification", {})
            classifier = LearnedTaskClassifier(
                model_path=learned_cfg.get("model_path", "data/task_model.pt"),
                stats_path=learned_cfg.get("stats_path", "data/task_model_stats.json"),
                window_size=learned_cfg.get("window_size", 30),
                confidence_threshold=class_cfg.get("confidence_threshold", 0.6),
                min_task_duration_frames=class_cfg.get("min_task_duration_frames", 8),
                task_cooldown_frames=class_cfg.get("task_cooldown_frames", 15),
            )
            logger.info("Using LEARNED task classifier")
        except Exception as e:
            logger.warning("Failed to load learned classifier (%s), falling back to rule-based", e)
            from src.classification import TaskClassifier
            class_cfg = config.get("classification", {})
            classifier = TaskClassifier(
                confidence_threshold=class_cfg.get("confidence_threshold", 0.6),
                min_task_duration_frames=class_cfg.get("min_task_duration_frames", 8),
                task_cooldown_frames=class_cfg.get("task_cooldown_frames", 15),
                pick_velocity_threshold=class_cfg.get("pick_velocity_threshold", 30.0),
                stationary_threshold=class_cfg.get("stationary_threshold", 20.0),
                movement_velocity_threshold=class_cfg.get("movement_velocity_threshold", 60.0),
            )
    else:
        from src.classification import TaskClassifier
        class_cfg = config.get("classification", {})
        classifier = TaskClassifier(
            confidence_threshold=class_cfg.get("confidence_threshold", 0.6),
            min_task_duration_frames=class_cfg.get("min_task_duration_frames", 8),
            task_cooldown_frames=class_cfg.get("task_cooldown_frames", 15),
            pick_velocity_threshold=class_cfg.get("pick_velocity_threshold", 30.0),
            stationary_threshold=class_cfg.get("stationary_threshold", 20.0),
            movement_velocity_threshold=class_cfg.get("movement_velocity_threshold", 60.0),
        )

    # === Phase 5: Metrics Engine ===
    from src.metrics import MetricsEngine

    zones_cfg = config.get("zones", {})
    metrics_engine = MetricsEngine(
        fps=reader.fps,
        frame_width=reader.width,
        frame_height=reader.height,
        grid_cols=zones_cfg.get("grid_cols", 3),
        grid_rows=zones_cfg.get("grid_rows", 2),
        idle_velocity_threshold=bio_cfg.get("idle_velocity_threshold_px_per_sec", 15.0),
    )

    # === Phase 6: Annotation & Heatmap Setup ===
    import cv2

    from src.annotation import FrameAnnotator
    from src.heatmap import HeatmapGenerator

    output_cfg = config.get("output", {})

    annotator = FrameAnnotator(
        frame_width=reader.width,
        frame_height=reader.height,
        grid_cols=zones_cfg.get("grid_cols", 3),
        grid_rows=zones_cfg.get("grid_rows", 2),
        zone_labels=metrics_engine.get_zone_labels(),
    )

    heatmap_gen = HeatmapGenerator(
        frame_width=reader.width,
        frame_height=reader.height,
    )

    # Video writer
    video_writer = None
    annotated_path = f"output/annotated/annotated_{run_id}.mp4"
    if output_cfg.get("annotated_video", True):
        Path(annotated_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            annotated_path, fourcc, reader.fps, (reader.width, reader.height)
        )
        logger.info("Video writer initialized: %s", annotated_path)

    # === Main Processing Loop ===
    previous_frames: Dict[int, "PersonFrame"] = {}
    feature_records: List[dict] = []  # For --extract-features

    frame_count = 0
    first_frame_set = False

    for frame_index, timestamp_sec, frame_bgr in reader.iter_frames():
        # Set heatmap background from first frame
        if not first_frame_set:
            heatmap_gen.set_background(frame_bgr)
            first_frame_set = True

        # Detection + Tracking
        persons = tracker.update(frame_bgr, frame_index, timestamp_sec)

        # Object detection for task confirmation
        frame_objects = []
        if obj_detector is not None:
            frame_objects = obj_detector.detect_objects(frame_bgr)

        # Per-person processing
        task_labels: Dict[int, tuple] = {}
        rula_labels: Dict[int, tuple] = {}
        live_metrics: Dict[int, dict] = {}

        for person in persons:
            # Biomechanics
            joint_angles = bio.compute_joint_angles(person.keypoints, track_id=person.track_id)
            prev_frame = previous_frames.get(person.track_id)
            velocity = bio.compute_velocity(person, prev_frame, reader.fps)
            rula_score, rula_label = bio.compute_rula_score(joint_angles)
            is_active = not bio.is_idle(velocity)

            # Object proximity for this person
            object_proximity = None
            if frame_objects:
                from src.object_detection import ObjectDetector
                object_proximity = ObjectDetector.compute_hand_object_proximity(
                    person.keypoints, frame_objects
                )

            # Task classification
            task_name, task_conf, task_events = classifier.classify_frame(
                person, joint_angles, velocity, rula_score, rula_label, reader.fps,
                object_proximity=object_proximity
            )

            # Metrics update
            metrics_engine.update(
                person, velocity, is_active, joint_angles, rula_score, task_events
            )

            # Heatmap accumulation
            heatmap_gen.accumulate(person.track_id, person.centroid)

            # Store for annotation
            task_labels[person.track_id] = (task_name, task_conf)
            rula_labels[person.track_id] = (rula_score, rula_label)

            # Live metrics for overlay
            state = metrics_engine.get_person_state(person.track_id)
            total_frames_seen = len(state.frame_history)
            active_count = sum(1 for v in state.velocity_history if v >= bio._idle_threshold)
            live_metrics[person.track_id] = {
                "active_ratio": active_count / total_frames_seen if total_frames_seen > 0 else 0.0,
                "total_movement": sum(
                    abs(state.velocity_history[i]) / reader.fps
                    for i in range(1, len(state.velocity_history))
                ) if len(state.velocity_history) > 1 else 0.0,
            }

            previous_frames[person.track_id] = person

            # Feature extraction for training
            if extract_features:
                feature_records.append({
                    "frame_index": frame_index,
                    "track_id": person.track_id,
                    "timestamp_sec": round(timestamp_sec, 4),
                    "elbow_left": round(joint_angles.get("elbow_left", 0.0), 2),
                    "elbow_right": round(joint_angles.get("elbow_right", 0.0), 2),
                    "shoulder_left": round(joint_angles.get("shoulder_left", 0.0), 2),
                    "shoulder_right": round(joint_angles.get("shoulder_right", 0.0), 2),
                    "trunk_flex": round(joint_angles.get("trunk_flex", 0.0), 2),
                    "wrist_height_left": round(joint_angles.get("wrist_height_left", 0.0), 4),
                    "wrist_height_right": round(joint_angles.get("wrist_height_right", 0.0), 4),
                    "velocity": round(velocity, 2),
                    "centroid_x": round(person.centroid[0], 2),
                    "centroid_y": round(person.centroid[1], 2),
                    "rula_score": round(rula_score, 2),
                    "rule_task": task_name,
                    "rule_confidence": round(task_conf, 4),
                })

        # Annotate frame
        if video_writer is not None:
            annotated = annotator.annotate_frame(
                frame_bgr, persons, task_labels, rula_labels,
                tracker.track_history, live_metrics,
                frame_index, timestamp_sec,
            )
            video_writer.write(annotated)

        frame_count += 1
        if frame_count % 100 == 0:
            logger.info("Processed %d frames...", frame_count)

    # Close video writer
    if video_writer is not None:
        video_writer.release()
        logger.info("Annotated video saved: %s", annotated_path)

    # === Finalize ===
    # Close any open task events
    final_frame = frame_count - 1
    for track_id in metrics_engine.get_all_track_ids():
        remaining_events = classifier.finalize(track_id, final_frame, reader.fps)
        metrics_engine.add_task_events(remaining_events)

    # Export metrics
    csv_path = f"output/metrics/metrics_{run_id}.csv"
    json_path = f"output/metrics/metrics_{run_id}.json"

    if output_cfg.get("metrics_csv", True):
        metrics_engine.export_csv(csv_path)

    if output_cfg.get("metrics_json", True):
        metrics_engine.export_json(json_path)

    # Generate heatmaps
    if output_cfg.get("heatmaps", True):
        for tid in heatmap_gen.get_all_track_ids():
            heatmap_gen.save_person_heatmap(
                tid, f"output/heatmaps/heatmap_person_{tid}_{run_id}.png"
            )
        heatmap_gen.save_combined_heatmap(f"output/heatmaps/heatmap_combined_{run_id}.png")

    # Generate PDF report
    if output_cfg.get("pdf_report", True):
        from src.reporting import ReportGenerator

        all_metrics = [metrics_engine.build_metrics(tid) for tid in sorted(metrics_engine.get_all_track_ids())]
        heatmap_paths = {
            tid: f"output/heatmaps/heatmap_person_{tid}_{run_id}.png"
            for tid in heatmap_gen.get_all_track_ids()
        }

        reporter = ReportGenerator(
            video_filename=Path(video_path).name,
            video_duration_sec=reader.duration_sec,
            fps=reader.fps,
            total_frames=frame_count,
        )
        reporter.generate(
            all_metrics, heatmap_paths,
            f"output/reports/report_{run_id}.pdf", run_id,
        )

    # Save extracted features
    if extract_features and feature_records:
        features_path = f"data/features_{run_id}.json"
        Path(features_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(features_path, "w") as f:
                json.dump(feature_records, f)
            logger.info("Saved %d feature records to %s", len(feature_records), features_path)
        except Exception as e:
            logger.error("Failed to save features: %s", e)

    # Release resources
    reader.release()

    logger.info("Pipeline complete. Run ID: %s | Frames: %d | Persons: %d",
                run_id, frame_count, len(metrics_engine.get_all_track_ids()))


def _train_learned_classifier(config: dict, run_id: str) -> None:
    """Train the learned task classifier on previously extracted features.

    Args:
        config: Pipeline configuration dict.
        run_id: Current run ID.
    """
    import glob

    # Find most recent features file
    feature_files = sorted(glob.glob("data/features_*.json"))
    if not feature_files:
        logger.error("No feature files found in data/. Run with --extract-features first.")
        return

    features_path = feature_files[-1]
    logger.info("Training learned classifier from: %s", features_path)

    learned_cfg = config.get("learned_classification", {})
    try:
        from src.learned_classifier import train_model
        stats = train_model(
            features_path=features_path,
            model_save_path=learned_cfg.get("model_path", "data/task_model.pt"),
            stats_save_path=learned_cfg.get("stats_path", "data/task_model_stats.json"),
            window_size=learned_cfg.get("window_size", 30),
            epochs=learned_cfg.get("epochs", 50),
            batch_size=learned_cfg.get("batch_size", 64),
            learning_rate=learned_cfg.get("learning_rate", 0.001),
            n_clusters=learned_cfg.get("n_clusters", 8),
        )
        logger.info("Training complete: %s", stats)
    except Exception as e:
        logger.error("Training failed: %s", e)
        raise


def main() -> None:
    """CLI entry point."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Factory Biomechanics Tracker Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default="config/default.yaml",
        help="Path to YAML configuration file (default: config/default.yaml)",
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to input video file (overrides config)",
    )
    parser.add_argument(
        "--extract-features", action="store_true",
        help="Extract per-frame features for training the learned classifier",
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Train the learned classifier on previously extracted features",
    )

    args = parser.parse_args()
    run_pipeline(args.config, args.input, args.extract_features, args.train)


if __name__ == "__main__":
    main()
