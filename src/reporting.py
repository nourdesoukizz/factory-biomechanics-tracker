"""PDF report generation module — auto-generates polished summary reports."""

import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
)

from src.models import PersonMetrics

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates PDF summary reports from pipeline metrics."""

    def __init__(self, video_filename: str, video_duration_sec: float,
                 fps: float, total_frames: int) -> None:
        """Initialize the report generator.

        Args:
            video_filename: Name of the input video file.
            video_duration_sec: Video duration in seconds.
            fps: Video frames per second.
            total_frames: Total frames processed.
        """
        self._video_filename = video_filename
        self._duration = video_duration_sec
        self._fps = fps
        self._total_frames = total_frames
        self._styles = getSampleStyleSheet()

    def generate(self, metrics: List[PersonMetrics], heatmap_paths: Dict[int, str],
                 output_path: str, run_id: str) -> None:
        """Generate the full PDF report.

        Args:
            metrics: List of PersonMetrics for all tracked persons.
            heatmap_paths: Dict of track_id -> heatmap PNG path.
            output_path: Path to save the PDF.
            run_id: Run identifier string.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4,
                rightMargin=20 * mm,
                leftMargin=20 * mm,
                topMargin=20 * mm,
                bottomMargin=20 * mm,
            )

            elements = []
            elements.extend(self._build_cover(run_id, len(metrics)))
            elements.append(PageBreak())
            elements.extend(self._build_summary_table(metrics))

            for m in metrics:
                elements.append(PageBreak())
                elements.extend(self._build_person_detail(m, heatmap_paths.get(m.track_id)))

            elements.append(PageBreak())
            elements.extend(self._build_methodology())

            doc.build(elements)
            logger.info("Generated PDF report: %s", output_path)
        except Exception as e:
            logger.error("Failed to generate PDF report: %s", e)
            raise

    def _build_cover(self, run_id: str, person_count: int) -> list:
        """Build cover page elements.

        Args:
            run_id: Run identifier.
            person_count: Number of tracked persons.

        Returns:
            List of reportlab flowable elements.
        """
        elements = []

        title_style = ParagraphStyle(
            "CoverTitle", parent=self._styles["Title"],
            fontSize=28, spaceAfter=30, alignment=1,
        )
        subtitle_style = ParagraphStyle(
            "CoverSubtitle", parent=self._styles["Normal"],
            fontSize=14, spaceAfter=10, alignment=1, textColor=colors.grey,
        )

        elements.append(Spacer(1, 80))
        elements.append(Paragraph("Factory Floor Analysis Report", title_style))
        elements.append(Spacer(1, 20))
        elements.append(Paragraph(f"Run ID: {run_id}", subtitle_style))
        elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", subtitle_style))
        elements.append(Spacer(1, 40))

        # Info table
        info_data = [
            ["Video File", self._video_filename],
            ["Duration", f"{self._duration:.1f} seconds"],
            ["FPS", f"{self._fps:.1f}"],
            ["Frames Processed", str(self._total_frames)],
            ["Persons Tracked", str(person_count)],
        ]

        info_table = Table(info_data, colWidths=[150, 250])
        info_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.Color(0.9, 0.9, 0.9)),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("PADDING", (0, 0), (-1, -1), 8),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ]))

        elements.append(info_table)
        return elements

    def _build_summary_table(self, metrics: List[PersonMetrics]) -> list:
        """Build summary table page.

        Args:
            metrics: List of PersonMetrics.

        Returns:
            List of reportlab flowable elements.
        """
        elements = []

        header_style = ParagraphStyle(
            "SectionHeader", parent=self._styles["Heading1"],
            fontSize=18, spaceAfter=15,
        )
        elements.append(Paragraph("Summary — All Workers", header_style))

        # Table header
        header = ["Person ID", "Active %", "Total Movement (px)", "Top Task", "Avg RULA", "Peak RULA"]
        rows = [header]

        for m in sorted(metrics, key=lambda x: x.track_id):
            top_task = max(m.task_counts, key=m.task_counts.get) if m.task_counts else "idle"
            rows.append([
                str(m.track_id),
                f"{m.active_ratio * 100:.1f}%",
                f"{m.total_movement_px:.0f}",
                top_task,
                f"{m.avg_rula_score:.1f}",
                f"{m.peak_rula_score:.1f}",
            ])

        table = Table(rows, colWidths=[60, 70, 100, 100, 70, 70])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.2, 0.3, 0.5)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 6),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.Color(0.95, 0.95, 0.95)]),
        ]))

        elements.append(table)
        return elements

    def _build_person_detail(self, m: PersonMetrics,
                             heatmap_path: Optional[str]) -> list:
        """Build per-person detail page.

        Args:
            m: PersonMetrics for this person.
            heatmap_path: Path to heatmap PNG (or None).

        Returns:
            List of reportlab flowable elements.
        """
        elements = []

        header_style = ParagraphStyle(
            "PersonHeader", parent=self._styles["Heading1"],
            fontSize=16, spaceAfter=10,
        )
        elements.append(Paragraph(f"Person {m.track_id} — Detailed Analysis", header_style))
        elements.append(Spacer(1, 10))

        # Active/Idle pie chart
        pie_img = self._create_pie_chart(m.active_ratio, m.track_id)
        if pie_img:
            elements.append(Image(pie_img, width=3 * inch, height=2.5 * inch))
            elements.append(Spacer(1, 10))

        # Zone dwell time bar chart
        if m.zone_dwell_times:
            zone_img = self._create_zone_chart(m.zone_dwell_times, m.track_id)
            if zone_img:
                elements.append(Image(zone_img, width=4 * inch, height=2.5 * inch))
                elements.append(Spacer(1, 10))

        # Task event summary
        if m.task_counts:
            task_img = self._create_task_chart(m.task_counts, m.track_id)
            if task_img:
                elements.append(Image(task_img, width=4 * inch, height=2 * inch))
                elements.append(Spacer(1, 10))

        # Heatmap thumbnail
        if heatmap_path and Path(heatmap_path).exists():
            elements.append(Paragraph("Spatial Presence Heatmap", self._styles["Heading3"]))
            elements.append(Image(heatmap_path, width=4 * inch, height=3 * inch))

        return elements

    def _create_pie_chart(self, active_ratio: float, track_id: int) -> Optional[str]:
        """Create active/idle pie chart as temporary image.

        Args:
            active_ratio: Fraction of time active.
            track_id: Person ID for title.

        Returns:
            Path to temporary PNG or None on error.
        """
        try:
            fig, ax = plt.subplots(figsize=(3, 2.5))
            sizes = [active_ratio * 100, (1 - active_ratio) * 100]
            labels = [f"Active ({sizes[0]:.1f}%)", f"Idle ({sizes[1]:.1f}%)"]
            chart_colors = ["#2ECC71", "#95A5A6"]

            ax.pie(sizes, labels=labels, colors=chart_colors, startangle=90,
                   autopct="", shadow=False)
            ax.set_title(f"Person {track_id} — Active vs Idle", fontsize=10)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            return buf
        except Exception as e:
            logger.error("Failed to create pie chart: %s", e)
            return None

    def _create_zone_chart(self, zone_dwell: Dict[str, float],
                           track_id: int) -> Optional[str]:
        """Create zone dwell time bar chart.

        Args:
            zone_dwell: Zone ID -> dwell time in seconds.
            track_id: Person ID for title.

        Returns:
            BytesIO PNG buffer or None.
        """
        try:
            fig, ax = plt.subplots(figsize=(4, 2.5))
            zones = sorted(zone_dwell.keys())
            times = [zone_dwell[z] for z in zones]

            bars = ax.bar(zones, times, color="#3498DB", alpha=0.8)
            ax.set_ylabel("Time (seconds)")
            ax.set_title(f"Person {track_id} — Zone Dwell Times", fontsize=10)
            ax.grid(axis="y", alpha=0.3)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            return buf
        except Exception as e:
            logger.error("Failed to create zone chart: %s", e)
            return None

    def _create_task_chart(self, task_counts: Dict[str, int],
                           track_id: int) -> Optional[str]:
        """Create task count bar chart.

        Args:
            task_counts: Task name -> count.
            track_id: Person ID for title.

        Returns:
            BytesIO PNG buffer or None.
        """
        try:
            fig, ax = plt.subplots(figsize=(4, 2))
            tasks = sorted(task_counts.keys())
            counts = [task_counts[t] for t in tasks]

            task_colors_map = {
                "pick_and_place": "#3498DB",
                "lift_and_place": "#E67E22",
                "move_rack": "#9B59B6",
            }
            bar_colors = [task_colors_map.get(t, "#95A5A6") for t in tasks]

            ax.barh(tasks, counts, color=bar_colors, alpha=0.8)
            ax.set_xlabel("Count")
            ax.set_title(f"Person {track_id} — Task Events", fontsize=10)
            ax.grid(axis="x", alpha=0.3)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            return buf
        except Exception as e:
            logger.error("Failed to create task chart: %s", e)
            return None

    def _build_methodology(self) -> list:
        """Build methodology page.

        Returns:
            List of reportlab flowable elements.
        """
        elements = []

        header_style = ParagraphStyle(
            "MethodHeader", parent=self._styles["Heading1"],
            fontSize=18, spaceAfter=15,
        )
        elements.append(Paragraph("Methodology", header_style))

        body_style = ParagraphStyle(
            "MethodBody", parent=self._styles["Normal"],
            fontSize=10, spaceAfter=8, leading=14,
        )

        sections = [
            ("<b>Detection Model</b>: YOLOv8m-pose was used for person detection and 17-keypoint "
             "pose estimation. The model provides bounding boxes and body keypoints per frame with "
             "high accuracy on factory floor scenarios."),

            ("<b>Tracking Algorithm</b>: ByteTrack was used for multi-object tracking with a post-tracking "
             "re-identification (ReID) layer. ReID uses HSV color histograms for appearance matching, "
             "linear motion prediction for spatial proximity, and temporal gap penalties to merge "
             "fragmented tracks after occlusion. 2D Kalman filters (tracking position and velocity) "
             "are applied per keypoint for temporal smoothing."),

            ("<b>Task Classification</b>: A learned 1D Temporal CNN (TemporalTaskNet) classifies tasks "
             "from 30-frame sliding windows of 11 pose features (joint angles, velocity, centroid, RULA). "
             "The model was trained on pseudo-labels from an initial rule-based pass, refined via K-means "
             "clustering. Three factory tasks are detected: pick &amp; place, lift &amp; place, and rack "
             "movement. A state machine with minimum duration and cooldown prevents noise."),

            ("<b>Ergonomic Scoring</b>: Full RULA (Rapid Upper Limb Assessment) scoring with Groups A "
             "(upper arm, lower arm, wrist, wrist twist) and B (neck, trunk, legs). Standard RULA lookup "
             "tables A, B, and C are used per McAtamney &amp; Corlett (1993). Muscle use (+1 for repetitive "
             "work) and configurable force/load adjustments are included. Neck flexion is computed from "
             "nose-to-shoulder vectors; trunk side-bend from shoulder/hip line asymmetry."),

            ("<b>2D Pose Mitigation</b>: Foreshortening detection compares observed limb lengths to "
             "expected proportions of person height, flagging unreliable angles. Temporal smoothing "
             "applies a 5-frame weighted moving average to joint angles. Body proportion validation "
             "detects implausible 2D projection artifacts."),

            ("<b>Metrics</b>: Active/idle classification is based on centroid velocity thresholding. "
             "Total movement is the cumulative Euclidean displacement of each person's centroid. "
             "Zone dwell times are computed by mapping centroids to a configurable 3x2 grid overlay."),
        ]

        for text in sections:
            elements.append(Paragraph(text, body_style))

        return elements
