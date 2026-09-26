"""Report Exporter Utility for SIGNASSIST (Phase 13).

Provides:
- ReportExporter class for exporting VideoProcessingResult into:
  - Structured JSON format (.json)
  - Formatted human-readable PDF document (.pdf) using ReportLab
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from src.video.processor import VideoProcessingResult


from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.utils.exceptions import DataError
from src.utils.logger import setup_logger

logger = setup_logger(__name__)



def _to_json_safe(obj: Any) -> Any:
    """Recursively converts objects to JSON-serializable primitives."""
    if obj is None:
        return None
    if isinstance(obj, (int, float, str, bool)):
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj):
        return {k: _to_json_safe(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_json_safe(item) for item in obj]
    if hasattr(obj, "item"):  # NumPy or PyTorch scalar
        return obj.item()
    if hasattr(obj, "tolist"):  # NumPy or PyTorch array
        return obj.tolist()
    return str(obj)


class ReportExporter:
    """Exports VideoProcessingResult artifacts to JSON and PDF formats."""

    @staticmethod
    def export_json(
        result: VideoProcessingResult,
        output_path: Union[str, Path],
        indent: int = 2,
    ) -> Path:
        """Exports VideoProcessingResult to a structured JSON file."""
        if result is None:
            raise DataError("Cannot export JSON for None VideoProcessingResult")

        out_path = Path(output_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            json_data = _to_json_safe(result)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=indent, ensure_ascii=False)

            logger.info(f"Successfully exported JSON report to '{out_path}'")
            return out_path
        except Exception as e:
            raise DataError(f"Failed to export JSON report to '{out_path}': {e}") from e

    @staticmethod
    def export_pdf(
        result: VideoProcessingResult,
        output_path: Union[str, Path],
    ) -> Path:
        """Exports VideoProcessingResult to a formatted ReportLab PDF document."""
        if result is None:
            raise DataError("Cannot export PDF for None VideoProcessingResult")

        out_path = Path(output_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            doc = SimpleDocTemplate(
                str(out_path),
                pagesize=letter,
                rightMargin=36,
                leftMargin=36,
                topMargin=36,
                bottomMargin=36,
            )
            styles = getSampleStyleSheet()

            # Custom styles
            title_style = ParagraphStyle(
                "ReportTitle",
                parent=styles["Heading1"],
                fontSize=20,
                leading=24,
                textColor=colors.HexColor("#1A365D"),
                spaceAfter=12,
            )
            h2_style = ParagraphStyle(
                "SectionHeading",
                parent=styles["Heading2"],
                fontSize=13,
                leading=16,
                textColor=colors.HexColor("#2B6CB0"),
                spaceBefore=10,
                spaceAfter=6,
            )
            body_style = ParagraphStyle(
                "ReportBody",
                parent=styles["Normal"],
                fontSize=10,
                leading=14,
                textColor=colors.HexColor("#2D3748"),
            )

            story = []

            # 1. Title
            story.append(Paragraph("SIGNASSIST — Video Recognition & Translation Report", title_style))
            story.append(Spacer(1, 8))

            # 2. Video Metadata Table
            story.append(Paragraph("1. Video File Metadata", h2_style))
            meta = result.video_metadata
            meta_data = [
                ["Property", "Value", "Property", "Value"],
                ["File Name", meta.file_name, "Duration", f"{meta.duration_sec:.2f} s"],
                ["Resolution", f"{meta.width} x {meta.height}", "Native FPS", f"{meta.native_fps:.2f}"],
                ["Codec", meta.codec, "Total Frames", str(meta.total_frames)],
                ["File Size", f"{meta.file_size_bytes / (1024*1024):.2f} MB", "Aspect Ratio", f"{meta.aspect_ratio:.2f}"],
            ]
            t_meta = Table(meta_data, colWidths=[100, 160, 100, 160])
            t_meta.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1A202C")),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 9),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ]
                )
            )
            story.append(t_meta)
            story.append(Spacer(1, 10))

            # 3. Quality Report Section
            story.append(Paragraph("2. Video Quality Assessment", h2_style))
            q = result.quality_report
            q_data = [
                ["Metric", "Score / Status", "Metric", "Score / Status"],
                ["Average Brightness", f"{q.average_brightness:.1f}", "Average Contrast", f"{q.average_contrast:.1f}"],
                ["Average Blur Score", f"{q.average_blur_score:.1f}", "Resolution OK", "Yes" if q.resolution_ok else "No"],
                ["Hand Visibility Ratio", f"{q.hand_visibility_ratio * 100:.1f}%", "Overall Rating", q.overall_rating],
            ]
            t_q = Table(q_data, colWidths=[100, 160, 100, 160])
            t_q.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1A202C")),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 9),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ]
                )
            )
            story.append(t_q)
            story.append(Spacer(1, 10))

            # 4. Execution Metrics
            story.append(Paragraph("3. Processing Performance", h2_style))
            p_data = [
                ["Frames Processed", str(result.frames_processed), "Execution Time", f"{result.execution_seconds:.2f} s"],
                ["Processing Rate", f"{result.processing_fps:.1f} FPS", "Recognition Engine", "Available" if result.recognition_available else "Unavailable / Mock"],
            ]
            t_p = Table(p_data, colWidths=[130, 130, 130, 130])
            t_p.setStyle(
                TableStyle(
                    [
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                        ("FONTSIZE", (0, 0), (-1, -1), 9),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ]
                )
            )
            story.append(t_p)
            story.append(Spacer(1, 10))

            # 5. Reconstructed Transcript Section
            story.append(Paragraph("4. Reconstructed Sentence Transcript", h2_style))
            if result.transcript and result.transcript.text.strip():
                txt = result.transcript.text
                conf_str = f"Mean Model Confidence: {result.transcript.mean_confidence * 100:.1f}%"
                story.append(Paragraph(f"<b>Reconstructed Text:</b> \"{txt}\"", body_style))
                story.append(Paragraph(f"<i>{conf_str}</i>", body_style))
            else:
                story.append(Paragraph("<i>No transcript reconstructed (recognition unavailable or no signs detected).</i>", body_style))
            story.append(Spacer(1, 10))

            # 6. Timestamped Recognition Timeline Table
            story.append(Paragraph("5. Timestamped Recognition Timeline", h2_style))
            if result.timeline:
                timeline_data = [["Time Range", "Gloss", "Conf", "Top Candidates", "Signer Track"]]
                for entry in result.timeline:
                    top_str = ", ".join([f"{g}:{c:.2f}" for g, c in entry.top_3_candidates]) if entry.top_3_candidates else "-"
                    trk_str = f"Track #{entry.primary_signer_track_id}" if entry.primary_signer_track_id else "-"
                    time_range = f"{entry.start_timestamp} - {entry.end_timestamp}"
                    timeline_data.append([
                        time_range,
                        entry.recognized_gloss,
                        f"{entry.confidence:.2f}",
                        top_str,
                        trk_str,
                    ])

                t_timeline = Table(timeline_data, colWidths=[120, 90, 50, 180, 80])
                t_timeline.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3182CE")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("FONTSIZE", (0, 0), (-1, -1), 8),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                            ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ]
                    )
                )
                story.append(t_timeline)
            else:
                story.append(Paragraph("<i>No timestamped sign entries in timeline.</i>", body_style))

            doc.build(story)
            logger.info(f"Successfully exported PDF report to '{out_path}'")
            return out_path
        except Exception as e:
            raise DataError(f"Failed to export PDF report to '{out_path}': {e}") from e
