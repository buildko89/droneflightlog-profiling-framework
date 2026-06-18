import os
import shutil
import tempfile
import unittest

import pandas as pd

from drone_app.report_exporter import DroneReportExporter
from drone_app.video_analyzer import VideoAnalyzer
from profilecore.core.context import ProfileCoreContext


class TestVideoAnalyzer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="video_analyzer_test_")
        self.context = ProfileCoreContext(workspace_dir=self.temp_dir)
        self.telemetry = pd.DataFrame(
            {"value": [1, 2, 3]},
            index=pd.to_timedelta([0, 5, 10], unit="s"),
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_missing_video_records_skipped_report(self):
        result = VideoAnalyzer(self.context).analyze(
            os.path.join(self.temp_dir, "missing.mp4"),
            self.telemetry,
            video_offset_s=2.0,
            camera_viewpoint="invalid",
        )

        self.assertEqual(result["video_parse_report"]["status"], "skipped")
        self.assertEqual(result["video_alignment"]["camera_viewpoint"], "external")
        self.assertEqual(result["video_coverage"]["start_elapsed_s"], 2.0)
        self.assertEqual(result["video_coverage"]["coverage_ratio"], 0.0)
        self.assertIn("video_features", self.context.data)
        self.assertIn("video_events", self.context.data)

    def test_telemetry_video_comparison_respects_coverage(self):
        self.context.set_artifact("anomaly_details", {
            "PC1": [
                {"timestamp": "00:03.0", "z_score": 4.0, "score": 1.0},
                {"timestamp": "00:08.0", "z_score": 4.5, "score": 2.0},
            ]
        })
        analyzer = VideoAnalyzer(self.context)
        coverage = {
            "start_elapsed_s": 2.0,
            "end_elapsed_s": 6.0,
            "coverage_ratio": 0.4,
        }
        rows = analyzer._compare_with_telemetry(
            self.context.get_artifact("anomaly_details"),
            [{"video_time_s": 1.0, "event": "rapid_movement", "confidence": 0.8}],
            coverage,
            event_window_s=1.0,
            alignment_confidence=0.9,
        )

        self.assertEqual(rows[0]["Result"], "Match")
        self.assertEqual(rows[1]["Result"], "No Coverage")
        self.assertEqual(rows[1]["Comment"], "動画による裏付けなし")
        self.assertEqual(rows[0]["Alignment Confidence"], 0.9)

    def test_phase_comparison_marks_low_confidence_contradiction(self):
        flight_phases = pd.DataFrame([
            {"telemetry_time_s": 3.0, "phase": "hover"},
        ])
        coverage = {
            "start_elapsed_s": 2.0,
            "end_elapsed_s": 6.0,
            "coverage_ratio": 0.4,
        }

        rows = VideoAnalyzer(self.context)._compare_phases_with_video(
            flight_phases,
            [{"video_time_s": 1.0, "event": "rapid_movement", "confidence": 0.6}],
            coverage,
            event_window_s=1.0,
            alignment_confidence=0.5,
        )

        self.assertEqual(rows[0]["Result"], "Contradiction(low-sync-confidence)")

    def test_report_exporter_writes_video_sections(self):
        self.context.set_artifact("video_parse_report", {
            "status": "completed",
            "file_name": "flight.mp4",
            "camera_viewpoint": "external",
            "duration_s": 4.0,
            "fps": 30.0,
            "width": 1920,
            "height": 1080,
            "codec": "mp4v",
            "frame_count": 120,
            "sample_interval_s": 1.0,
            "feature_rows": 4,
            "event_count": 1,
        })
        self.context.set_artifact("video_alignment", {
            "mode": "manual_offset",
            "video_offset_s": 2.0,
            "confidence": 0.8,
            "event_window_s": 1.0,
            "formula": "telemetry_time_s = video_time_s + video_offset_s",
            "camera_viewpoint": "external",
        })
        self.context.set_artifact("video_coverage", {
            "start_elapsed_s": 2.0,
            "end_elapsed_s": 6.0,
            "duration_s": 4.0,
            "telemetry_duration_s": 10.0,
            "coverage_ratio": 0.4,
        })
        self.context.set_data("video_events", pd.DataFrame([
            {"video_time_s": 1.0, "event": "rapid_movement", "confidence": 0.8},
        ]))
        self.context.set_artifact("telemetry_video_comparison", [
            {
                "Time": "00:03.0",
                "Log": "PC1 PCA Spike",
                "Video": "rapid_movement",
                "Result": "Match",
                "Comment": "PCA異常時刻の近傍に動画イベントがあります",
            }
        ])

        DroneReportExporter(self.context, output_dir=self.temp_dir).export_markdown("report.md")

        with open(os.path.join(self.temp_dir, "report.md"), encoding="utf-8") as report_file:
            report = report_file.read()
        self.assertIn("Video Summary", report)
        self.assertIn("Video Coverage", report)
        self.assertIn("Video Events", report)
        self.assertIn("Telemetry vs Video", report)
        self.assertIn("rapid_movement", report)


if __name__ == "__main__":
    unittest.main()
