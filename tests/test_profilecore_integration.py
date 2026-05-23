import os
import shutil
import tempfile
import unittest

os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd

from drone_app.analyzer import TelemetryAnalyzer
from drone_app.visualizer import TelemetryVisualizer
from profilecore.core.context import ProfileCoreContext
from profilecore.core.quality import build_data_quality_summary
from profilecore.io.exporter import ReportExporter


class TestDroneProfileCoreIntegration(unittest.TestCase):
    def setUp(self):
        self.output_dir = tempfile.mkdtemp(prefix="drone_profilecore_test_")
        self.context = ProfileCoreContext(workspace_dir=self.output_dir)
        self.df = self._synthetic_telemetry()
        self.context.set_data("raw_data", self.df)
        self.context.set_artifact("data_quality", build_data_quality_summary(self.df))

    def tearDown(self):
        shutil.rmtree(self.output_dir, ignore_errors=True)

    def test_analyzer_registers_report_artifacts(self):
        TelemetryAnalyzer(self.context).analyze(data_key="raw_data", n_components=3)

        self.assertIn("pca_summary", self.context.artifacts)
        self.assertIn("pca_loadings", self.context.artifacts)
        self.assertIn("anomaly_timestamps", self.context.artifacts)
        self.assertIn("pca_loadings", self.context.data)
        self.assertIn("anomaly_timestamps", self.context.data)

        summary = self.context.get_artifact("pca_summary")
        self.assertEqual(summary["n_components"], 3)
        self.assertEqual(len(summary["explained_variance_ratio"]), 3)

        loadings = self.context.get_artifact("pca_loadings")
        self.assertIn("PC1", loadings)
        self.assertIn("positive", loadings["PC1"])
        self.assertIn("negative", loadings["PC1"])

    def test_visualizer_registers_figure_outputs(self):
        TelemetryAnalyzer(self.context).analyze(data_key="raw_data", n_components=3)

        visualizer = TelemetryVisualizer(self.context, output_dir=self.output_dir)
        visualizer.plot_raw_telemetry(filename="raw_telemetry.png")
        visualizer.plot_pca_results(filename="pca_plot.png")
        visualizer.plot_variance(filename="pca_variance.png")

        figures = self.context.outputs.get("figure", [])
        self.assertEqual(len(figures), 3)
        self.assertTrue(all(os.path.exists(path) for path in figures))

    def test_report_exporter_contains_drone_sections(self):
        TelemetryAnalyzer(self.context).analyze(data_key="raw_data", n_components=3)
        pca_variance = self.context.get_data("pca_variance")
        self.context.set_artifact("feature_extraction_status", {
            "status": "completed",
            "method": "telemetry_pca",
            "n_components": int(len(pca_variance)),
        })
        self.context.set_artifact("summary_insights", [{
            "level": "info",
            "message": "Synthetic telemetry integration test.",
        }])

        visualizer = TelemetryVisualizer(self.context, output_dir=self.output_dir)
        visualizer.plot_raw_telemetry(filename="raw_telemetry.png")
        visualizer.plot_pca_results(filename="pca_plot.png")
        visualizer.plot_variance(filename="pca_variance.png")

        exporter = ReportExporter(self.context, output_dir=self.output_dir)
        exporter.export_markdown(filename="drone_analysis_report.md")

        report_path = os.path.join(self.output_dir, "drone_analysis_report.md")
        with open(report_path, encoding="utf-8") as report_file:
            report = report_file.read()

        self.assertNotIn("Rows analyzed: N/A", report)
        self.assertNotIn("No data quality summary was generated.", report)
        self.assertIn("Feature extraction: completed", report)
        self.assertIn("Key Findings", report)
        self.assertIn("PCA Summary", report)
        self.assertIn("PCA Loadings", report)
        self.assertIn("raw_telemetry.png", report)
        self.assertIn("pca_plot.png", report)
        self.assertIn("pca_variance.png", report)

    def _synthetic_telemetry(self):
        index = pd.to_timedelta([i * 100 for i in range(60)], unit="ms")
        return pd.DataFrame({
            "sensor_combined_accelerometer_m_s2[0]": [i * 0.1 for i in range(60)],
            "sensor_combined_accelerometer_m_s2[1]": [((-1) ** i) * i * 0.05 for i in range(60)],
            "sensor_combined_accelerometer_m_s2[2]": [i % 7 for i in range(60)],
            "actuator_outputs_output[0]": [1000 + i for i in range(60)],
            "actuator_outputs_output[1]": [1000 + (i % 5) for i in range(60)],
        }, index=index)


if __name__ == "__main__":
    unittest.main()
