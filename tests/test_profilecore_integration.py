import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd

from drone_app.analyzer import TelemetryAnalyzer
from drone_app.pipeline import run_analysis_pipeline
from drone_app.report_exporter import DroneReportExporter
from drone_app.visualizer import TelemetryVisualizer
from profilecore.core.context import ProfileCoreContext
from profilecore.core.quality import build_data_quality_summary
import dronelog_uiapps


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
        self.assertIn("pca_preprocessing_report", self.context.artifacts)
        self.assertIn("pca_loadings", self.context.artifacts)
        self.assertIn("anomaly_timestamps", self.context.artifacts)
        self.assertIn("anomaly_detection_config", self.context.artifacts)
        self.assertIn("pca_loadings", self.context.data)
        self.assertIn("anomaly_timestamps", self.context.data)

        summary = self.context.get_artifact("pca_summary")
        self.assertEqual(summary["n_components"], 3)
        self.assertEqual(len(summary["explained_variance_ratio"]), 3)

        loadings = self.context.get_artifact("pca_loadings")
        self.assertIn("PC1", loadings)
        self.assertIn("positive", loadings["PC1"])
        self.assertIn("negative", loadings["PC1"])

        preprocessing = self.context.get_artifact("pca_preprocessing_report")
        self.assertEqual(preprocessing["status"], "completed")
        self.assertEqual(preprocessing["selected_column_count"], 5)
        self.assertEqual(preprocessing["effective_n_components"], 3)

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
        self.context.set_artifact("ulg_parse_report", {
            "parsed_topics": ["sensor_combined", "actuator_outputs"],
            "missing_topics": [],
            "failed_topics": {},
            "resolved_alternatives": {"vehicle_global_position": "vehicle_gps_position"},
            "topic_stats": {
                "sensor_combined": {
                    "rows": 60,
                    "columns": 4,
                    "instances": 1,
                    "parsed_instances": 1,
                },
            },
            "instance_stats": {
                "sensor_combined": {
                    "topic": "sensor_combined",
                    "multi_instance": 0,
                    "rows": 60,
                    "columns": 4,
                },
            },
            "resample_rate": "100ms",
            "fill_strategy": "bounded",
            "fill_limits": {"sensor_combined": 1},
            "sparse_sources": [],
            "output_rows": 60,
            "output_columns": 5,
            "output_missing_values": 0,
        })
        self.context.set_artifact("csv_parse_report", {
            "source_file": "sample.csv",
            "rows": 60,
            "columns": 5,
            "timestamp_column": "time_s",
            "timestamp_status": "numeric",
            "timestamp_unit": "s",
            "monotonic_timestamp": True,
            "duplicate_timestamps": 0,
            "numeric_column_count": 5,
            "missing_values": 0,
            "column_mapping_applied": {
                "acc_x": "sensor_combined_accelerometer_m_s2[0]",
            },
            "numeric_columns": ["sensor_combined_accelerometer_m_s2[0]"],
            "coerced_numeric_columns": [],
            "unmapped_config_columns": [],
            "all_nan_columns": [],
            "constant_columns": [],
        })

        visualizer = TelemetryVisualizer(self.context, output_dir=self.output_dir)
        visualizer.plot_raw_telemetry(filename="raw_telemetry.png")
        visualizer.plot_pca_results(filename="pca_plot.png")
        visualizer.plot_variance(filename="pca_variance.png")

        exporter = DroneReportExporter(self.context, output_dir=self.output_dir)
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
        self.assertIn("ULog Parse Report", report)
        self.assertIn("Resolved Alternatives", report)
        self.assertIn("vehicle_gps_position", report)
        self.assertIn("CSV Parse Report", report)
        self.assertIn("Column Mapping", report)
        self.assertIn("PCA Preprocessing Report", report)
        self.assertIn("PCA Anomaly Detection Report", report)
        self.assertIn("raw_telemetry.png", report)
        self.assertIn("pca_plot.png", report)
        self.assertIn("pca_variance.png", report)

    def test_pipeline_runs_csv_input_with_dummy_llm(self):
        csv_path = os.path.join(self.output_dir, "telemetry.csv")
        self._synthetic_telemetry().reset_index(names="time").to_csv(csv_path, index=False)

        results = run_analysis_pipeline(
            csv_path,
            llm_type="dummy",
            workspace_dir=self.output_dir,
            output_dir=self.output_dir,
            diagnosis_filename="diagnosis.md",
            report_filename="drone_analysis_report.md",
            anomaly_z_threshold=2.5,
        )

        self.assertTrue(os.path.exists(results["report_path"]))
        self.assertTrue(os.path.exists(results["diagnosis_path"]))
        self.assertIn("context", results)

        history = results["context"].get_data("flight_history")
        self.assertIsNotNone(history)
        self.assertIn("source_sha256", history.columns)
        self.assertIn(os.path.join("runs", ""), results["output_dir"] + os.sep)

        anomaly_config = results["context"].get_artifact("anomaly_detection_config")
        self.assertEqual(anomaly_config["z_threshold"], 2.5)

        with open(results["report_path"], encoding="utf-8") as report_file:
            report = report_file.read()
        self.assertIn("CSV Parse Report", report)
        self.assertIn("PCA Preprocessing Report", report)
        self.assertIn("PCA Anomaly Detection Report", report)
        self.assertIn("Structural Break Report", report)

    def test_pipeline_export_mode_writes_prompt_without_api_key(self):
        csv_path = os.path.join(self.output_dir, "telemetry_export.csv")
        self._synthetic_telemetry().reset_index(names="time").to_csv(csv_path, index=False)
        old_api_key = os.environ.pop("GEMINI_API_KEY", None)
        try:
            results = run_analysis_pipeline(
                csv_path,
                llm_type="gemini",
                model_name="gemini-export-test",
                mode="export",
                workspace_dir=self.output_dir,
                output_dir=self.output_dir,
                run_output_subdir=False,
                diagnosis_filename="diagnosis_export.md",
                report_filename="drone_analysis_report_export.md",
            )
        finally:
            if old_api_key is not None:
                os.environ["GEMINI_API_KEY"] = old_api_key

        prompt_path = os.path.join(self.output_dir, "llm_prompt.txt")
        self.assertEqual(results["llm_settings"]["mode"], "export")
        self.assertEqual(results["llm_settings"]["service"], "gemini")
        self.assertTrue(os.path.exists(prompt_path))
        self.assertTrue(os.path.exists(results["diagnosis_path"]))
        self.assertTrue(os.path.exists(results["report_path"]))

        with open(results["diagnosis_path"], encoding="utf-8") as diagnosis_file:
            diagnosis = diagnosis_file.read()
        self.assertIn("プロンプト・エクスポート", diagnosis)
        self.assertIn("gemini-export-test", diagnosis)

        with open(results["report_path"], encoding="utf-8") as report_file:
            report = report_file.read()
        self.assertIn("CSV Parse Report", report)
        self.assertIn("PCA Preprocessing Report", report)

    def test_ui_analysis_passes_resolved_export_settings_to_pipeline(self):
        config_path = os.path.join(self.output_dir, "llm_config.json")
        with open(config_path, "w", encoding="utf-8") as config_file:
            config_file.write('{"service": "gemini", "model": "gemini-config-model", "mode": "api"}')

        with patch.object(dronelog_uiapps, "LLM_CONFIG_PATH", config_path), \
                patch.object(dronelog_uiapps, "WORKSPACE_DIR", self.output_dir), \
                patch.object(dronelog_uiapps, "OUTPUT_DIR", self.output_dir), \
                patch.object(dronelog_uiapps, "run_analysis_pipeline") as pipeline_mock:
            pipeline_mock.return_value = {"context": self.context}
            result = dronelog_uiapps.run_ui_analysis(
                "uploaded.csv",
                "gemini",
                "",
                mode="export",
                anomaly_z_threshold=2.0,
            )

        self.assertEqual(result, {"context": self.context})
        pipeline_mock.assert_called_once()
        kwargs = pipeline_mock.call_args.kwargs
        self.assertEqual(kwargs["llm_type"], "gemini")
        self.assertEqual(kwargs["model_name"], "gemini-config-model")
        self.assertEqual(kwargs["mode"], "export")
        self.assertEqual(kwargs["anomaly_z_threshold"], 2.0)

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
