import os

import pandas as pd

from profilecore.io.exporter import ReportExporter


class DroneReportExporter:
    """
    Adds drone-specific report sections on top of the generic ProfileCore report.
    """
    def __init__(self, context, output_dir: str = "output"):
        self.context = context
        self.output_dir = output_dir
        self.base_exporter = ReportExporter(context, output_dir=output_dir)

    def export_markdown(self, filename: str = "drone_analysis_report.md"):
        self.base_exporter.export_markdown(filename=filename)
        output_path = os.path.join(self.output_dir, filename)

        with open(output_path, "a", encoding="utf-8") as report_file:
            self._write_drone_sections(report_file)

    def _write_drone_sections(self, report_file):
        ulg_report = self.context.get_artifact("ulg_parse_report")
        csv_report = self.context.get_artifact("csv_parse_report")
        pca_preprocessing = self.context.get_artifact("pca_preprocessing_report")
        anomaly_config = self.context.get_artifact("anomaly_detection_config")
        anomaly_details = self.context.get_artifact("anomaly_details")
        structural_break = self.context.get_data("structural_break")
        flight_phase_report = self.context.get_artifact("flight_phase_report")
        flight_phases = self.context.get_data("flight_phases")
        video_report = self.context.get_artifact("video_parse_report")
        video_alignment = self.context.get_artifact("video_alignment")
        video_coverage = self.context.get_artifact("video_coverage")
        video_events = self.context.get_data("video_events")
        telemetry_video_comparison = self.context.get_artifact("telemetry_video_comparison")
        raw_summary = self.context.get_artifact("raw_telemetry_summary")

        if not any([
            ulg_report,
            csv_report,
            pca_preprocessing,
            anomaly_config,
            anomaly_details,
            structural_break,
            flight_phase_report,
            flight_phases is not None and not flight_phases.empty,
            video_report,
            video_alignment,
            video_coverage,
            video_events is not None and not video_events.empty,
            telemetry_video_comparison,
            raw_summary,
        ]):
            return

        report_file.write("## Drone Flight Log Details\n\n")
        self._write_ulg_parse_report(report_file, ulg_report)
        self._write_csv_parse_report(report_file, csv_report)
        self._write_raw_telemetry_summary(report_file, raw_summary)
        self._write_pca_preprocessing_report(report_file, pca_preprocessing)
        self._write_anomaly_report(report_file, anomaly_config, anomaly_details)
        self._write_structural_break_report(report_file, structural_break)
        self._write_flight_phase_report(report_file, flight_phase_report, flight_phases)
        self._write_video_report(
            report_file,
            video_report,
            video_alignment,
            video_coverage,
            video_events,
            telemetry_video_comparison,
        )

    def _write_ulg_parse_report(self, report_file, parse_report):
        if not parse_report:
            return

        report_file.write("### ULog Parse Report\n\n")
        overview_keys = [
            "resample_rate",
            "fill_strategy",
            "global_start_us",
            "output_rows",
            "output_columns",
            "output_missing_values",
        ]
        self._write_overview_table(report_file, parse_report, overview_keys)

        self._write_list(report_file, "Parsed Topics", parse_report.get("parsed_topics"))
        self._write_list(report_file, "Missing Topics", parse_report.get("missing_topics"))
        self._write_dict_table(
            report_file,
            "Resolved Alternatives",
            parse_report.get("resolved_alternatives"),
            "Requested Topic",
            "Resolved Topic",
        )
        self._write_dict_table(
            report_file,
            "Failed Topics",
            parse_report.get("failed_topics"),
            "Topic",
            "Error",
        )
        self._write_nested_dict_table(report_file, "Topic Stats", parse_report.get("topic_stats"))
        self._write_nested_dict_table(report_file, "Instance Stats", parse_report.get("instance_stats"))
        self._write_dict_table(
            report_file,
            "Fill Limits",
            parse_report.get("fill_limits"),
            "Source",
            "Limit",
        )
        self._write_list(report_file, "Sparse Sources", parse_report.get("sparse_sources"))

    def _write_csv_parse_report(self, report_file, parse_report):
        if not parse_report:
            return

        report_file.write("### CSV Parse Report\n\n")
        overview_keys = [
            "source_file",
            "config_path",
            "rows",
            "columns",
            "timestamp_column",
            "timestamp_status",
            "timestamp_unit",
            "monotonic_timestamp",
            "duplicate_timestamps",
            "numeric_column_count",
            "missing_values",
        ]
        self._write_overview_table(report_file, parse_report, overview_keys)

        self._write_dict_table(
            report_file,
            "Column Mapping",
            parse_report.get("column_mapping_applied"),
            "Source Column",
            "Normalized Column",
        )
        self._write_list(report_file, "Unmapped Config Columns", parse_report.get("unmapped_config_columns"))
        self._write_list(report_file, "Coerced Numeric Columns", parse_report.get("coerced_numeric_columns"))
        self._write_list(report_file, "All-NaN Columns", parse_report.get("all_nan_columns"))
        self._write_list(report_file, "Constant Columns", parse_report.get("constant_columns"))
        self._write_list(report_file, "Numeric Columns", parse_report.get("numeric_columns"))

    def _write_structural_break_report(self, report_file, structural_break):
        if not isinstance(structural_break, dict):
            return

        report_file.write("### Structural Break Report\n\n")
        overview_keys = [
            "status",
            "detected",
            "reason",
            "timestamp",
            "break_timestamp",
        ]
        self._write_overview_table(report_file, structural_break, overview_keys)
        self._write_list(report_file, "Detected Columns", structural_break.get("detected_columns"))
        self._write_nested_dict_table(
            report_file,
            "Break Details",
            structural_break.get("break_details"),
        )

    def _write_pca_preprocessing_report(self, report_file, preprocessing_report):
        if not preprocessing_report:
            return

        report_file.write("### PCA Preprocessing Report\n\n")
        overview_keys = [
            "status",
            "data_key",
            "input_rows",
            "input_columns",
            "numeric_column_count",
            "selected_column_count",
            "requested_n_components",
            "effective_n_components",
            "anomaly_z_threshold",
            "missing_values_before_fill",
            "missing_values_after_fill",
            "component_adjustment_reason",
            "reason",
        ]
        self._write_overview_table(report_file, preprocessing_report, overview_keys)
        self._write_list(report_file, "Selected PCA Columns", preprocessing_report.get("selected_columns"))
        self._write_list(report_file, "All-NaN Columns Excluded", preprocessing_report.get("all_nan_columns"))
        self._write_list(report_file, "Constant Columns Excluded", preprocessing_report.get("constant_columns"))
        self._write_dict_table(
            report_file,
            "Missing Values Before Fill",
            preprocessing_report.get("columns_with_missing"),
            "Column",
            "Missing",
        )

    def _write_anomaly_report(self, report_file, anomaly_config, anomaly_details):
        if not anomaly_config and not anomaly_details:
            return

        report_file.write("### PCA Anomaly Detection Report\n\n")
        if anomaly_config:
            self._write_overview_table(
                report_file,
                anomaly_config,
                ["method", "z_threshold", "comparison"],
            )
        if anomaly_details:
            rows = []
            for component, details in anomaly_details.items():
                if not details:
                    rows.append({
                        "Component": component,
                        "Timestamp": "none",
                        "Z-score": "",
                        "Score": "",
                    })
                    continue
                for detail in details:
                    rows.append({
                        "Component": component,
                        "Timestamp": detail.get("timestamp"),
                        "Z-score": detail.get("z_score"),
                        "Score": detail.get("score"),
                    })
            self._write_records(report_file, "Detected PCA Score Spikes", rows)

    def _write_flight_phase_report(self, report_file, flight_phase_report, flight_phases):
        if not flight_phase_report and (flight_phases is None or flight_phases.empty):
            return

        report_file.write("### Telemetry Flight Phases\n\n")
        if flight_phase_report:
            self._write_overview_table(
                report_file,
                flight_phase_report,
                [
                    "status",
                    "reason",
                    "rows",
                    "altitude_source",
                    "vertical_speed_source",
                    "throttle_source",
                ],
            )
            self._write_dict_table(
                report_file,
                "Phase Counts",
                flight_phase_report.get("phase_counts"),
                "Phase",
                "Count",
            )
        if flight_phases is not None and not flight_phases.empty:
            self._write_records(report_file, "Flight Phase Samples", flight_phases.head(50).to_dict("records"))

    def _write_video_report(
        self,
        report_file,
        video_report,
        video_alignment,
        video_coverage,
        video_events,
        telemetry_video_comparison,
    ):
        if not any([
            video_report,
            video_alignment,
            video_coverage,
            video_events is not None and not video_events.empty,
            telemetry_video_comparison,
        ]):
            return

        report_file.write("### Video Summary\n\n")
        if video_report:
            self._write_overview_table(
                report_file,
                video_report,
                [
                    "status",
                    "reason",
                    "file_name",
                    "video_path",
                    "camera_viewpoint",
                    "duration_s",
                    "fps",
                    "width",
                    "height",
                    "codec",
                    "frame_count",
                    "sample_interval_s",
                    "feature_rows",
                    "event_count",
                ],
            )
        if video_alignment:
            self._write_overview_table(
                report_file,
                video_alignment,
                ["mode", "video_offset_s", "confidence", "event_window_s", "formula", "camera_viewpoint"],
            )

        if video_coverage:
            report_file.write("### Video Coverage\n\n")
            self._write_overview_table(
                report_file,
                video_coverage,
                [
                    "start_elapsed_s",
                    "end_elapsed_s",
                    "duration_s",
                    "telemetry_duration_s",
                    "coverage_ratio",
                ],
            )

        if video_events is not None and not video_events.empty:
            rows = video_events.head(100).to_dict("records")
            self._write_records(report_file, "Video Events", rows)

        if telemetry_video_comparison:
            self._write_records(report_file, "Telemetry vs Video", telemetry_video_comparison)

    def _write_raw_telemetry_summary(self, report_file, raw_summary):
        if not raw_summary:
            return

        report_file.write("### Raw Telemetry Direct Summary\n\n")
        self._write_overview_table(
            report_file,
            raw_summary,
            [
                "status",
                "method",
                "rows",
                "columns",
                "numeric_columns",
                "duration_s",
            ],
        )
        self._write_dict_table(
            report_file,
            "Raw Telemetry Category Counts",
            raw_summary.get("category_counts"),
            "Category",
            "Columns",
        )
        self._write_records(report_file, "Raw Telemetry Missing Columns", raw_summary.get("missing_columns"))
        self._write_records(report_file, "Raw Telemetry Representative Column Stats", raw_summary.get("column_stats"))
        self._write_records(report_file, "Raw Telemetry Sudden Change Candidates", raw_summary.get("sudden_changes"))
        self._write_records(report_file, "Raw Telemetry Range Flags", raw_summary.get("range_flags"))

    def _write_overview_table(self, report_file, values, keys):
        rows = [
            {"Metric": key, "Value": values.get(key)}
            for key in keys
            if key in values
        ]
        self._write_records(report_file, "Overview", rows)

    def _write_dict_table(self, report_file, title, values, key_name, value_name):
        if not values:
            return
        rows = [
            {key_name: key, value_name: self._format_value(value)}
            for key, value in values.items()
        ]
        self._write_records(report_file, title, rows)

    def _write_nested_dict_table(self, report_file, title, values):
        if not values:
            return
        rows = []
        for key, nested_values in values.items():
            row = {"Name": key}
            if isinstance(nested_values, dict):
                row.update(nested_values)
            else:
                row["Value"] = nested_values
            rows.append(row)
        self._write_records(report_file, title, rows)

    def _write_list(self, report_file, title, values):
        if not values:
            return
        rows = [{"Value": value} for value in values]
        self._write_records(report_file, title, rows)

    def _write_records(self, report_file, title, rows):
        if not rows:
            return
        report_file.write(f"#### {title}\n\n")
        report_file.write(pd.DataFrame(rows).to_markdown(index=False))
        report_file.write("\n\n")

    def _format_value(self, value):
        if isinstance(value, (dict, list)):
            return str(value)
        return value
