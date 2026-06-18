import os
from datetime import datetime

import pandas as pd


class VideoOnlyReportExporter:
    """
    Markdown exporter for video-only analysis.
    """

    def __init__(self, context, output_dir="output"):
        self.context = context
        self.output_dir = output_dir

    def export_markdown(self, filename="video_analysis_report.md"):
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, "w", encoding="utf-8") as report_file:
            report_file.write("# 動画単体解析レポート\n\n")
            report_file.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            self._write_executive_summary(report_file)
            self._write_video_metadata(report_file)
            self._write_feature_summary(report_file)
            self._write_video_events(report_file)
            self._write_ai_notes(report_file)
            self._write_warnings(report_file)
            self._write_appendix(report_file)

        self.context.add_output("report", output_path)
        self.context.add_log(f"Video-only report exported to: {output_path}")
        return output_path

    def _write_executive_summary(self, report_file):
        parse_report = self.context.get_artifact("video_parse_report", {})
        feature_summary = self.context.get_artifact("video_feature_summary", {})
        events = self.context.get_data("video_events")

        report_file.write("## 概要\n\n")
        report_file.write(f"- 動画解析ステータス: {self._translate_status(parse_report.get('status', 'not_run'))}\n")
        if parse_report.get("reason"):
            report_file.write(f"- 補足: {self._translate_reason(parse_report['reason'])}\n")
        report_file.write(f"- カメラ視点: {self._translate_viewpoint(parse_report.get('camera_viewpoint', 'external'))}\n")
        report_file.write(f"- 動画長: {parse_report.get('duration_s', 'N/A')} 秒\n")
        report_file.write(f"- サンプリング済みフレーム数: {feature_summary.get('sample_count', 0)}\n")
        report_file.write(f"- 推定イベント数: {0 if events is None else len(events)}\n")
        report_file.write("\n")

    def _write_video_metadata(self, report_file):
        parse_report = self.context.get_artifact("video_parse_report")
        alignment = self.context.get_artifact("video_alignment")
        coverage = self.context.get_artifact("video_coverage")

        report_file.write("## 動画メタデータ\n\n")
        if parse_report:
            self._write_overview_table(
                report_file,
                parse_report,
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
                    "size_bytes",
                ],
            )
        else:
            report_file.write("動画メタデータは生成されませんでした。\n\n")

        if alignment:
            report_file.write("## 動画同期情報\n\n")
            self._write_overview_table(
                report_file,
                alignment,
                ["mode", "video_offset_s", "confidence", "event_window_s", "formula", "camera_viewpoint"],
            )
        if coverage:
            report_file.write("## カバレッジ\n\n")
            self._write_overview_table(
                report_file,
                coverage,
                ["status", "reason", "start_elapsed_s", "end_elapsed_s", "duration_s", "telemetry_duration_s", "coverage_ratio"],
            )

    def _write_feature_summary(self, report_file):
        feature_summary = self.context.get_artifact("video_feature_summary")
        features = self.context.get_data("video_features")

        report_file.write("## 特徴量統計\n\n")
        if feature_summary:
            self._write_overview_table(
                report_file,
                feature_summary,
                [
                    "status",
                    "reason",
                    "sample_count",
                    "brightness_mean",
                    "brightness_min",
                    "brightness_max",
                    "blur_score_mean",
                    "blur_score_min",
                    "blur_score_max",
                    "frame_diff_mean",
                    "frame_diff_min",
                    "frame_diff_max",
                    "motion_score_mean",
                    "motion_score_min",
                    "motion_score_max",
                    "frame_visible_ratio",
                    "motion_direction_counts",
                ],
            )
        else:
            report_file.write("特徴量サマリーは生成されませんでした。\n\n")

        if features is not None and not features.empty:
            self._write_records(report_file, "特徴量サンプル", features.head(50).to_dict("records"))

    def _write_video_events(self, report_file):
        events = self.context.get_data("video_events")
        report_file.write("## 動画イベント\n\n")
        if events is None or events.empty:
            report_file.write("動画イベントは推定されませんでした。\n\n")
            return
        self._write_records(report_file, "イベント一覧", events.to_dict("records"))

    def _write_ai_notes(self, report_file):
        diagnosis = self.context.get_data("video_llm_diagnosis")
        report_file.write("## AIによる解釈\n\n")
        if diagnosis is None or diagnosis.empty:
            report_file.write("AIによる解釈は生成されませんでした。\n\n")
            return
        text = str(diagnosis.iloc[0].get("interpretation", ""))
        report_file.write(text)
        report_file.write("\n\n")

    def _write_warnings(self, report_file):
        report_file.write("## 警告\n\n")
        warnings = self.context.get_warnings()
        if not warnings:
            report_file.write("警告はありません。\n\n")
            return
        for warning in warnings:
            report_file.write(f"- {self._translate_warning(warning)}\n")
        report_file.write("\n")

    def _write_appendix(self, report_file):
        report_file.write("## 付録\n\n")
        report_file.write("### 解析ログ\n\n")
        for log in self.context.get_logs():
            report_file.write(f"- {self._translate_log(log)}\n")
        report_file.write("\n")

    def _write_overview_table(self, report_file, values, keys):
        rows = [
            {"項目": self._translate_key(key), "値": self._format_value(values.get(key), key)}
            for key in keys
            if key in values
        ]
        self._write_records(report_file, "一覧", rows)

    def _write_records(self, report_file, title, rows):
        if not rows:
            return
        report_file.write(f"### {title}\n\n")
        report_file.write(pd.DataFrame(self._translate_rows(rows)).to_markdown(index=False))
        report_file.write("\n\n")

    def _format_value(self, value, key=None):
        if isinstance(value, (dict, list)):
            return str(value)
        if key == "camera_viewpoint":
            return self._translate_viewpoint(value)
        if key == "status":
            return self._translate_status(value)
        if key == "reason":
            return self._translate_reason(value)
        return value

    def _translate_rows(self, rows):
        translated = []
        for row in rows:
            translated.append({
                self._translate_key(key): self._format_value(value, key)
                for key, value in row.items()
            })
        return translated

    def _translate_key(self, key):
        labels = {
            "Metric": "項目",
            "Value": "値",
            "status": "ステータス",
            "reason": "理由",
            "file_name": "ファイル名",
            "video_path": "動画パス",
            "camera_viewpoint": "カメラ視点",
            "duration_s": "動画長(秒)",
            "fps": "FPS",
            "width": "幅",
            "height": "高さ",
            "codec": "コーデック",
            "frame_count": "フレーム数",
            "sample_interval_s": "サンプリング間隔(秒)",
            "feature_rows": "特徴量行数",
            "event_count": "イベント数",
            "size_bytes": "サイズ(bytes)",
            "mode": "方式",
            "video_offset_s": "動画オフセット秒",
            "confidence": "信頼度",
            "event_window_s": "イベント照合窓(秒)",
            "formula": "同期式",
            "start_elapsed_s": "開始経過秒",
            "end_elapsed_s": "終了経過秒",
            "telemetry_duration_s": "ログ長(秒)",
            "coverage_ratio": "カバレッジ率",
            "sample_count": "サンプル数",
            "brightness_mean": "平均輝度",
            "brightness_min": "最小輝度",
            "brightness_max": "最大輝度",
            "blur_score_mean": "平均ブレスコア",
            "blur_score_min": "最小ブレスコア",
            "blur_score_max": "最大ブレスコア",
            "frame_diff_mean": "平均フレーム差分",
            "frame_diff_min": "最小フレーム差分",
            "frame_diff_max": "最大フレーム差分",
            "motion_score_mean": "平均モーション量",
            "motion_score_min": "最小モーション量",
            "motion_score_max": "最大モーション量",
            "frame_visible_ratio": "視認可能フレーム比率",
            "motion_direction_counts": "モーション方向カウント",
            "video_time_s": "動画時刻(秒)",
            "brightness": "輝度",
            "blur_score": "ブレスコア",
            "frame_diff": "フレーム差分",
            "motion_score": "モーション量",
            "motion_dx": "モーションX",
            "motion_dy": "モーションY",
            "motion_direction": "モーション方向",
            "frame_visible": "視認可否",
            "event": "イベント",
            "confidence": "信頼度",
        }
        return labels.get(str(key), key)

    def _translate_status(self, value):
        labels = {
            "completed": "完了",
            "skipped": "スキップ",
            "not_run": "未実行",
            "not_applicable": "対象外",
        }
        return labels.get(str(value), value)

    def _translate_reason(self, value):
        labels = {
            "video_only_analysis": "動画単体解析のためログカバレッジは対象外",
            "video_features_empty": "動画特徴量が空です",
            "opencv_not_installed": "OpenCV がインストールされていません",
            "video_open_failed": "動画ファイルを開けませんでした",
            "video_not_provided": "動画ファイルが指定されていません",
        }
        text = str(value)
        if text.startswith("video_not_found:"):
            return text.replace("video_not_found:", "動画ファイルが見つかりません:")
        return labels.get(text, value)

    def _translate_viewpoint(self, value):
        labels = {
            "external": "外部カメラ",
            "onboard": "機体搭載カメラ",
        }
        return labels.get(str(value), value)

    def _translate_warning(self, value):
        text = str(value)
        if text.startswith("Video analysis skipped:"):
            reason = text.replace("Video analysis skipped:", "").strip()
            return f"動画解析をスキップしました: {self._translate_reason(reason)}"
        return text

    def _translate_log(self, value):
        labels = {
            "Video-only pipeline started.": "動画単体解析パイプラインを開始しました。",
            "Video-only pipeline completed.": "動画単体解析パイプラインが完了しました。",
        }
        text = str(value)
        if text.startswith("WARNING: Video analysis skipped:"):
            reason = text.replace("WARNING: Video analysis skipped:", "").strip()
            return f"警告: 動画解析をスキップしました: {self._translate_reason(reason)}"
        if text.startswith("Video-only report exported to:"):
            path = text.replace("Video-only report exported to:", "").strip()
            return f"動画単体解析レポートを出力しました: {path}"
        if text.startswith("Video-only interpretation saved to:"):
            path = text.replace("Video-only interpretation saved to:", "").strip()
            return f"動画単体AI診断を保存しました: {path}"
        return labels.get(text, text)
