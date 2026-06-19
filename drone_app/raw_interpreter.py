import os

import numpy as np
import pandas as pd

from profilecore.core.module import AnalysisModule


class RawTelemetryInterpreter(AnalysisModule):
    """
    LLM interpretation based directly on raw telemetry statistics.
    """

    CATEGORY_TERMS = {
        "attitude": ["attitude", "roll", "pitch", "yaw", "q["],
        "altitude": ["alt", "height", "local_position_z", "global_position_alt"],
        "velocity": ["velocity", "vel_", "_vx", "_vy", "_vz", "local_position_v"],
        "battery": ["battery", "voltage", "current", "remaining"],
        "motor_actuator": ["actuator", "output", "motor", "throttle", "pwm"],
        "imu_accel": ["accelerometer", "accel", "acceleration"],
        "imu_gyro": ["gyro", "angular_velocity"],
        "gps": ["gps", "global_position", "satellites", "hdop", "eph", "epv"],
    }

    def __init__(self, context, llm_client):
        super().__init__(context)
        self.llm_client = llm_client

    def run_interpretation(self, output_file="output/raw_telemetry_diagnosis.md"):
        df = self.context.get_data("raw_data")
        if df is None or df.empty:
            self.log("Error: raw_data not found in context.")
            return False

        summary = self._build_summary(df)
        summary["flight_phase_report"] = self.context.get_artifact("flight_phase_report")
        flight_phases = self.context.get_data("flight_phases")
        if flight_phases is not None and not flight_phases.empty:
            summary["flight_phase_samples"] = flight_phases.head(30).to_dict("records")

        self.context.set_artifact("raw_telemetry_summary", summary)
        self.context.set_data("raw_telemetry_column_stats", pd.DataFrame(summary["column_stats"]))
        self.context.set_data("raw_telemetry_sudden_changes", pd.DataFrame(summary["sudden_changes"]))
        self.context.set_data("raw_telemetry_range_flags", pd.DataFrame(summary["range_flags"]))

        prompt = self._create_prompt(summary)
        mode = self.context.settings.get("llm_mode", "api")
        if mode == "export":
            return self._export_prompt(prompt, output_file)

        try:
            diagnosis_text = self.llm_client.generate_text(prompt)
            self.log("Raw telemetry LLM interpretation generated successfully.")
        except Exception as exc:
            self.log(f"Error calling LLM client: {str(exc)}")
            return False

        client_type = self.llm_client.__class__.__name__.replace("Client", "")
        model_name = self.llm_client.model_name
        full_diagnosis = (
            "# LLMによる生テレメトリ直接診断\n\n"
            f"- **使用AI**: {client_type}\n"
            f"- **使用モデル**: {model_name}\n"
            "- **解析モード**: raw_telemetry\n\n"
            f"{diagnosis_text}"
        )
        return self._write_diagnosis(full_diagnosis, output_file)

    def _build_summary(self, df):
        numeric_df = df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        numeric_df = numeric_df.dropna(axis=1, how="all")
        duration_s = _duration_seconds(df.index)
        columns = list(numeric_df.columns)

        column_stats = []
        for column in columns:
            series = numeric_df[column]
            clean = series.dropna()
            if clean.empty:
                continue
            column_stats.append({
                "column": column,
                "category": self._category_for(column),
                "count": int(clean.count()),
                "missing": int(series.isna().sum()),
                "missing_ratio": _round(series.isna().mean()),
                "mean": _round(clean.mean()),
                "std": _round(clean.std()),
                "min": _round(clean.min()),
                "p05": _round(clean.quantile(0.05)),
                "median": _round(clean.median()),
                "p95": _round(clean.quantile(0.95)),
                "max": _round(clean.max()),
                "start": _round(clean.iloc[0]),
                "end": _round(clean.iloc[-1]),
                "trend": _round(clean.iloc[-1] - clean.iloc[0]),
            })

        missing_rows = sorted(
            (
                {
                    "column": column,
                    "missing": int(df[column].isna().sum()),
                    "missing_ratio": _round(df[column].isna().mean()),
                }
                for column in df.columns
                if int(df[column].isna().sum()) > 0
            ),
            key=lambda row: row["missing_ratio"],
            reverse=True,
        )[:20]

        sudden_changes = self._detect_sudden_changes(numeric_df)
        range_flags = self._detect_range_flags(numeric_df)

        categories = {}
        for row in column_stats:
            categories.setdefault(row["category"], 0)
            categories[row["category"]] += 1

        return {
            "status": "completed",
            "method": "raw_telemetry_direct",
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "numeric_columns": int(len(columns)),
            "duration_s": _round(duration_s),
            "category_counts": categories,
            "missing_columns": missing_rows,
            "column_stats": self._select_representative_stats(column_stats),
            "sudden_changes": sudden_changes,
            "range_flags": range_flags,
        }

    def _select_representative_stats(self, rows):
        ranked = sorted(
            rows,
            key=lambda row: (
                row["category"] == "other",
                -abs(row["trend"] or 0.0),
                -((row["std"] or 0.0)),
            ),
        )
        return ranked[:40]

    def _detect_sudden_changes(self, numeric_df):
        rows = []
        elapsed = _elapsed_seconds(numeric_df.index)
        for column in numeric_df.columns:
            clean = numeric_df[column].ffill().bfill()
            if clean.dropna().shape[0] < 3:
                continue
            diffs = clean.diff().abs()
            baseline = diffs.median()
            spread = diffs.mad() if hasattr(diffs, "mad") else (diffs - baseline).abs().median()
            threshold = baseline + max(spread, 1e-9) * 8.0
            candidates = diffs[diffs > threshold].dropna().sort_values(ascending=False).head(3)
            for index_value, delta in candidates.items():
                position = numeric_df.index.get_loc(index_value)
                if isinstance(position, slice):
                    position = position.start
                rows.append({
                    "column": column,
                    "category": self._category_for(column),
                    "time_s": _round(elapsed[int(position)] if position is not None else None),
                    "delta": _round(delta),
                    "previous": _round(clean.shift(1).loc[index_value]),
                    "current": _round(clean.loc[index_value]),
                })
        return sorted(rows, key=lambda row: abs(row["delta"] or 0.0), reverse=True)[:30]

    def _detect_range_flags(self, numeric_df):
        flags = []
        for column in numeric_df.columns:
            series = numeric_df[column].dropna()
            if series.empty:
                continue
            lowered = column.lower()
            minimum = float(series.min())
            maximum = float(series.max())
            mean = float(series.mean())

            rules = []
            if "battery" in lowered and "voltage" in lowered:
                rules.append(("battery_voltage_low", minimum < 9.0, f"min={minimum:.3f}"))
                rules.append(("battery_voltage_high", maximum > 26.0, f"max={maximum:.3f}"))
            if "remaining" in lowered:
                rules.append(("battery_remaining_low", minimum < 0.2, f"min={minimum:.3f}"))
            if "satellites" in lowered or "satellite" in lowered:
                rules.append(("gps_satellite_count_low", minimum < 6.0, f"min={minimum:.3f}"))
            if "hdop" in lowered or "eph" in lowered or "epv" in lowered:
                rules.append(("gps_accuracy_degraded", maximum > 2.5, f"max={maximum:.3f}"))
            if "accelerometer" in lowered or "accel" in lowered:
                rules.append(("acceleration_large", maximum > 30.0 or minimum < -30.0, f"range={minimum:.3f}..{maximum:.3f}"))
            if "gyro" in lowered or "angular_velocity" in lowered:
                rules.append(("angular_velocity_large", maximum > 8.0 or minimum < -8.0, f"range={minimum:.3f}..{maximum:.3f}"))
            if "actuator" in lowered or "output" in lowered or "pwm" in lowered:
                if 0.0 <= mean <= 1.5:
                    rules.append(("actuator_normalized_out_of_range", minimum < -0.05 or maximum > 1.05, f"range={minimum:.3f}..{maximum:.3f}"))
                else:
                    rules.append(("actuator_pwm_out_of_range", minimum < 900.0 or maximum > 2200.0, f"range={minimum:.3f}..{maximum:.3f}"))

            for rule, detected, detail in rules:
                if detected:
                    flags.append({
                        "column": column,
                        "category": self._category_for(column),
                        "rule": rule,
                        "detail": detail,
                    })
        return flags[:40]

    def _create_prompt(self, summary):
        prompt = f"""
あなたはドローンのテレメトリデータ解析のエキスパートです。
以下は PCA や主成分スコアを使わず、PX4 ULog または CSV から得た生テレメトリ列を直接集計した結果です。
列名はログ設定やCSVマッピングで意味が変わる可能性があるため、断定しすぎず、数値系列から読み取れる範囲で日本語診断してください。

### データ概要
- 解析方式: {summary['method']}
- 行数/列数: {summary['rows']} / {summary['columns']}
- 数値列数: {summary['numeric_columns']}
- 推定ログ時間: {summary['duration_s']} 秒
- カテゴリ別列数: {summary['category_counts']}

### 欠損の多い列
{summary['missing_columns']}

### 代表的な列統計
{summary['column_stats']}

### 急変候補
{summary['sudden_changes']}

### 範囲外・注意候補
{summary['range_flags']}
"""
        flight_phase_report = summary.get("flight_phase_report")
        if flight_phase_report:
            prompt += f"\n### フライトフェーズ推定\n{flight_phase_report}\n"
        if summary.get("flight_phase_samples"):
            prompt += f"\n### フライトフェーズサンプル\n{summary['flight_phase_samples']}\n"

        prompt += """
### 依頼内容
- PCAではなく、生ログ統計だけに基づいて、姿勢、高度、速度、IMU、GPS、バッテリー、モーター/アクチュエータの観点から診断してください。
- 急変候補がある場合は、該当時刻、列名、変化量から、操縦入力、外乱、接地、フェーズ遷移、センサー欠損補完など複数の可能性を挙げてください。
- 範囲外・注意候補は、列名から推測できる範囲で保守点検や追加確認項目につなげてください。
- 欠損や列不足により判断できない点を明示してください。
- 確定診断ではなく、元系列グラフ、該当時刻前後、機体ログイベントとの照合が必要であることを明記してください。
- Markdown形式で、要約、観点別診断、注意時刻、保守推奨、追加確認項目の順に整理してください。
"""
        return prompt

    def _export_prompt(self, prompt, output_file):
        workspace_dir = getattr(self.context, "workspace_dir", "workspace")
        prompt_file = os.path.join(workspace_dir, "raw_telemetry_llm_prompt.txt")
        try:
            os.makedirs(os.path.dirname(prompt_file), exist_ok=True)
            with open(prompt_file, "w", encoding="utf-8") as file_obj:
                file_obj.write(prompt)
        except Exception as exc:
            self.log(f"Error exporting raw telemetry prompt to {prompt_file}: {str(exc)}")
            return False

        model_name = self.llm_client.model_name
        instructions = f"""# 生テレメトリ直接診断プロンプト出力

- **想定モデル**: {model_name}
- **解析モード**: raw_telemetry
- **ステータス**: プロンプト出力済み

PCAを使わないログ直接診断用プロンプトを出力しました。

- **プロンプトファイルパス**: `{prompt_file}`
"""
        return self._write_diagnosis(instructions, output_file)

    def _write_diagnosis(self, text, output_file):
        text_df = pd.DataFrame({"interpretation": [text]})
        self.context.set_data("llm_diagnosis", text_df)
        self.context.set_data("raw_llm_diagnosis", text_df)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as file_obj:
            file_obj.write(text)
        self.log(f"Raw telemetry interpretation saved to: {output_file}")
        return True

    def _category_for(self, column):
        lowered = str(column).lower()
        for category, terms in self.CATEGORY_TERMS.items():
            if any(term in lowered for term in terms):
                return category
        return "other"


def _elapsed_seconds(index):
    if hasattr(index, "total_seconds"):
        return list(index.total_seconds())
    values = []
    for value in index:
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            values.append(float(len(values)))
    return values


def _duration_seconds(index):
    elapsed = _elapsed_seconds(index)
    if len(elapsed) < 2:
        return 0.0
    return float(elapsed[-1] - elapsed[0])


def _round(value, digits=6):
    if value is None or pd.isna(value):
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return round(value, digits)
