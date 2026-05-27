import json
import os

import pandas as pd


DEFAULT_TIMESTAMP_CANDIDATES = [
    "timestamp",
    "time",
    "time_s",
    "time_sec",
    "time_seconds",
    "time_ms",
    "elapsed_time",
    "elapsed",
    "datetime",
    "date_time",
]


class CsvTelemetryLoader:
    """
    Loads telemetry CSV files and normalizes them to the DataFrame shape used by
    the drone analysis pipeline.
    """
    def __init__(self, csv_file_path, config_path=None):
        self.csv_file_path = csv_file_path
        self.config_path = config_path
        self.parse_report = {}
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"CSV file not found: {csv_file_path}")

    def load(self):
        config = self._load_config()
        encoding = config.get("encoding")
        df = pd.read_csv(self.csv_file_path, encoding=encoding) if encoding else pd.read_csv(self.csv_file_path)
        original_columns = list(df.columns)

        timestamp_column = self._resolve_timestamp_column(df, config)
        timestamp_status = "not_found"
        timestamp_unit = config.get("timestamp_unit")
        duplicate_timestamps = 0
        monotonic_timestamp = None

        if timestamp_column:
            index, timestamp_status, timestamp_unit = self._build_timedelta_index(
                df[timestamp_column],
                timestamp_column,
                timestamp_unit,
            )
            duplicate_timestamps = int(index.duplicated().sum())
            monotonic_timestamp = bool(index.is_monotonic_increasing)
            df = df.drop(columns=[timestamp_column])
            df.index = index
            df.index.name = "timestamp"

            if duplicate_timestamps:
                df = df.groupby(level=0).mean(numeric_only=True)
        else:
            df.index = pd.to_timedelta(range(len(df)), unit="s")
            df.index.name = "timestamp"

        df, applied_mapping, unmapped_config_columns = self._apply_column_mapping(df, config)
        df, coerced_numeric_columns = self._coerce_numeric_columns(df)

        numeric_columns = list(df.select_dtypes(include=["number"]).columns)
        all_nan_columns = [
            column for column in numeric_columns
            if df[column].isna().all()
        ]
        constant_columns = [
            column for column in numeric_columns
            if df[column].dropna().nunique() <= 1
        ]

        self.parse_report = {
            "source_file": self.csv_file_path,
            "config_path": self.config_path,
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "original_columns": original_columns,
            "timestamp_column": timestamp_column,
            "timestamp_status": timestamp_status,
            "timestamp_unit": timestamp_unit,
            "monotonic_timestamp": monotonic_timestamp,
            "duplicate_timestamps": duplicate_timestamps,
            "column_mapping_applied": applied_mapping,
            "unmapped_config_columns": unmapped_config_columns,
            "numeric_columns": numeric_columns,
            "numeric_column_count": int(len(numeric_columns)),
            "coerced_numeric_columns": coerced_numeric_columns,
            "all_nan_columns": all_nan_columns,
            "constant_columns": constant_columns,
            "missing_values": int(df.isna().sum().sum()),
        }
        return df

    def get_parse_report(self):
        return self.parse_report

    def _load_config(self):
        if not self.config_path:
            return {}
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"CSV mapping config not found: {self.config_path}")
        with open(self.config_path, encoding="utf-8") as config_file:
            return json.load(config_file)

    def _resolve_timestamp_column(self, df, config):
        configured = config.get("timestamp") or config.get("timestamp_column")
        if configured:
            if configured not in df.columns:
                raise ValueError(f"Configured timestamp column not found in CSV: {configured}")
            return configured

        if len(df.columns) > 0:
            first_column = df.columns[0]
            if str(first_column).lower().startswith("unnamed:"):
                return first_column

        lower_to_actual = {str(column).lower(): column for column in df.columns}
        for candidate in DEFAULT_TIMESTAMP_CANDIDATES:
            if candidate.lower() in lower_to_actual:
                return lower_to_actual[candidate.lower()]

        for column in df.columns:
            lowered = str(column).lower()
            if "time" in lowered or "date" in lowered:
                return column
        return None

    def _build_timedelta_index(self, series, column_name, timestamp_unit):
        if pd.api.types.is_numeric_dtype(series):
            unit = timestamp_unit or self._infer_numeric_time_unit(column_name)
            return pd.to_timedelta(series, unit=unit), "numeric", unit

        try:
            return pd.to_timedelta(series), "timedelta", timestamp_unit
        except (TypeError, ValueError):
            parsed = pd.to_datetime(series, errors="raise")
            return parsed - parsed.iloc[0], "datetime", timestamp_unit

    def _infer_numeric_time_unit(self, column_name):
        lowered = str(column_name).lower()
        if "usec" in lowered or "micro" in lowered or lowered.endswith("_us"):
            return "us"
        if "msec" in lowered or "millis" in lowered or lowered.endswith("_ms"):
            return "ms"
        if lowered.endswith("_ns"):
            return "ns"
        return "s"

    def _apply_column_mapping(self, df, config):
        mapping = config.get("columns") or config.get("column_mapping") or {}
        if not mapping:
            return df, {}, []

        rename_map = {}
        unmapped = []
        for source, target in mapping.items():
            if source in df.columns:
                rename_map[source] = target
            else:
                unmapped.append(source)
        return df.rename(columns=rename_map), rename_map, unmapped

    def _coerce_numeric_columns(self, df):
        coerced = []
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                continue
            converted = pd.to_numeric(df[column], errors="coerce")
            original_non_null = int(df[column].notna().sum())
            converted_non_null = int(converted.notna().sum())
            if original_non_null > 0 and converted_non_null / original_non_null >= 0.8:
                df[column] = converted
                coerced.append(column)
        return df, coerced
