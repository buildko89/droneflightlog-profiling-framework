import pandas as pd


class FlightPhaseAnalyzer:
    """
    Derives coarse telemetry-side flight phases for video consistency checks.
    """

    def __init__(self, context):
        self.context = context

    def analyze(self, data_key="raw_data"):
        df = self.context.get_data(data_key)
        if df is None or df.empty:
            report = {
                "status": "skipped",
                "reason": "telemetry_data_missing",
            }
            self.context.set_artifact("flight_phase_report", report)
            self.context.set_data("flight_phases", pd.DataFrame())
            return report

        altitude = self._altitude_series(df)
        vertical_speed = self._vertical_speed_series(df, altitude)
        throttle = self._throttle_series(df)
        phases = self._classify(df.index, altitude, vertical_speed, throttle)
        report = {
            "status": "completed" if not phases.empty else "skipped",
            "reason": None if not phases.empty else "insufficient_phase_signals",
            "rows": int(len(phases)),
            "altitude_source": altitude.name if altitude is not None else None,
            "vertical_speed_source": vertical_speed.name if vertical_speed is not None else None,
            "throttle_source": throttle.name if throttle is not None else None,
            "phase_counts": phases["phase"].value_counts().to_dict() if not phases.empty else {},
        }
        self.context.set_artifact("flight_phase_report", report)
        self.context.set_data("flight_phases", phases)
        self.context.add_log(f"Telemetry flight phase analysis {report['status']}: {report['phase_counts']}")
        return report

    def _classify(self, index, altitude, vertical_speed, throttle):
        if altitude is None and vertical_speed is None and throttle is None:
            return pd.DataFrame()

        elapsed = _elapsed_seconds(index)
        phases = []
        alt_start = float(altitude.iloc[0]) if altitude is not None and not altitude.empty else 0.0
        for position, time_s in enumerate(elapsed):
            alt = _value_at(altitude, position)
            vz = _value_at(vertical_speed, position)
            thr = _value_at(throttle, position)
            phase = "unknown"

            if alt is not None and abs(alt - alt_start) < 0.3 and (thr is None or thr < 0.15):
                phase = "ground"
            elif vz is not None and vz > 0.4:
                phase = "takeoff"
            elif vz is not None and vz < -0.4:
                phase = "landing"
            elif vz is not None and abs(vz) <= 0.25:
                phase = "hover"
            elif thr is not None and thr > 0.2:
                phase = "moving"

            phases.append({
                "telemetry_time_s": float(time_s),
                "phase": phase,
                "altitude": alt,
                "vertical_speed": vz,
                "throttle": thr,
            })

        return pd.DataFrame(phases)

    def _altitude_series(self, df):
        preferred_terms = [
            "vehicle_local_position_z",
            "local_position_z",
            "alt",
            "height",
        ]
        column = _find_column(df, preferred_terms)
        if column is None:
            return None
        series = pd.to_numeric(df[column], errors="coerce").ffill().bfill()
        if "local_position_z" in column.lower() or column.lower().endswith("_z"):
            series = -series
        series.name = column
        return series

    def _vertical_speed_series(self, df, altitude):
        column = _find_column(df, ["vehicle_local_position_vz", "local_position_vz", "velocity_z", "vel_z"])
        if column is not None:
            series = pd.to_numeric(df[column], errors="coerce").ffill().bfill()
            if "vz" in column.lower() or column.lower().endswith("_z"):
                series = -series
            series.name = column
            return series

        if altitude is None or len(altitude) < 2:
            return None
        elapsed = pd.Series(_elapsed_seconds(df.index), index=df.index)
        delta_t = elapsed.diff().replace(0, pd.NA)
        series = altitude.diff() / delta_t
        series = series.fillna(0.0)
        series.name = f"{altitude.name}_derived_vertical_speed"
        return series

    def _throttle_series(self, df):
        columns = [
            column for column in df.columns
            if any(term in column.lower() for term in ["actuator_outputs_output", "throttle"])
        ]
        if not columns:
            return None
        numeric = df[columns].apply(pd.to_numeric, errors="coerce").ffill().bfill()
        series = numeric.mean(axis=1)
        span = float(series.max() - series.min()) if len(series) else 0.0
        if span > 0:
            series = (series - series.min()) / span
        series.name = "throttle_estimate"
        return series


def _find_column(df, terms):
    for term in terms:
        for column in df.columns:
            if term in column.lower():
                return column
    return None


def _elapsed_seconds(index):
    if hasattr(index, "total_seconds"):
        return list(index.total_seconds())
    return [float(value) for value in index]


def _value_at(series, position):
    if series is None or position >= len(series):
        return None
    value = series.iloc[position]
    if pd.isna(value):
        return None
    return float(value)
