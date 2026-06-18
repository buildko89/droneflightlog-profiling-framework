import os
from typing import Any, Dict, List, Optional

import pandas as pd


class VideoAnalyzer:
    """
    Local video analysis kept separate from telemetry PCA.

    OpenCV is optional. If it is unavailable, the analyzer records a skipped
    report instead of failing the telemetry pipeline.
    """

    def __init__(self, context):
        self.context = context

    def analyze(
        self,
        video_path: str,
        telemetry_df: Optional[pd.DataFrame] = None,
        *,
        video_offset_s: float = 0.0,
        camera_viewpoint: str = "external",
        alignment_confidence: float = 0.5,
        sample_interval_s: float = 1.0,
        event_window_s: float = 1.0,
    ) -> Dict[str, Any]:
        camera_viewpoint = camera_viewpoint if camera_viewpoint in {"external", "onboard"} else "external"
        alignment_confidence = _clamp(float(alignment_confidence), 0.0, 1.0)
        alignment = {
            "mode": "manual_offset",
            "video_offset_s": float(video_offset_s),
            "confidence": alignment_confidence,
            "event_window_s": float(event_window_s),
            "formula": "telemetry_time_s = video_time_s + video_offset_s",
            "camera_viewpoint": camera_viewpoint,
        }
        self.context.set_artifact("video_alignment", alignment)

        if not video_path:
            return self._set_skipped("video_not_provided", alignment, telemetry_df)
        if not os.path.exists(video_path):
            return self._set_skipped(f"video_not_found: {video_path}", alignment, telemetry_df)

        try:
            import cv2
        except ImportError:
            return self._set_skipped("opencv_not_installed", alignment, telemetry_df, video_path=video_path)

        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            return self._set_skipped("video_open_failed", alignment, telemetry_df, video_path=video_path)

        try:
            metadata = self._read_metadata(capture, video_path)
            features = self._extract_features(capture, metadata, sample_interval_s, cv2)
        finally:
            capture.release()

        events = self._infer_events(features)
        coverage = self._build_coverage(metadata, telemetry_df, video_offset_s)
        feature_summary = self._build_feature_summary(features)
        comparisons = self._compare_with_telemetry(
            self.context.get_artifact("anomaly_details"),
            events,
            coverage,
            event_window_s,
            alignment_confidence,
        )
        comparisons.extend(self._compare_phases_with_video(
            self.context.get_data("flight_phases"),
            events,
            coverage,
            event_window_s,
            alignment_confidence,
        ))

        parse_report = {
            "status": "completed",
            "video_path": video_path,
            "file_name": os.path.basename(video_path),
            "camera_viewpoint": camera_viewpoint,
            "sample_interval_s": float(sample_interval_s),
            "feature_rows": int(len(features)),
            "event_count": int(len(events)),
            **metadata,
        }

        self.context.set_artifact("video_parse_report", parse_report)
        self.context.set_artifact("video_coverage", coverage)
        self.context.set_artifact("video_feature_summary", feature_summary)
        self.context.set_artifact("telemetry_video_comparison", comparisons)
        self.context.set_data("video_features", features)
        self.context.set_data("video_events", pd.DataFrame(events))
        self.context.add_log(
            f"Video analysis completed: {len(features)} sampled frames, {len(events)} inferred events."
        )
        return {
            "video_parse_report": parse_report,
            "video_alignment": alignment,
            "video_coverage": coverage,
            "video_feature_summary": feature_summary,
            "telemetry_video_comparison": comparisons,
        }

    def _set_skipped(self, reason, alignment, telemetry_df, video_path=None):
        parse_report = {
            "status": "skipped",
            "reason": reason,
        }
        if video_path:
            parse_report["video_path"] = video_path
            parse_report["file_name"] = os.path.basename(video_path)
        coverage = self._build_coverage({"duration_s": 0.0}, telemetry_df, alignment["video_offset_s"])
        feature_summary = self._build_feature_summary(pd.DataFrame())
        self.context.set_artifact("video_parse_report", parse_report)
        self.context.set_artifact("video_coverage", coverage)
        self.context.set_artifact("video_feature_summary", feature_summary)
        self.context.set_artifact("telemetry_video_comparison", [])
        self.context.set_data("video_features", pd.DataFrame())
        self.context.set_data("video_events", pd.DataFrame())
        if hasattr(self.context, "add_warning"):
            self.context.add_warning(f"Video analysis skipped: {reason}")
        return {
            "video_parse_report": parse_report,
            "video_alignment": alignment,
            "video_coverage": coverage,
            "video_feature_summary": feature_summary,
            "telemetry_video_comparison": [],
        }

    def _read_metadata(self, capture, video_path):
        fps = float(capture.get(5) or 0.0)
        frame_count = int(capture.get(7) or 0)
        width = int(capture.get(3) or 0)
        height = int(capture.get(4) or 0)
        duration_s = frame_count / fps if fps > 0 else 0.0
        codec_int = int(capture.get(6) or 0)
        codec = "".join(chr((codec_int >> 8 * i) & 0xFF) for i in range(4)).strip()
        size_bytes = os.path.getsize(video_path) if os.path.exists(video_path) else None
        return {
            "duration_s": float(duration_s),
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "codec": codec,
            "size_bytes": size_bytes,
        }

    def _extract_features(self, capture, metadata, sample_interval_s, cv2):
        fps = metadata.get("fps") or 0.0
        frame_count = metadata.get("frame_count") or 0
        if fps <= 0 or frame_count <= 0:
            return pd.DataFrame()

        step = max(1, int(round(fps * max(sample_interval_s, 0.1))))
        rows = []
        previous_gray = None
        previous_time_s = None

        for frame_index in range(0, frame_count, step):
            capture.set(1, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            video_time_s = frame_index / fps
            brightness = float(gray.mean())
            blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            frame_diff = 0.0
            motion_score = 0.0
            motion_dx = 0.0
            motion_dy = 0.0

            if previous_gray is not None:
                diff = cv2.absdiff(gray, previous_gray)
                frame_diff = float(diff.mean())
                shift, response = cv2.phaseCorrelate(
                    previous_gray.astype("float32"),
                    gray.astype("float32"),
                )
                elapsed = max(video_time_s - previous_time_s, 1e-6)
                motion_dx = float(shift[0] / elapsed)
                motion_dy = float(shift[1] / elapsed)
                motion_score = float(((motion_dx ** 2 + motion_dy ** 2) ** 0.5) * max(response, 0.0))

            rows.append({
                "video_time_s": float(video_time_s),
                "brightness": brightness,
                "blur_score": blur_score,
                "frame_diff": frame_diff,
                "motion_score": motion_score,
                "motion_dx": motion_dx,
                "motion_dy": motion_dy,
                "motion_direction": self._motion_direction(motion_dx, motion_dy),
                "frame_visible": bool(15.0 <= brightness <= 245.0 and blur_score >= 5.0),
            })
            previous_gray = gray
            previous_time_s = video_time_s

        return pd.DataFrame(rows)

    def _infer_events(self, features):
        if features.empty:
            return []

        motion_threshold = max(
            2.0,
            float(features["motion_score"].median() + features["motion_score"].std(ddof=0) * 2.0),
        )
        low_motion_threshold = max(0.5, float(features["motion_score"].median() * 0.5))
        blur_threshold = max(5.0, float(features["blur_score"].quantile(0.10)))
        events = []

        for _, row in features.iterrows():
            time_s = float(row["video_time_s"])
            if not bool(row["frame_visible"]):
                events.append(self._event(time_s, "visibility_loss", 0.6))
            if float(row["blur_score"]) <= blur_threshold:
                events.append(self._event(time_s, "severe_blur", 0.55))
            if float(row["motion_score"]) >= motion_threshold:
                events.append(self._event(time_s, "rapid_movement", 0.6))
                if row["motion_direction"] in {"left", "right"}:
                    events.append(self._event(time_s, "lateral_motion", 0.45))
                elif row["motion_direction"] in {"up", "down"}:
                    events.append(self._event(time_s, "forward_motion", 0.45))
            elif float(row["motion_score"]) <= low_motion_threshold and bool(row["frame_visible"]):
                events.append(self._event(time_s, "hover", 0.45))

        return self._deduplicate_events(events)

    def _compare_with_telemetry(self, anomaly_details, events, coverage, event_window_s, alignment_confidence=0.5):
        rows = []
        if not anomaly_details:
            return rows

        for component, details in anomaly_details.items():
            for detail in details or []:
                timestamp = detail.get("timestamp")
                telemetry_time_s = _parse_timestamp_s(timestamp)
                if telemetry_time_s is None:
                    continue
                if not _is_in_coverage(telemetry_time_s, coverage):
                    rows.append({
                        "Time": timestamp,
                        "Log": f"{component} PCA Spike",
                        "Video": "N/A",
                        "Result": "No Coverage",
                        "Alignment Confidence": alignment_confidence,
                        "Tolerance Window s": event_window_s,
                        "Comment": "動画による裏付けなし",
                    })
                    continue

                video_time_s = telemetry_time_s - float(coverage.get("start_elapsed_s", 0.0))
                nearby = [
                    event for event in events
                    if abs(float(event["video_time_s"]) - video_time_s) <= event_window_s
                ]
                if not nearby:
                    rows.append({
                        "Time": timestamp,
                        "Log": f"{component} PCA Spike",
                        "Video": "No inferred video event",
                        "Result": self._uncertain_result("Undetermined", alignment_confidence),
                        "Alignment Confidence": alignment_confidence,
                        "Tolerance Window s": event_window_s,
                        "Comment": "動画範囲内だが対応イベントは検出されませんでした",
                    })
                    continue

                event_names = sorted({event["event"] for event in nearby})
                result = "Match" if "rapid_movement" in event_names else "Partial Match"
                rows.append({
                    "Time": timestamp,
                    "Log": f"{component} PCA Spike",
                    "Video": ", ".join(event_names),
                    "Result": result,
                    "Alignment Confidence": alignment_confidence,
                    "Tolerance Window s": event_window_s,
                    "Comment": "PCA異常時刻の近傍に動画イベントがあります",
                })
        return rows

    def _compare_phases_with_video(self, flight_phases, events, coverage, event_window_s, alignment_confidence=0.5):
        if flight_phases is None or flight_phases.empty or not events:
            return []

        rows = []
        comparable_events = {"hover", "takeoff", "landing", "rapid_movement", "lateral_motion", "forward_motion"}
        sampled = flight_phases.iloc[::max(1, int(len(flight_phases) / 100))]
        for _, phase_row in sampled.iterrows():
            telemetry_time_s = float(phase_row["telemetry_time_s"])
            if not _is_in_coverage(telemetry_time_s, coverage):
                continue
            telemetry_phase = str(phase_row["phase"])
            if telemetry_phase in {"unknown", "ground"}:
                continue
            video_time_s = telemetry_time_s - float(coverage.get("start_elapsed_s", 0.0))
            nearby = [
                event for event in events
                if event["event"] in comparable_events
                and abs(float(event["video_time_s"]) - video_time_s) <= event_window_s
            ]
            if not nearby:
                continue

            event_names = sorted({event["event"] for event in nearby})
            result = self._phase_result(telemetry_phase, event_names, alignment_confidence)
            rows.append({
                "Time": _format_time_s(telemetry_time_s),
                "Log": f"Telemetry phase: {telemetry_phase}",
                "Video": ", ".join(event_names),
                "Result": result,
                "Alignment Confidence": alignment_confidence,
                "Tolerance Window s": event_window_s,
                "Comment": "ログ側フライトフェーズと動画イベントの整合性チェック",
            })
        return rows

    def _phase_result(self, telemetry_phase, event_names, alignment_confidence):
        matching = {
            "takeoff": {"takeoff", "forward_motion"},
            "landing": {"landing"},
            "hover": {"hover"},
            "moving": {"rapid_movement", "lateral_motion", "forward_motion"},
        }
        if set(event_names) & matching.get(telemetry_phase, set()):
            return "Match"
        if alignment_confidence < 0.7:
            return "Contradiction(low-sync-confidence)"
        return "Contradiction"

    def _uncertain_result(self, base_result, alignment_confidence):
        if alignment_confidence < 0.7:
            return f"{base_result}(low-sync-confidence)"
        return base_result

    def _build_coverage(self, metadata, telemetry_df, video_offset_s):
        duration_s = float(metadata.get("duration_s") or 0.0)
        if telemetry_df is None:
            return {
                "status": "not_applicable",
                "reason": "video_only_analysis",
                "start_elapsed_s": None,
                "end_elapsed_s": None,
                "duration_s": duration_s,
                "telemetry_duration_s": None,
                "coverage_ratio": None,
            }
        telemetry_duration_s = _telemetry_duration_s(telemetry_df)
        start_elapsed_s = float(video_offset_s)
        end_elapsed_s = start_elapsed_s + duration_s
        coverage_ratio = (
            max(0.0, min(end_elapsed_s, telemetry_duration_s) - max(start_elapsed_s, 0.0)) / telemetry_duration_s
            if telemetry_duration_s > 0 and duration_s > 0
            else 0.0
        )
        return {
            "start_elapsed_s": start_elapsed_s,
            "end_elapsed_s": end_elapsed_s,
            "duration_s": duration_s,
            "telemetry_duration_s": telemetry_duration_s,
            "coverage_ratio": float(coverage_ratio),
        }

    def _build_feature_summary(self, features):
        if features is None or features.empty:
            return {
                "status": "skipped",
                "reason": "video_features_empty",
                "sample_count": 0,
            }

        summary = {
            "status": "completed",
            "sample_count": int(len(features)),
        }
        numeric_columns = ["brightness", "blur_score", "frame_diff", "motion_score"]
        for column in numeric_columns:
            if column not in features:
                continue
            series = pd.to_numeric(features[column], errors="coerce").dropna()
            if series.empty:
                continue
            summary[f"{column}_mean"] = float(series.mean())
            summary[f"{column}_min"] = float(series.min())
            summary[f"{column}_max"] = float(series.max())

        if "frame_visible" in features:
            summary["frame_visible_ratio"] = float(features["frame_visible"].astype(bool).mean())
        if "motion_direction" in features:
            summary["motion_direction_counts"] = features["motion_direction"].value_counts().to_dict()
        return summary

    def _motion_direction(self, dx, dy):
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return "none"
        if abs(dx) >= abs(dy):
            return "right" if dx > 0 else "left"
        return "down" if dy > 0 else "up"

    def _event(self, time_s, event, confidence):
        return {
            "video_time_s": round(float(time_s), 3),
            "event": event,
            "confidence": float(confidence),
        }

    def _deduplicate_events(self, events):
        deduped = []
        seen = set()
        for event in events:
            bucket = (event["event"], round(event["video_time_s"]))
            if bucket in seen:
                continue
            seen.add(bucket)
            deduped.append(event)
        return deduped


def _telemetry_duration_s(telemetry_df):
    if telemetry_df is None or telemetry_df.empty:
        return 0.0
    index = telemetry_df.index
    if hasattr(index, "total_seconds"):
        return float(index.max().total_seconds())
    try:
        return float(index.max())
    except (TypeError, ValueError):
        return float(len(telemetry_df) - 1)


def _parse_timestamp_s(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    parts = text.split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60.0 + float(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600.0 + int(parts[1]) * 60.0 + float(parts[2])
        return float(text)
    except ValueError:
        return None


def _is_in_coverage(telemetry_time_s, coverage):
    return (
        float(coverage.get("start_elapsed_s", 0.0))
        <= telemetry_time_s
        <= float(coverage.get("end_elapsed_s", 0.0))
    )


def _clamp(value, lower, upper):
    return max(lower, min(upper, value))


def _format_time_s(value):
    minutes = int(value // 60)
    seconds = value % 60
    return f"{minutes:02d}:{seconds:04.1f}"
