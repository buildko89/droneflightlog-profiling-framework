import pandas as pd
from pyulog import ULog
import os

DEFAULT_TOPICS = [
    'vehicle_attitude',
    'vehicle_angular_velocity',
    'vehicle_local_position',
    'vehicle_global_position',
    'estimator_states',
    'sensor_combined',
    'actuator_outputs',
    'vehicle_attitude_setpoint',
    'vehicle_rates_setpoint',
    'trajectory_setpoint',
    'input_rc',
    'manual_control_setpoint'
]

TOPIC_ALTERNATIVES = {
    'vehicle_global_position': [
        'vehicle_global_position',
        'vehicle_gps_position',
        'sensor_gps',
    ],
    'input_rc': [
        'input_rc',
        'manual_control_setpoint',
    ],
}

class UlgParser:
    """
    Parser for .ulg files using pyulog.
    Extracts specified topics, synchronizes them by resampling, and merges into a single DataFrame.
    """
    def __init__(self, ulg_file_path: str):
        self.ulg_file_path = ulg_file_path
        self.parse_report = {}
        if not os.path.exists(ulg_file_path):
            raise FileNotFoundError(f"ULog file not found: {ulg_file_path}")

    def parse(self, topics=None, resample_rate='100ms', fill_strategy='bounded') -> pd.DataFrame:
        """
        Parses the ULog file and returns a merged DataFrame.
        """
        if topics is None:
            topics = DEFAULT_TOPICS

        ulog = ULog(self.ulg_file_path)
        topic_instances = self._topic_instances(ulog)
        available_topics = {topic: len(instances) for topic, instances in topic_instances.items()}
        raw_dfs = []
        parsed_topics = []
        missing_topics = []
        failed_topics = {}
        topic_stats = {}
        instance_stats = {}
        resolved_alternatives = {}
        parsed_actual_topics = set()

        for topic in topics:
            actual_topic = self._resolve_topic(topic, available_topics)
            if actual_topic is None:
                missing_topics.append(topic)
                print(f"Warning: Missing topic {topic}")
                continue

            if actual_topic in parsed_actual_topics:
                if actual_topic != topic:
                    resolved_alternatives[topic] = actual_topic
                continue

            parsed_instance_count = 0
            topic_rows = 0
            topic_columns = 0
            for multi_instance in topic_instances.get(actual_topic, [0]):
                try:
                    data = ulog.get_dataset(actual_topic, multi_instance=multi_instance)
                    df = self._dataset_to_frame(data, actual_topic, multi_instance, available_topics[actual_topic])

                    source_key = self._source_key(actual_topic, multi_instance, available_topics[actual_topic])
                    raw_dfs.append((source_key, df))
                    parsed_instance_count += 1
                    topic_rows += len(df)
                    topic_columns += len(df.columns)
                    instance_stats[source_key] = {
                        "topic": actual_topic,
                        "multi_instance": int(multi_instance),
                        "rows": int(len(df)),
                        "columns": int(len(df.columns)),
                    }
                except Exception as e:
                    failed_key = self._source_key(actual_topic, multi_instance, available_topics[actual_topic])
                    failed_topics[failed_key] = str(e)
                    print(f"Warning: Failed to parse topic {failed_key}: {e}")

            if parsed_instance_count:
                parsed_topics.append(actual_topic)
                parsed_actual_topics.add(actual_topic)
                topic_stats[actual_topic] = {
                    "rows": int(topic_rows),
                    "columns": int(topic_columns),
                    "instances": int(available_topics.get(actual_topic, 0)),
                    "parsed_instances": int(parsed_instance_count),
                }
                if actual_topic != topic:
                    resolved_alternatives[topic] = actual_topic

        if not raw_dfs:
            self.parse_report = {
                "parsed_topics": parsed_topics,
                "missing_topics": missing_topics,
                "failed_topics": failed_topics,
                "resolved_alternatives": resolved_alternatives,
                "topic_stats": topic_stats,
                "instance_stats": instance_stats,
                "resample_rate": resample_rate,
                "fill_strategy": fill_strategy,
            }
            raise ValueError("No specified topics found in the ULog file.")

        global_start = min(df.index.min() for _, df in raw_dfs)
        resampled = [
            (source_key, self._resample_to_common_grid(df, resample_rate, global_start), df)
            for source_key, df in raw_dfs
        ]
        global_end = max(df.index.max() for _, df, _ in resampled)
        common_index = pd.timedelta_range(start=global_start, end=global_end, freq=resample_rate)
        aligned_dfs = []
        fill_limits = {}
        sparse_sources = []
        for source_key, df, raw_df in resampled:
            aligned_df, fill_limit = self._align_to_common_grid(
                df,
                raw_df,
                common_index,
                resample_rate,
                fill_strategy,
            )
            aligned_dfs.append(aligned_df)
            fill_limits[source_key] = fill_limit
            if fill_limit == 0:
                sparse_sources.append(source_key)

        merged_df = pd.concat(aligned_dfs, axis=1).sort_index()
        
        # Drop columns that are completely NaN
        merged_df = merged_df.dropna(axis=1, how='all')

        self.parse_report = {
            "parsed_topics": parsed_topics,
            "missing_topics": missing_topics,
            "failed_topics": failed_topics,
            "resolved_alternatives": resolved_alternatives,
            "topic_stats": topic_stats,
            "instance_stats": instance_stats,
            "topic_instances": topic_instances,
            "resample_rate": resample_rate,
            "fill_strategy": fill_strategy,
            "fill_limits": fill_limits,
            "sparse_sources": sparse_sources,
            "global_start_us": int(global_start / pd.Timedelta(microseconds=1)),
            "output_rows": int(len(merged_df)),
            "output_columns": int(len(merged_df.columns)),
            "output_missing_values": int(merged_df.isna().sum().sum()),
        }
        
        return merged_df

    def get_parse_report(self) -> dict:
        """
        Returns metadata from the most recent parse call.
        """
        return self.parse_report

    def to_csv(self, output_path: str, topics=None, resample_rate='100ms', fill_strategy='bounded') -> str:
        """
        Parses the ULog file and saves the result to a CSV file.
        Returns the path to the saved CSV.
        """
        df = self.parse(topics, resample_rate, fill_strategy=fill_strategy)
        df.to_csv(output_path)
        return output_path

    def _resolve_topic(self, requested_topic, available_topics):
        candidates = TOPIC_ALTERNATIVES.get(requested_topic, [requested_topic])
        for candidate in candidates:
            if candidate in available_topics:
                return candidate
        return None

    def _topic_instances(self, ulog):
        topic_instances = {}
        for dataset in ulog.data_list:
            instances = topic_instances.setdefault(dataset.name, [])
            instances.append(getattr(dataset, "multi_id", 0))
        return {
            topic: sorted(int(instance) for instance in instances)
            for topic, instances in topic_instances.items()
        }

    def _dataset_to_frame(self, data, topic, multi_instance, instance_count):
        df = pd.DataFrame(data.data)

        if df.empty or 'timestamp' not in df.columns:
            raise ValueError("dataset is empty or has no timestamp column")

        df['timestamp'] = pd.to_timedelta(df['timestamp'], unit='us')
        df.set_index('timestamp', inplace=True)
        df = df.sort_index()

        prefix = topic if multi_instance == 0 else f"{topic}_{multi_instance}"
        if instance_count == 1:
            prefix = topic
        df.columns = [f"{prefix}_{col}" for col in df.columns]
        return df

    def _source_key(self, topic, multi_instance, instance_count):
        if instance_count == 1 or multi_instance == 0:
            return topic
        return f"{topic}_{multi_instance}"

    def _resample_to_common_grid(self, df, resample_rate, global_start):
        freq = pd.Timedelta(resample_rate)
        if freq <= pd.Timedelta(0):
            raise ValueError(f"resample_rate must be positive: {resample_rate}")

        offsets = df.index - global_start
        buckets = (offsets // freq) * freq + global_start
        return df.groupby(buckets).mean()

    def _align_to_common_grid(self, df, raw_df, common_index, resample_rate, fill_strategy):
        aligned_df = df.reindex(common_index)
        if fill_strategy in (None, 'none'):
            return aligned_df, 0
        if fill_strategy == 'unbounded':
            return aligned_df.ffill().bfill(), -1
        if fill_strategy != 'bounded':
            raise ValueError(f"Unsupported fill_strategy: {fill_strategy}")

        fill_limit = self._bounded_fill_limit(raw_df, resample_rate)
        if fill_limit <= 0:
            return aligned_df, 0
        return aligned_df.ffill(limit=fill_limit), fill_limit

    def _bounded_fill_limit(self, df, resample_rate):
        if len(df.index) < 2:
            return 0
        freq = pd.Timedelta(resample_rate)
        deltas = df.index.to_series().diff().dropna()
        if deltas.empty:
            return 0
        median_delta = deltas.median()
        max_gap = max(freq, median_delta * 1.5)
        return max(1, int(max_gap / freq))
