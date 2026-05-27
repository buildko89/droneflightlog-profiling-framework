import os
import tempfile
import unittest
from unittest.mock import patch

from drone_app.parser import UlgParser


class FakeDataset:
    def __init__(self, name, multi_id, data):
        self.name = name
        self.multi_id = multi_id
        self.data = data


class FakeULog:
    datasets = []

    def __init__(self, path):
        self.data_list = self.datasets

    def get_dataset(self, name, multi_instance=0):
        for dataset in self.data_list:
            if dataset.name == name and dataset.multi_id == multi_instance:
                return dataset
        raise KeyError(f"{name}[{multi_instance}]")


class TestUlgParser(unittest.TestCase):
    def setUp(self):
        handle = tempfile.NamedTemporaryFile(suffix=".ulg", delete=False)
        handle.close()
        self.ulg_path = handle.name

    def tearDown(self):
        if os.path.exists(self.ulg_path):
            os.remove(self.ulg_path)

    def test_bounded_fill_does_not_expand_single_sample_topic(self):
        FakeULog.datasets = [
            FakeDataset("sensor_combined", 0, {
                "timestamp": [0, 100000, 200000, 300000],
                "gyro_rad[0]": [1.0, 2.0, 3.0, 4.0],
            }),
            FakeDataset("manual_control_setpoint", 0, {
                "timestamp": [100000],
                "roll": [0.25],
            }),
        ]

        with patch("drone_app.parser.ULog", FakeULog):
            parser = UlgParser(self.ulg_path)
            df = parser.parse(
                topics=["sensor_combined", "manual_control_setpoint"],
                resample_rate="100ms",
            )

        self.assertEqual(len(df), 4)
        self.assertEqual(df["manual_control_setpoint_roll"].notna().sum(), 1)
        self.assertEqual(df["manual_control_setpoint_roll"].iloc[1], 0.25)

        report = parser.get_parse_report()
        self.assertEqual(report["fill_limits"]["manual_control_setpoint"], 0)
        self.assertIn("manual_control_setpoint", report["sparse_sources"])
        self.assertGreater(report["output_missing_values"], 0)

    def test_multiple_instances_are_exported_as_separate_columns(self):
        FakeULog.datasets = [
            FakeDataset("estimator_states", 0, {
                "timestamp": [0, 100000],
                "states[0]": [1.0, 2.0],
            }),
            FakeDataset("estimator_states", 1, {
                "timestamp": [0, 100000],
                "states[0]": [10.0, 20.0],
            }),
        ]

        with patch("drone_app.parser.ULog", FakeULog):
            parser = UlgParser(self.ulg_path)
            df = parser.parse(
                topics=["estimator_states"],
                resample_rate="100ms",
                fill_strategy="none",
            )

        self.assertIn("estimator_states_states[0]", df.columns)
        self.assertIn("estimator_states_1_states[0]", df.columns)
        self.assertEqual(df["estimator_states_states[0]"].iloc[1], 2.0)
        self.assertEqual(df["estimator_states_1_states[0]"].iloc[1], 20.0)

        report = parser.get_parse_report()
        self.assertEqual(report["topic_stats"]["estimator_states"]["instances"], 2)
        self.assertEqual(report["topic_stats"]["estimator_states"]["parsed_instances"], 2)
        self.assertIn("estimator_states_1", report["instance_stats"])


if __name__ == "__main__":
    unittest.main()
