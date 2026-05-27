import json
import os
import tempfile
import unittest

import pandas as pd

from drone_app.csv_loader import CsvTelemetryLoader


class TestCsvTelemetryLoader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="csv_loader_test_")

    def tearDown(self):
        for name in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, name))
        os.rmdir(self.temp_dir)

    def test_loads_csv_with_mapping_and_timedelta_index(self):
        csv_path = os.path.join(self.temp_dir, "telemetry.csv")
        config_path = os.path.join(self.temp_dir, "mapping.json")
        pd.DataFrame({
            "time_s": [0.0, 0.1, 0.2],
            "acc_x": ["1.0", "2.0", "3.0"],
            "motor_1": [1000, 1001, 1002],
            "mode": ["takeoff", "takeoff", "hover"],
        }).to_csv(csv_path, index=False)
        with open(config_path, "w", encoding="utf-8") as config_file:
            json.dump({
                "timestamp_column": "time_s",
                "timestamp_unit": "s",
                "columns": {
                    "acc_x": "sensor_combined_accelerometer_m_s2[0]",
                    "motor_1": "actuator_outputs_output[0]",
                },
            }, config_file)

        loader = CsvTelemetryLoader(csv_path, config_path=config_path)
        df = loader.load()

        self.assertEqual(df.index[1], pd.Timedelta(milliseconds=100))
        self.assertIn("sensor_combined_accelerometer_m_s2[0]", df.columns)
        self.assertIn("actuator_outputs_output[0]", df.columns)
        self.assertTrue(pd.api.types.is_numeric_dtype(df["sensor_combined_accelerometer_m_s2[0]"]))

        report = loader.get_parse_report()
        self.assertEqual(report["timestamp_column"], "time_s")
        self.assertEqual(report["timestamp_status"], "numeric")
        self.assertEqual(report["numeric_column_count"], 2)
        self.assertEqual(
            report["column_mapping_applied"]["acc_x"],
            "sensor_combined_accelerometer_m_s2[0]",
        )

    def test_datetime_timestamp_is_converted_to_elapsed_time(self):
        csv_path = os.path.join(self.temp_dir, "telemetry_datetime.csv")
        pd.DataFrame({
            "datetime": [
                "2026-03-10T10:40:44.000",
                "2026-03-10T10:40:44.250",
            ],
            "value": [1, 2],
        }).to_csv(csv_path, index=False)

        loader = CsvTelemetryLoader(csv_path)
        df = loader.load()

        self.assertEqual(df.index[0], pd.Timedelta(0))
        self.assertEqual(df.index[1], pd.Timedelta(milliseconds=250))
        self.assertEqual(loader.get_parse_report()["timestamp_status"], "datetime")

    def test_missing_timestamp_uses_row_number_seconds(self):
        csv_path = os.path.join(self.temp_dir, "telemetry_no_time.csv")
        pd.DataFrame({"value": [1, 2, 3]}).to_csv(csv_path, index=False)

        loader = CsvTelemetryLoader(csv_path)
        df = loader.load()

        self.assertEqual(df.index[2], pd.Timedelta(seconds=2))
        self.assertIsNone(loader.get_parse_report()["timestamp_column"])
        self.assertEqual(loader.get_parse_report()["timestamp_status"], "not_found")

    def test_unnamed_first_column_is_treated_as_saved_index_timestamp(self):
        csv_path = os.path.join(self.temp_dir, "telemetry_saved_index.csv")
        with open(csv_path, "w", encoding="utf-8") as csv_file:
            csv_file.write(",value,vehicle_attitude_timestamp_sample\n")
            csv_file.write("0 days 00:00:00,1,100\n")
            csv_file.write("0 days 00:00:00.100000,2,200\n")

        loader = CsvTelemetryLoader(csv_path)
        df = loader.load()

        self.assertEqual(df.index[1], pd.Timedelta(milliseconds=100))
        self.assertEqual(loader.get_parse_report()["timestamp_column"], "Unnamed: 0")
        self.assertEqual(loader.get_parse_report()["timestamp_status"], "timedelta")


if __name__ == "__main__":
    unittest.main()
