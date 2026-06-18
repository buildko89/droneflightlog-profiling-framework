import unittest

import pandas as pd

from drone_app.flight_phase_analyzer import FlightPhaseAnalyzer
from profilecore.core.context import ProfileCoreContext


class TestFlightPhaseAnalyzer(unittest.TestCase):
    def test_derives_basic_phases_from_altitude(self):
        context = ProfileCoreContext()
        context.set_data("raw_data", pd.DataFrame({
            "vehicle_local_position_z": [0.0, -0.2, -1.0, -2.0, -2.0, -2.1, -1.0],
            "actuator_outputs_output[0]": [1000, 1200, 1500, 1500, 1400, 1200, 1000],
        }, index=pd.to_timedelta([0, 1, 2, 3, 4, 5, 6], unit="s")))

        report = FlightPhaseAnalyzer(context).analyze()
        phases = context.get_data("flight_phases")

        self.assertEqual(report["status"], "completed")
        self.assertIn("takeoff", set(phases["phase"]))
        self.assertIn("hover", set(phases["phase"]))
        self.assertIn("landing", set(phases["phase"]))
        self.assertIn("flight_phase_report", context.artifacts)


if __name__ == "__main__":
    unittest.main()
