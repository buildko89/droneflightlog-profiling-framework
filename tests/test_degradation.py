import os
import shutil
import tempfile
import unittest
import pandas as pd
from unittest.mock import MagicMock

from profilecore.core.context import ProfileCoreContext
from drone_app.history_manager import FlightHistoryManager
from drone_app.break_detector import StructuralBreakAnalyzer
from drone_app.interpreter import LLMInterpreter

class TestDegradationModules(unittest.TestCase):
    def setUp(self):
        self.workspace_dir = tempfile.mkdtemp(prefix="drone_degradation_test_")
        self.context = ProfileCoreContext(workspace_dir=self.workspace_dir)

    def tearDown(self):
        shutil.rmtree(self.workspace_dir, ignore_errors=True)

    def test_history_manager_calculates_and_saves_metrics(self):
        # Setup mock PCA scores and anomaly timestamps
        pca_scores = pd.DataFrame(
            {
                'PC1': [1.0, 2.0, 3.0, 4.0, 5.0],
                'PC2': [-1.0, 0.0, 1.0, 0.0, -1.0]
            },
            index=pd.to_timedelta([0, 100, 200, 300, 400], unit='ms')
        )
        anomaly_timestamps = {
            'PC1': ['00:00.1', '00:00.3'],
            'PC2': []
        }
        self.context.set_data('pca_scores', pca_scores)
        self.context.set_data('anomaly_timestamps', anomaly_timestamps)

        # Run history manager
        manager = FlightHistoryManager(self.context)
        history_df = manager.update_history()

        self.assertIsNotNone(history_df)
        self.assertEqual(len(history_df), 1)
        self.assertIn('PC1_variance', history_df.columns)
        self.assertIn('PC1_trend', history_df.columns)
        self.assertIn('PC1_anomaly_count', history_df.columns)
        self.assertIn('source_sha256', history_df.columns)

        # Expected metrics
        # PC1 variance of [1, 2, 3, 4, 5] is 2.5
        self.assertAlmostEqual(history_df['PC1_variance'].iloc[0], 2.5)
        # PC1 trend is 5.0 - 1.0 = 4.0
        self.assertAlmostEqual(history_df['PC1_trend'].iloc[0], 4.0)
        # PC1 anomaly count is 2
        self.assertEqual(history_df['PC1_anomaly_count'].iloc[0], 2)

        # Verify CSV is saved
        csv_path = os.path.join(self.workspace_dir, "flight_history.csv")
        self.assertTrue(os.path.exists(csv_path))
        loaded_df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        self.assertEqual(len(loaded_df), 1)

    def test_history_manager_records_source_metadata_and_duplicate(self):
        source_path = os.path.join(self.workspace_dir, "flight.csv")
        with open(source_path, "w", encoding="utf-8") as source_file:
            source_file.write("time,value\n0,1\n1,2\n")

        pca_scores = pd.DataFrame(
            {'PC1': [1.0, 2.0, 3.0]},
            index=pd.to_timedelta([0, 100, 200], unit='ms')
        )
        self.context.set_data('pca_scores', pca_scores)
        self.context.set_data('anomaly_timestamps', {})

        manager = FlightHistoryManager(self.context)
        first_history = manager.update_history(source_path=source_path)
        second_history = manager.update_history(source_path=source_path, duplicate_policy='skip')

        self.assertEqual(len(first_history), 1)
        self.assertEqual(len(second_history), 1)
        self.assertEqual(second_history['source_file_name'].iloc[0], "flight.csv")
        self.assertTrue(isinstance(second_history['source_sha256'].iloc[0], str))

        duplicate = self.context.get_artifact('flight_history_duplicate')
        self.assertTrue(duplicate['detected'])
        self.assertEqual(duplicate['matching_rows'], 1)

    def test_history_manager_appends_to_existing_file(self):
        csv_path = os.path.join(self.workspace_dir, "flight_history.csv")
        os.makedirs(self.workspace_dir, exist_ok=True)
        
        # Pre-create history CSV with 1 row
        old_time = pd.Timestamp.now() - pd.Timedelta(days=1)
        old_df = pd.DataFrame(
            [{'PC1_variance': 1.0, 'PC1_trend': 0.1, 'PC1_anomaly_count': 0}],
            index=[old_time]
        )
        old_df.index.name = 'timestamp'
        old_df.to_csv(csv_path)

        # Setup PCA scores for new run
        pca_scores = pd.DataFrame(
            {'PC1': [10.0, 12.0]},
            index=pd.to_timedelta([0, 100], unit='ms')
        )
        self.context.set_data('pca_scores', pca_scores)
        self.context.set_data('anomaly_timestamps', {})

        manager = FlightHistoryManager(self.context)
        history_df = manager.update_history()

        self.assertEqual(len(history_df), 2)
        self.assertIn('PC1_variance', history_df.columns)
        # New variance is var of [10, 12] which is 2.0
        self.assertAlmostEqual(history_df['PC1_variance'].iloc[-1], 2.0)

    def test_break_detector_skips_when_less_than_min_history(self):
        # Create history with fewer flights than the default minimum.
        times = [pd.Timestamp.now() - pd.Timedelta(hours=i) for i in range(2)]
        history_df = pd.DataFrame(
            {'PC1_variance': [1.0, 1.1]},
            index=times
        )
        history_df.index.name = 'timestamp'
        self.context.set_data('flight_history', history_df)

        analyzer = StructuralBreakAnalyzer(self.context)
        result = analyzer.detect_breaks()

        self.assertFalse(result['detected'])
        self.assertEqual(result['status'], 'skipped')
        self.assertIn('reason', result)
        self.assertEqual(result['config']['min_history'], 5)
        self.assertEqual(self.context.get_data('structural_break'), result)

    def test_break_detector_no_break_when_values_under_threshold(self):
        # Create history with enough stable flights.
        times = [pd.Timestamp.now() - pd.Timedelta(hours=i) for i in range(5)]
        history_df = pd.DataFrame(
            {'PC1_variance': [1.0, 1.0, 1.0, 1.0, 1.0]},
            index=times
        )
        history_df.index.name = 'timestamp'
        self.context.set_data('flight_history', history_df)

        analyzer = StructuralBreakAnalyzer(self.context)
        result = analyzer.detect_breaks()

        self.assertFalse(result['detected'])
        self.assertEqual(result['status'], 'success')
        self.assertEqual(len(result['detected_columns']), 0)
        self.assertEqual(result['config']['recent_window'], 2)

    def test_break_detector_detects_break(self):
        # Create history with 6 flights where the last 2 are breaks
        # First 4 flights: variance = 1.0 (mean=1.0, std=0.0, threshold=1.0)
        # Last 2 flights: variance = 2.0 (exceeds threshold)
        times = [pd.Timestamp.now() - pd.Timedelta(hours=5-i) for i in range(6)]
        history_df = pd.DataFrame(
            {'PC1_variance': [1.0, 1.0, 1.0, 1.0, 2.0, 2.0]},
            index=times
        )
        history_df.index.name = 'timestamp'
        self.context.set_data('flight_history', history_df)

        analyzer = StructuralBreakAnalyzer(self.context)
        result = analyzer.detect_breaks()

        self.assertTrue(result['detected'])
        self.assertEqual(result['status'], 'success')
        self.assertIn('PC1_variance', result['detected_columns'])
        self.assertEqual(result['timestamp'], str(times[-1]))
        self.assertEqual(result['break_timestamp'], str(times[-1]))

    def test_interpreter_includes_history_and_break_in_prompt(self):
        # Mock LLM Client
        mock_client = MagicMock()
        mock_client.model_name = "test-model"
        mock_client.generate_text.return_value = "Test Diagnosis Response"

        # Mock PCA outputs
        pca_variance = pd.DataFrame({'Component': ['PC1'], 'Explained_Variance_Ratio': [1.0]})
        pca_scores = pd.DataFrame({'PC1': [1.0, 2.0]}, index=pd.to_timedelta([0, 100], unit='ms'))
        pca_loadings = pd.DataFrame(
            {'sensor_combined_accelerometer_m_s2[0]': [0.9]},
            index=['PC1'],
        )
        self.context.set_data('pca_variance', pca_variance)
        self.context.set_data('pca_scores', pca_scores)
        self.context.set_data('pca_loadings', pca_loadings)
        self.context.set_artifact('pca_preprocessing_report', {
            'selected_column_count': 1,
            'requested_n_components': 1,
            'effective_n_components': 1,
            'all_nan_columns': [],
            'constant_columns': [],
        })
        self.context.set_artifact('anomaly_detection_config', {
            'method': 'pca_score_zscore',
            'z_threshold': 3.0,
            'comparison': 'absolute_z_score_greater_than_threshold',
        })

        # Setup flight history and structural break
        times = [pd.Timestamp.now() - pd.Timedelta(hours=5-i) for i in range(6)]
        history_df = pd.DataFrame(
            {'PC1_variance': [1.0, 1.0, 1.0, 1.0, 2.0, 2.0]},
            index=times
        )
        history_df.index.name = 'timestamp'
        self.context.set_data('flight_history', history_df)

        structural_break = {
            'detected': True,
            'status': 'success',
            'detected_columns': ['PC1_variance'],
            'break_details': {
                'PC1_variance': {
                    'mean': 1.0, 'std': 0.1, 'threshold': 1.2,
                    'last_value': 2.0, 'prev_value': 2.0, 'detected': True
                }
            },
            'timestamp': str(times[-1])
        }
        self.context.set_data('structural_break', structural_break)

        interpreter = LLMInterpreter(self.context, llm_client=mock_client)
        # Create output file
        diag_output_path = os.path.join(self.workspace_dir, "diagnosis.md")
        success = interpreter.run_interpretation(output_file=diag_output_path)

        self.assertTrue(success)
        mock_client.generate_text.assert_called_once()
        prompt_arg = mock_client.generate_text.call_args[0][0]

        # Verify prompt components
        self.assertIn("過去のフライト履歴との比較", prompt_arg)
        self.assertIn("警告：構造的変化（経年劣化の兆候）の検出", prompt_arg)
        self.assertIn("モーター等の摩耗が疑われます", prompt_arg)
        self.assertIn("長期的な予知保全", prompt_arg)
        self.assertIn("PCAの主成分は統計的な合成軸", prompt_arg)
        self.assertIn("Z-score閾値", prompt_arg)
        self.assertIn("加速度", prompt_arg)

    def test_interpreter_export_mode(self):
        # Mock LLM Client
        mock_client = MagicMock()
        mock_client.model_name = "test-model"

        # Mock PCA outputs
        pca_variance = pd.DataFrame({'Component': ['PC1'], 'Explained_Variance_Ratio': [1.0]})
        pca_scores = pd.DataFrame({'PC1': [1.0, 2.0]}, index=pd.to_timedelta([0, 100], unit='ms'))
        self.context.set_data('pca_variance', pca_variance)
        self.context.set_data('pca_scores', pca_scores)

        # Set export mode in settings
        self.context.set_setting('llm_mode', 'export')

        interpreter = LLMInterpreter(self.context, llm_client=mock_client)
        diag_output_path = os.path.join(self.workspace_dir, "diagnosis.md")
        
        success = interpreter.run_interpretation(output_file=diag_output_path)

        self.assertTrue(success)
        # In export mode, LLM client's generate_text should not be called
        mock_client.generate_text.assert_not_called()

        # Check prompt file exported
        prompt_file = os.path.join(self.workspace_dir, "llm_prompt.txt")
        self.assertTrue(os.path.exists(prompt_file))
        with open(prompt_file, encoding='utf-8') as f:
            prompt_content = f.read()
        self.assertIn("主成分スコアの基本統計量", prompt_content)

        # Check diagnosis.md output has instructions
        self.assertTrue(os.path.exists(diag_output_path))
        with open(diag_output_path, encoding='utf-8') as f:
            diagnosis_content = f.read()
        self.assertIn("プロンプト・エクスポート", diagnosis_content)
        self.assertIn("ローカルの自律型エージェントツール", diagnosis_content)

if __name__ == '__main__':
    unittest.main()
