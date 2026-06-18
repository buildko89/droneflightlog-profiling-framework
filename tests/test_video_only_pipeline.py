import os
import shutil
import tempfile
import unittest

from drone_app.video_pipeline import run_video_only_pipeline


class TestVideoOnlyPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="video_only_pipeline_test_")
        self.video_path = os.path.join(self.temp_dir, "invalid.mp4")
        with open(self.video_path, "wb") as file:
            file.write(b"not a real video")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_video_only_pipeline_writes_report_without_telemetry(self):
        results = run_video_only_pipeline(
            self.video_path,
            camera_viewpoint="external",
            workspace_dir=self.temp_dir,
            output_dir=self.temp_dir,
            run_output_subdir=False,
        )

        self.assertTrue(os.path.exists(results["report_path"]))
        context = results["context"]
        self.assertEqual(context.get_artifact("analysis_mode")["mode"], "video_only")
        self.assertEqual(context.get_artifact("video_coverage")["status"], "not_applicable")
        self.assertEqual(context.get_artifact("video_feature_summary")["status"], "skipped")

        with open(results["report_path"], encoding="utf-8") as report_file:
            report = report_file.read()
        self.assertIn("動画単体解析レポート", report)
        self.assertIn("動画メタデータ", report)
        self.assertIn("特徴量統計", report)
        self.assertIn("動画イベント", report)
        self.assertIn("動画解析をスキップしました", report)

    def test_video_only_pipeline_can_export_llm_prompt(self):
        results = run_video_only_pipeline(
            self.video_path,
            workspace_dir=self.temp_dir,
            output_dir=self.temp_dir,
            run_output_subdir=False,
            enable_llm=True,
            llm_type="dummy",
            mode="export",
        )

        self.assertTrue(os.path.exists(results["diagnosis_path"]))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "video_llm_prompt.txt")))

    def test_missing_video_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            run_video_only_pipeline(
                os.path.join(self.temp_dir, "missing.mp4"),
                workspace_dir=self.temp_dir,
                output_dir=self.temp_dir,
            )


if __name__ == "__main__":
    unittest.main()
