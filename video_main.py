import argparse
import os
import sys

from dotenv import load_dotenv

from drone_app.video_pipeline import run_video_only_pipeline


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Video-only drone footage analysis")
    parser.add_argument("video_path", metavar="VIDEO", help="解析対象の動画ファイルへのパス")
    parser.add_argument(
        "--camera-viewpoint",
        choices=["external", "onboard"],
        default="external",
        help="動画のカメラ視点。external=地上撮影, onboard=機体搭載 (デフォルト: external)",
    )
    parser.add_argument(
        "--sample-interval-s",
        type=float,
        default=1.0,
        help="動画フレームのサンプリング間隔秒 (デフォルト: 1.0)",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="成果物の出力先ディレクトリ (デフォルト: output)",
    )
    parser.add_argument(
        "--workspace-dir",
        default="workspace",
        help="作業ファイルの出力先ディレクトリ (デフォルト: workspace)",
    )
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help="output/runs配下ではなくoutput直下に成果物を出力する",
    )
    parser.add_argument(
        "--llm",
        choices=["gemini", "openai", "anthropic", "dummy"],
        help="動画単体診断に使用するLLMタイプ。未指定ならAI診断をスキップする",
    )
    parser.add_argument("--model", "-m", help="使用するモデル名")
    parser.add_argument(
        "--mode",
        choices=["api", "export"],
        help="LLM連携モード。指定した場合はAI診断を有効化する",
    )
    parser.add_argument(
        "--llm-config",
        default="llm_config.json",
        help="LLM設定JSONファイル (デフォルト: llm_config.json)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"エラー: 指定された動画ファイルが見つかりません: {args.video_path}")
        return 1

    enable_llm = bool(args.llm or args.mode or args.model)
    try:
        results = run_video_only_pipeline(
            args.video_path,
            camera_viewpoint=args.camera_viewpoint,
            sample_interval_s=args.sample_interval_s,
            workspace_dir=args.workspace_dir,
            output_dir=args.output_dir,
            run_output_subdir=not args.flat_output,
            enable_llm=enable_llm,
            llm_type=args.llm,
            model_name=args.model,
            mode=args.mode,
            llm_config_path=args.llm_config,
        )
    except Exception as exc:
        print(f"\n[CRITICAL ERROR] Video-only pipeline failed: {exc}")
        return 1

    print("\n--- Video Analysis Complete ---")
    print(f"Report: {results['report_path']}")
    if results.get("diagnosis_path"):
        print(f"Diagnosis: {results['diagnosis_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
