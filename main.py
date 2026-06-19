import os
import sys
import argparse
from dotenv import load_dotenv
from drone_app.pipeline import run_analysis_pipeline

def main():
    # 0. 環境変数の読み込み (.env ファイルがある場合)
    load_dotenv()

    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description="Drone Telemetry Analysis Pipeline")
    parser.add_argument("file_path", metavar="FILE", help="解析対象のファイル(.ulg または .csv)へのパス")
    
    # LLMタイプの選択
    parser.add_argument(
        "--llm", 
        choices=["gemini", "openai", "anthropic", "dummy"], 
        help="使用するLLMタイプ (未指定時は llm_config.json または gemini)"
    )
    
    # モデル名指定用の引数
    parser.add_argument(
        "--model", "-m", 
        help="使用するモデル名 (未指定時は llm_config.json または各クライアントのデフォルト)"
    )

    parser.add_argument(
        "--llm-config",
        default="llm_config.json",
        help="LLMサービスとモデル名を指定するJSONファイル (デフォルト: llm_config.json)"
    )

    parser.add_argument(
        "--csv-config",
        help="CSVのtimestamp列や列名マッピングを指定するJSONファイル"
    )
    
    parser.add_argument(
        "--mode",
        choices=["api", "export"],
        help="LLM連携モード (api: API接続, export: プロンプト書き出し。未指定時は llm_config.json または api)"
    )

    parser.add_argument(
        "--break-min-history",
        type=int,
        default=5,
        help="経年劣化検知に必要な最低履歴数 (デフォルト: 5)"
    )

    parser.add_argument(
        "--break-threshold-sigma",
        type=float,
        default=2.0,
        help="構造的変化検知の閾値倍率。平均 + N*標準偏差 (デフォルト: 2.0)"
    )

    parser.add_argument(
        "--anomaly-z-threshold",
        type=float,
        default=3.0,
        help="PCAスコア異常検知のZ-score閾値 (デフォルト: 3.0)"
    )

    parser.add_argument(
        "--analysis-mode",
        choices=["pca", "raw"],
        default="pca",
        help="解析方式。pca=従来のPCA診断, raw=生テレメトリ直接診断 (デフォルト: pca)"
    )

    parser.add_argument(
        "--flat-output",
        action="store_true",
        help="output/runs配下ではなく従来どおりoutput直下に成果物を出力する"
    )

    parser.add_argument(
        "--video",
        help="ログと照合する動画ファイルへのパス"
    )

    parser.add_argument(
        "--video-offset-s",
        type=float,
        default=0.0,
        help="動画時刻に加算するログ上の開始オフセット秒 (telemetry_time = video_time + offset)"
    )

    parser.add_argument(
        "--camera-viewpoint",
        choices=["external", "onboard"],
        default="external",
        help="動画のカメラ視点。external=地上撮影, onboard=機体搭載 (デフォルト: external)"
    )

    parser.add_argument(
        "--video-alignment-confidence",
        type=float,
        default=0.5,
        help="動画とログの同期信頼度 0.0-1.0。低い場合、矛盾判定を要確認扱いにする"
    )
    
    args = parser.parse_args()

    input_file = args.file_path
    if not os.path.exists(input_file):
        print(f"エラー: 指定されたファイルが見つかりません: {input_file}")
        return
    if args.video and not os.path.exists(args.video):
        print(f"エラー: 指定された動画ファイルが見つかりません: {args.video}")
        return

    try:
        results = run_analysis_pipeline(
            input_file,
            llm_type=args.llm,
            model_name=args.model,
            llm_config_path=args.llm_config,
            csv_config_path=args.csv_config,
            mode=args.mode,
            run_output_subdir=not args.flat_output,
            anomaly_z_threshold=args.anomaly_z_threshold,
            video_path=args.video,
            video_offset_s=args.video_offset_s,
            camera_viewpoint=args.camera_viewpoint,
            video_alignment_confidence=args.video_alignment_confidence,
            break_min_history=args.break_min_history,
            break_threshold_sigma=args.break_threshold_sigma,
            analysis_mode=args.analysis_mode,
        )
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Pipeline failed: {str(e)}")
        sys.exit(1)

    print("\n--- Analysis Complete ---")
    print(
        "LLM Client: "
        f"{results['llm_settings']['service']} "
        f"(Model: {results['llm_client'].model_name})"
    )
    print(f"Check '{results['report_path']}' for the results.")

if __name__ == "__main__":
    main()
