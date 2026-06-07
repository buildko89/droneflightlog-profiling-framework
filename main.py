import os
import sys
import argparse
from dotenv import load_dotenv
from profilecore.core.context import ProfileCoreContext
from profilecore.core.quality import build_data_quality_summary
from profilecore.io.exporter import ReportExporter
from drone_app.csv_loader import CsvTelemetryLoader
from drone_app.parser import UlgParser
from drone_app.analyzer import TelemetryAnalyzer
from drone_app.visualizer import TelemetryVisualizer
from drone_app.interpreter import LLMInterpreter
from drone_app.llm_clients import create_llm_client, resolve_llm_settings

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
    
    args = parser.parse_args()

    input_file = args.file_path
    if not os.path.exists(input_file):
        print(f"エラー: 指定されたファイルが見つかりません: {input_file}")
        return

    # 1. ProfileCoreContext Instantiation
    workspace_dir = "workspace"
    if not os.path.exists(workspace_dir):
        os.makedirs(workspace_dir)
        
    context = ProfileCoreContext(workspace_dir=workspace_dir)
    context.add_log("Pipeline started.")

    # 2. .ulg または .csv から DataFrame への変換
    if input_file.lower().endswith('.csv'):
        loader = CsvTelemetryLoader(input_file, config_path=args.csv_config)
        context.add_log(f"Reading CSV file: {input_file}")
        df = loader.load()
        context.set_artifact("csv_parse_report", loader.get_parse_report())
    else:
        parser_obj = UlgParser(input_file)
        context.add_log(f"Parsing ULog file: {input_file}")
        df = parser_obj.parse(resample_rate='100ms')
        context.set_artifact("ulg_parse_report", parser_obj.get_parse_report())
    context.set_data('raw_data', df)
    context.set_artifact("data_quality", build_data_quality_summary(df))
    context.add_log(f"Parsed {len(df)} samples into context with key 'raw_data'")

    csv_path = os.path.join(workspace_dir, "telemetry_data.csv")
    df.to_csv(csv_path)
    context.add_log(f"Raw data saved to: {csv_path}")

    # 3. TelemetryAnalyzer の分析実行
    analyzer = TelemetryAnalyzer(context)
    analyzer.analyze(data_key='raw_data', n_components=3)
    pca_variance = context.get_data("pca_variance")
    if pca_variance is not None:
        context.set_artifact("feature_extraction_status", {
            "status": "completed",
            "method": "telemetry_pca",
            "n_components": int(len(pca_variance)),
        })
    else:
        context.set_artifact("feature_extraction_status", {
            "status": "skipped",
            "reason": "PCA results were not generated",
        })

    insights = []
    if pca_variance is not None:
        cumulative = float(pca_variance["Explained_Variance_Ratio"].sum())
        insights.append({
            "level": "info",
            "message": f"PCA cumulative explained variance: {cumulative:.1%}",
        })

    anomalies = context.get_data("anomaly_timestamps")
    if anomalies:
        detected = {pc: times for pc, times in anomalies.items() if times}
        if detected:
            for pc, times in detected.items():
                insights.append({
                    "level": "warning",
                    "message": f"{pc} anomaly spikes detected at {', '.join(times[:5])}",
                })
        else:
            insights.append({
                "level": "info",
                "message": "No PCA anomaly spikes were detected.",
            })

    context.set_artifact("summary_insights", insights)

    # 4. TelemetryVisualizer によるグラフ出力
    visualizer = TelemetryVisualizer(context, output_dir="output")
    visualizer.plot_raw_telemetry(filename="raw_telemetry.png")
    visualizer.plot_pca_results(filename="pca_plot.png")
    visualizer.plot_variance(filename="pca_variance.png")
    context.add_log("Visualizations generated and saved to 'output/' folder.")

    # 履歴の記録と経年劣化の検知
    from drone_app.history_manager import FlightHistoryManager
    from drone_app.break_detector import StructuralBreakAnalyzer
    
    FlightHistoryManager(context).update_history()
    StructuralBreakAnalyzer(context).detect_breaks()

    # 5. LLMクライアントの動的生成とInterpreterによる分析の言語化
    try:
        llm_settings = resolve_llm_settings(
            service=args.llm,
            model_name=args.model,
            config_path=args.llm_config,
            mode=args.mode,
        )
        context.set_setting('llm_mode', llm_settings["mode"])
        client = create_llm_client(
            service=llm_settings["service"],
            model_name=llm_settings["model"],
            config_path=args.llm_config,
        )
            
        interpreter = LLMInterpreter(context, llm_client=client)
        # モデル名をファイル名に使用（スラッシュなどはアンダースコアに置換）
        safe_model_name = client.model_name.replace("/", "_").replace(":", "_")
        diag_output_path = f"output/diagnosis_{safe_model_name}.md"
        
        # APIエラー発生時に処理を中断する
        if not interpreter.run_interpretation(output_file=diag_output_path):
            print(f"\n[CRITICAL ERROR] LLM Interpretation failed using client '{args.llm}'. Pipeline aborted.")
            sys.exit(1)
        context.set_artifact("llm_interpretation", {
            "status": "completed",
            "client": llm_settings["service"],
            "model": client.model_name,
            "output_file": diag_output_path,
        })
            
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Failed to initialize LLM client: {str(e)}")
        sys.exit(1)

    # 6. profilecore の ReportExporter による Markdown レポートの出力
    exporter = ReportExporter(context)
    exporter.export_markdown(filename="drone_analysis_report.md")
    context.add_log("Final report generated.")

    print("\n--- Analysis Complete ---")
    print(f"LLM Client: {llm_settings['service']} (Model: {client.model_name})")
    print(f"Check 'output/drone_analysis_report.md' for the results.")

if __name__ == "__main__":
    main()
