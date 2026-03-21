import os
import sys
import argparse
from dotenv import load_dotenv
from profilecore.core.context import ProfileCoreContext
from profilecore.io.exporter import ReportExporter
from drone_app.parser import UlgParser
from drone_app.analyzer import TelemetryAnalyzer
from drone_app.visualizer import TelemetryVisualizer
from drone_app.interpreter import LLMInterpreter
from drone_app.llm_clients import GeminiClient, OpenAIClient, AnthropicClient, DummyClient

def main():
    # 0. 環境変数の読み込み (.env ファイルがある場合)
    load_dotenv()

    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description="Drone Telemetry Analysis Pipeline")
    parser.add_argument("ulg_file", help="解析対象の .ulg ファイルへのパス")
    
    # LLMタイプの選択
    parser.add_argument(
        "--llm", 
        choices=["gemini", "openai", "anthropic", "dummy"], 
        default="gemini", 
        help="使用するLLMタイプ (デフォルト: gemini)"
    )
    
    # モデル名指定用の引数
    parser.add_argument(
        "--model", "-m", 
        help="使用するモデル名 (未指定の場合は各クライアントのデフォルトを使用)"
    )
    
    args = parser.parse_args()

    ulg_file = args.ulg_file
    if not os.path.exists(ulg_file):
        print(f"エラー: 指定されたファイルが見つかりません: {ulg_file}")
        return

    # 1. ProfileCoreContext Instantiation
    workspace_dir = "workspace"
    if not os.path.exists(workspace_dir):
        os.makedirs(workspace_dir)
        
    context = ProfileCoreContext(workspace_dir=workspace_dir)
    context.add_log("Pipeline started.")

    # 2. UlgParser による .ulg から DataFrame への変換
    parser_obj = UlgParser(ulg_file)
    context.add_log(f"Parsing ULog file: {ulg_file}")
    
    df = parser_obj.parse(topics=['sensor_combined', 'actuator_outputs'], resample_rate='100ms')
    context.set_data('raw_data', df)
    context.add_log(f"Parsed {len(df)} samples into context with key 'raw_data'")

    csv_path = os.path.join(workspace_dir, "telemetry_data.csv")
    df.to_csv(csv_path)
    context.add_log(f"Raw data saved to: {csv_path}")

    # 3. TelemetryAnalyzer の分析実行
    analyzer = TelemetryAnalyzer(context)
    analyzer.analyze(data_key='raw_data', n_components=3)

    # 4. TelemetryVisualizer によるグラフ出力
    visualizer = TelemetryVisualizer(context, output_dir="output")
    visualizer.plot_raw_telemetry(filename="raw_telemetry.png")
    visualizer.plot_pca_results(filename="pca_plot.png")
    visualizer.plot_variance(filename="pca_variance.png")
    context.add_log("Visualizations generated and saved to 'output/' folder.")

    # 5. LLMクライアントの動的生成とInterpreterによる分析の言語化
    try:
        # クライアントの初期化（モデル名の指定がある場合とない場合に対応）
        client_kwargs = {}
        if args.model:
            client_kwargs['model_name'] = args.model
            
        if args.llm == "gemini":
            client = GeminiClient(**client_kwargs)
        elif args.llm == "openai":
            client = OpenAIClient(**client_kwargs)
        elif args.llm == "anthropic":
            client = AnthropicClient(**client_kwargs)
        elif args.llm == "dummy":
            client = DummyClient(**client_kwargs)
        else:
            raise ValueError(f"Unknown LLM type: {args.llm}")
            
        interpreter = LLMInterpreter(context, llm_client=client)
        # モデル名をファイル名に使用（スラッシュなどはアンダースコアに置換）
        safe_model_name = client.model_name.replace("/", "_").replace(":", "_")
        diag_output_path = f"output/diagnosis_{safe_model_name}.md"
        
        # APIエラー発生時に処理を中断する
        if not interpreter.run_interpretation(output_file=diag_output_path):
            print(f"\n[CRITICAL ERROR] LLM Interpretation failed using client '{args.llm}'. Pipeline aborted.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Failed to initialize LLM client: {str(e)}")
        sys.exit(1)

    # 6. profilecore の ReportExporter による Markdown レポートの出力
    exporter = ReportExporter(context)
    exporter.export_markdown(filename="drone_analysis_report.md")
    context.add_log("Final report generated.")

    print("\n--- Analysis Complete ---")
    print(f"LLM Client: {args.llm} (Model: {client.model_name})")
    print(f"Check 'output/drone_analysis_report.md' for the results.")

if __name__ == "__main__":
    main()
