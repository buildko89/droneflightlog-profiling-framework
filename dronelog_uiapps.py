import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from drone_app.analyzer import TelemetryAnalyzer
from drone_app.csv_loader import CsvTelemetryLoader
from drone_app.interpreter import LLMInterpreter
from drone_app.history_manager import FlightHistoryManager
from drone_app.break_detector import StructuralBreakAnalyzer
from drone_app.llm_clients import (
    SUPPORTED_LLM_TYPES,
    create_llm_client,
    get_required_api_key_name,
    load_llm_config,
    resolve_llm_settings,
)
from drone_app.parser import UlgParser
from drone_app.visualizer import TelemetryVisualizer
from profilecore.core.context import ProfileCoreContext
from profilecore.core.quality import build_data_quality_summary
from profilecore.io.exporter import ReportExporter


WORKSPACE_DIR = "workspace"
OUTPUT_DIR = "output"
DIAGNOSIS_FILENAME = "diagnosis.md"
REPORT_FILENAME = "drone_analysis_report.md"
LLM_CONFIG_PATH = "llm_config.json"
CSV_CONFIG_PATH = "csv_mapping.json"


def validate_api_key(llm_type):
    env_key = get_required_api_key_name(llm_type)
    if env_key and not os.getenv(env_key):
        raise ValueError(f"環境変数 {env_key} が設定されていません。")


def build_summary_insights(context, pca_variance):
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

    return insights


def write_uploaded_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix or ".ulg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return temp_file.name


def run_ui_analysis(file_path, llm_type, model_name, status=None):
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    context = ProfileCoreContext(workspace_dir=WORKSPACE_DIR)
    context.add_log("Streamlit pipeline started.")

    if status:
        status.write("データパース中...")
    if file_path.lower().endswith(".csv"):
        csv_config_path = CSV_CONFIG_PATH if os.path.exists(CSV_CONFIG_PATH) else None
        loader = CsvTelemetryLoader(file_path, config_path=csv_config_path)
        context.add_log(f"Reading CSV file: {file_path}")
        df = loader.load()
        context.set_artifact("csv_parse_report", loader.get_parse_report())
    else:
        parser_obj = UlgParser(file_path)
        context.add_log(f"Parsing ULog file: {file_path}")
        df = parser_obj.parse(resample_rate="100ms")
        context.set_artifact("ulg_parse_report", parser_obj.get_parse_report())
    context.set_data("raw_data", df)
    context.set_artifact("data_quality", build_data_quality_summary(df))
    context.add_log(f"Parsed {len(df)} samples into context with key 'raw_data'")

    csv_path = os.path.join(WORKSPACE_DIR, "telemetry_data.csv")
    df.to_csv(csv_path)
    context.add_log(f"Raw data saved to: {csv_path}")

    if status:
        status.write("PCA分析中...")
    analyzer = TelemetryAnalyzer(context)
    analyzer.analyze(data_key="raw_data", n_components=3)

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

    context.set_artifact("summary_insights", build_summary_insights(context, pca_variance))

    if status:
        status.write("グラフ生成中...")
    visualizer = TelemetryVisualizer(context, output_dir=OUTPUT_DIR)
    visualizer.plot_raw_telemetry(filename="raw_telemetry.png")
    visualizer.plot_pca_results(filename="pca_plot.png")
    visualizer.plot_variance(filename="pca_variance.png")
    context.add_log("Visualizations generated and saved to 'output/' folder.")

    if status:
        status.write("履歴の記録と経年劣化の検知中...")
    
    # 履歴の記録と経年劣化の検知
    FlightHistoryManager(context).update_history()
    StructuralBreakAnalyzer(context).detect_breaks()

    if status:
        status.write("AI診断文生成中...")
        
    # LLMの設定解決
    llm_settings = resolve_llm_settings(
        service=llm_type,
        model_name=model_name or None,
        config_path=LLM_CONFIG_PATH,
    )
    context.set_setting('llm_mode', llm_settings["mode"])

    validate_api_key(llm_type)
    client = create_llm_client(
        service=llm_type,
        model_name=model_name or None,
        config_path=None,
    )
    diagnosis_path = os.path.join(OUTPUT_DIR, DIAGNOSIS_FILENAME)
    interpreter = LLMInterpreter(context, llm_client=client)
    if not interpreter.run_interpretation(output_file=diagnosis_path):
        raise RuntimeError("LLM診断の実行に失敗しました。")

    context.set_artifact("llm_interpretation", {
        "status": "completed",
        "client": llm_type,
        "model": client.model_name,
        "output_file": diagnosis_path,
    })

    if status:
        status.write("Markdownレポート出力中...")
    exporter = ReportExporter(context, output_dir=OUTPUT_DIR)
    exporter.export_markdown(filename=REPORT_FILENAME)
    context.add_log("Final report generated.")

    return {
        "context": context,
        "csv_path": csv_path,
        "diagnosis_path": diagnosis_path,
        "report_path": os.path.join(OUTPUT_DIR, REPORT_FILENAME),
        "raw_telemetry_path": os.path.join(OUTPUT_DIR, "raw_telemetry.png"),
        "pca_plot_path": os.path.join(OUTPUT_DIR, "pca_plot.png"),
        "pca_variance_path": os.path.join(OUTPUT_DIR, "pca_variance.png"),
    }


def render_markdown_file(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as markdown_file:
            st.markdown(markdown_file.read())
    else:
        st.warning(f"ファイルが見つかりません: {path}")


def render_download_button(label, path, mime, key):
    if not os.path.exists(path):
        return
    with open(path, "rb") as output_file:
        st.download_button(
            label=label,
            data=output_file,
            file_name=os.path.basename(path),
            mime=mime,
            key=key,
        )


def render_results(results):
    context = results["context"]
    tab1, tab2, tab3, tab4 = st.tabs(["AI診断結果", "統計ビジュアル", "詳細データ", "経年劣化と飛行履歴"])

    with tab1:
        st.header("AIによるフライト状態の診断")
        render_markdown_file(results["diagnosis_path"])
        render_download_button(
            "AI診断結果をダウンロード",
            results["diagnosis_path"],
            "text/markdown",
            "download_diagnosis_markdown",
        )

    with tab2:
        st.header("テレメトリ可視化グラフ")
        # 左右のカラムに対比しやすい時系列グラフを配置
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists(results["raw_telemetry_path"]):
                st.image(
                    results["raw_telemetry_path"],
                    caption="時系列センサーデータとアクチュエータ出力",
                    width="stretch",
                )
        with col2:
            if os.path.exists(results["pca_plot_path"]):
                st.image(
                    results["pca_plot_path"],
                    caption="主成分スコア時系列推移",
                    width="stretch",
                )
        
        # 寄与率は下に全幅で配置し、バランスを整える
        st.write("---")
        if os.path.exists(results["pca_variance_path"]):
            st.image(
                results["pca_variance_path"],
                caption="各主成分の寄与率",
                width="stretch",
            )

    with tab3:
        st.header("解析詳細データ")

        pca_variance = context.get_data("pca_variance")
        st.subheader("主成分寄与率")
        if pca_variance is not None:
            st.dataframe(pca_variance, width="stretch")
        else:
            st.info("主成分寄与率のデータがありません。")

        st.subheader("主成分負荷量")
        pca_loadings = context.get_data("pca_loadings")
        if pca_loadings is not None:
            st.dataframe(pca_loadings, width="stretch")
        else:
            st.info("主成分負荷量のデータがありません。")

        st.subheader("異常検知タイムスタンプ")
        anomaly_timestamps = context.get_data("anomaly_timestamps")
        if anomaly_timestamps:
            rows = [
                {
                    "主成分": component,
                    "検出スパイク数": len(times),
                    "検出時刻": ", ".join(times) if times else "なし",
                }
                for component, times in anomaly_timestamps.items()
            ]
            st.dataframe(pd.DataFrame(rows), width="stretch")
        else:
            st.info("異常検知タイムスタンプのデータがありません。")

        st.subheader("データ品質")
        data_quality = context.get_artifact("data_quality")
        if data_quality:
            st.json(data_quality)
        else:
            st.info("データ品質サマリーがありません。")

        st.subheader("成果物")
        render_download_button("CSVをダウンロード", results["csv_path"], "text/csv", "download_analysis_csv")
        render_download_button(
            "統合レポートをダウンロード",
            results["report_path"],
            "text/markdown",
            "download_integrated_report",
        )

    with tab4:
        st.header("経年劣化（構造的変化）の分析と履歴")
        
        structural_break = context.get_data('structural_break')
        flight_history = context.get_data('flight_history')
        
        # 1. 判定結果の表示
        if isinstance(structural_break, dict):
            status = structural_break.get('status', 'success')
            detected = structural_break.get('detected', False)
            
            if status == 'skipped':
                st.info(f"📊 経年劣化判定ステータス: スキップ ({structural_break.get('reason', 'データ不足')})")
            else:
                if detected:
                    st.warning("⚠️ 【警告】経年劣化（構造的変化の兆候）が検出されました！")
                    st.write("過去の安定飛行データに基づく閾値（平均 + 2標準偏差）を、直近2フライト連続で超過している項目があります。モーターの摩耗や構造的な異常が疑われます。保守点検・部品交換をご検討ください。")
                else:
                    st.success("✅ 経年劣化判定ステータス: 正常 (特記すべき構造的変化は検出されませんでした)")
                
                # 詳細データ表示
                st.subheader("判定詳細")
                details = []
                for col, det in structural_break.get('break_details', {}).items():
                    details.append({
                        "項目 (分散)": col,
                        "直近の値": f"{det.get('last_value', 0.0):.4f}",
                        "前回の値": f"{det.get('prev_value', 0.0):.4f}",
                        "閾値 (平均+2SD)": f"{det.get('threshold', 0.0):.4f}",
                        "過去平均": f"{det.get('mean', 0.0):.4f}",
                        "標準偏差": f"{det.get('std', 0.0):.4f}",
                        "超過判定": "⚠️ 劣化の疑い" if det.get('detected') else "正常"
                    })
                if details:
                    st.dataframe(pd.DataFrame(details), use_container_width=True)
        else:
            st.info("経年劣化分析データが存在しません。解析を実行してください。")
            
        st.write("---")
        
        # 2. 履歴推移グラフとデータフレーム表示
        if flight_history is not None:
            st.subheader("複数フライトの統計量（分散）の推移")
            var_cols = [c for c in flight_history.columns if c.endswith('_variance')]
            if var_cols:
                try:
                    chart_df = flight_history[var_cols].copy()
                    chart_df.index = pd.to_datetime(chart_df.index)
                    chart_df.index = chart_df.index.strftime('%Y-%m-%d %H:%M')
                    st.line_chart(chart_df)
                except Exception:
                    st.line_chart(flight_history[var_cols])
                
            st.subheader("履歴データ一覧")
            st.dataframe(flight_history, use_container_width=True)
            
            history_file = os.path.join(WORKSPACE_DIR, "flight_history.csv")
            if os.path.exists(history_file):
                render_download_button(
                    "飛行履歴データ (flight_history.csv) をダウンロード",
                    history_file,
                    "text/csv",
                    "download_flight_history_csv"
                )
        else:
            st.info("フライト履歴データが蓄積されていません。")


def main():
    load_dotenv()

    st.set_page_config(
        page_title="profilecore - ドローンフライトログ AI×統計解析ダッシュボード",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # セッション状態の初期化（再実行による描画消失対策）
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None

    try:
        llm_config = load_llm_config(LLM_CONFIG_PATH)
    except Exception as exc:
        llm_config = {}
        st.sidebar.error(f"LLM設定ファイルの読み込みに失敗しました: {exc}")

    default_llm_type = llm_config.get("service", "gemini")
    if default_llm_type not in SUPPORTED_LLM_TYPES:
        default_llm_type = "gemini"
    default_llm_index = list(SUPPORTED_LLM_TYPES).index(default_llm_type)
    default_model_name = llm_config.get("model", "")

    st.sidebar.title("profilecore 解析設定")
    llm_type = st.sidebar.selectbox(
        "LLMタイプの選択",
        options=list(SUPPORTED_LLM_TYPES),
        index=default_llm_index,
        key="sidebar_llm_type",
    )
    model_default = default_model_name if llm_type == default_llm_type else ""
    model_name = st.sidebar.text_input(
        "モデル名の指定",
        value=model_default,
        help="空欄の場合はクライアントのデフォルト値を使用します。",
        key=f"model_name_{llm_type}",
    ).strip()

    if llm_config:
        st.sidebar.caption(f"設定ファイル: {LLM_CONFIG_PATH}")

    if llm_type == "dummy":
        st.sidebar.info("dummy はAPIキーなしで動作確認できます。")

    st.title("profilecore - ドローンフライトログ AI×統計解析ダッシュボード")
    st.write("フライトログ（.ulg または .csv）をアップロードし、PCAとLLMによる診断を実行します。")

    uploaded_file = st.file_uploader(
        "フライトログファイルをアップロードしてください (.ulg, .csv)",
        type=["ulg", "csv"],
        accept_multiple_files=False,
    )

    # アップロードファイルが削除された場合、表示結果をクリアする
    if uploaded_file is None:
        st.session_state.analysis_results = None

    analyze_button = st.button(
        "解析実行 (Analyze)",
        disabled=uploaded_file is None,
        type="primary",
    )

    # 解析実行時の処理（セッション状態へ保存）
    if analyze_button and uploaded_file is not None:
        temp_file_path = None
        try:
            temp_file_path = write_uploaded_file(uploaded_file)
            with st.status("解析を実行しています...", expanded=True) as status:
                results = run_ui_analysis(temp_file_path, llm_type, model_name, status=status)
                status.update(label="解析が完了しました。", state="complete")

            st.session_state.analysis_results = results
            st.success("すべての解析処理が完了しました。")
        except Exception as exc:
            st.error(f"解析中にエラーが発生しました: {exc}")
            with st.expander("詳細エラー"):
                st.exception(exc)
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except OSError:
                    pass

    # セッション状態に結果があれば、ボタン押下状態でなくても常に描画する
    if st.session_state.analysis_results is not None:
        render_results(st.session_state.analysis_results)


if __name__ == "__main__":
    main()
