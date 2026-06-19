import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from drone_app.llm_clients import (
    SUPPORTED_LLM_TYPES,
    get_model_choices,
    get_required_api_key_name,
    load_llm_config,
    resolve_llm_settings,
)
from drone_app.pipeline import run_analysis_pipeline


WORKSPACE_DIR = "workspace"
OUTPUT_DIR = "output"
DIAGNOSIS_FILENAME = "diagnosis.md"
REPORT_FILENAME = "drone_analysis_report.md"
LLM_CONFIG_PATH = "llm_config.json"
CSV_CONFIG_PATH = "csv_mapping.json"


def validate_api_key(llm_type, mode="api"):
    if mode == "export":
        return
    env_key = get_required_api_key_name(llm_type)
    if env_key and not os.getenv(env_key):
        raise ValueError(f"環境変数 {env_key} が設定されていません。")


def write_uploaded_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix or ".ulg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return temp_file.name


def run_ui_analysis(
    file_path,
    llm_type,
    model_name,
    mode="api",
    anomaly_z_threshold=3.0,
    analysis_mode="pca",
    video_path=None,
    video_offset_s=0.0,
    camera_viewpoint="external",
    video_alignment_confidence=0.5,
    status=None,
):
    llm_settings = resolve_llm_settings(
        service=llm_type,
        model_name=model_name or None,
        config_path=LLM_CONFIG_PATH,
        mode=mode,
    )
    validate_api_key(llm_settings["service"], llm_settings["mode"])
    csv_config_path = CSV_CONFIG_PATH if os.path.exists(CSV_CONFIG_PATH) else None
    return run_analysis_pipeline(
        file_path,
        llm_type=llm_settings["service"],
        model_name=llm_settings["model"],
        llm_config_path=LLM_CONFIG_PATH,
        csv_config_path=csv_config_path,
        mode=llm_settings["mode"],
        workspace_dir=WORKSPACE_DIR,
        output_dir=OUTPUT_DIR,
        diagnosis_filename=DIAGNOSIS_FILENAME,
        report_filename=REPORT_FILENAME,
        anomaly_z_threshold=anomaly_z_threshold,
        analysis_mode=analysis_mode,
        video_path=video_path,
        video_offset_s=video_offset_s,
        camera_viewpoint=camera_viewpoint,
        video_alignment_confidence=video_alignment_confidence,
        status_callback=status.write if status else None,
    )


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
            if results.get("pca_plot_path") and os.path.exists(results["pca_plot_path"]):
                st.image(
                    results["pca_plot_path"],
                    caption="主成分スコア時系列推移",
                    width="stretch",
                )
        
        # 寄与率は下に全幅で配置し、バランスを整える
        st.write("---")
        if results.get("pca_variance_path") and os.path.exists(results["pca_variance_path"]):
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

        st.subheader("動画解析")
        video_report = context.get_artifact("video_parse_report")
        video_coverage = context.get_artifact("video_coverage")
        video_events = context.get_data("video_events")
        telemetry_video_comparison = context.get_artifact("telemetry_video_comparison")
        if video_report:
            st.json({
                "summary": video_report,
                "coverage": video_coverage,
            })
            if video_events is not None and not video_events.empty:
                st.dataframe(video_events, width="stretch")
            if telemetry_video_comparison:
                st.dataframe(pd.DataFrame(telemetry_video_comparison), width="stretch")
        else:
            st.info("動画ファイルは指定されていません。")

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
    default_mode = llm_config.get("mode", "api")
    if default_mode not in ("api", "export"):
        default_mode = "api"

    st.sidebar.title("profilecore 解析設定")
    llm_type = st.sidebar.selectbox(
        "LLMタイプの選択",
        options=list(SUPPORTED_LLM_TYPES),
        index=default_llm_index,
        key="sidebar_llm_type",
    )
    configured_model = default_model_name if llm_type == default_llm_type else None
    model_choices = get_model_choices(llm_type, configured_model)
    model_index = model_choices.index(configured_model) if configured_model in model_choices else 0
    model_name = st.sidebar.selectbox(
        "モデルの選択",
        options=model_choices,
        index=model_index,
        help="llm_config.json に独自モデルが設定されている場合は候補に追加されます。",
        key=f"model_name_{llm_type}",
    )
    mode = st.sidebar.radio(
        "LLM連携モード",
        options=["api", "export"],
        index=["api", "export"].index(default_mode),
        horizontal=True,
        help="export はAPIを呼び出さず、診断用プロンプトをファイルに書き出します。",
        key="sidebar_llm_mode",
    )
    analysis_mode = st.sidebar.radio(
        "解析方式",
        options=["pca", "raw"],
        index=0,
        horizontal=True,
        help="raw は PCA を使わず、生テレメトリ統計だけでAI診断します。",
        key="sidebar_analysis_mode",
    )
    anomaly_z_threshold = st.sidebar.number_input(
        "異常検知Z-score閾値",
        min_value=1.0,
        max_value=10.0,
        value=3.0,
        step=0.1,
        key="anomaly_z_threshold",
    )
    video_offset_s = st.sidebar.number_input(
        "動画同期オフセット秒",
        value=0.0,
        step=1.0,
        help="telemetry_time = video_time + offset として照合します。",
        key="video_offset_s",
    )
    camera_viewpoint = st.sidebar.radio(
        "動画カメラ視点",
        options=["external", "onboard"],
        index=0,
        horizontal=True,
        help="external は地上撮影、onboard は機体搭載カメラとして扱います。",
        key="camera_viewpoint",
    )
    video_alignment_confidence = st.sidebar.slider(
        "動画同期信頼度",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="低い場合、矛盾判定は同期不確実として扱います。",
        key="video_alignment_confidence",
    )

    if llm_config:
        st.sidebar.caption(f"設定ファイル: {LLM_CONFIG_PATH}")

    if mode == "export":
        st.sidebar.info("export はAPIキーなしでプロンプトを書き出します。")
    elif llm_type == "dummy":
        st.sidebar.info("dummy はAPIキーなしで動作確認できます。")

    st.title("profilecore - ドローンフライトログ AI×統計解析ダッシュボード")
    st.write("フライトログ（.ulg または .csv）をアップロードし、PCAまたは生テレメトリ直接要約とLLMによる診断を実行します。")

    uploaded_file = st.file_uploader(
        "フライトログファイルをアップロードしてください (.ulg, .csv)",
        type=["ulg", "csv"],
        accept_multiple_files=False,
    )
    uploaded_video = st.file_uploader(
        "照合用動画ファイルをアップロードしてください（任意）",
        type=["mp4", "mov", "avi", "mkv"],
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
        temp_video_path = None
        try:
            temp_file_path = write_uploaded_file(uploaded_file)
            if uploaded_video is not None:
                temp_video_path = write_uploaded_file(uploaded_video)
            with st.status("解析を実行しています...", expanded=True) as status:
                results = run_ui_analysis(
                    temp_file_path,
                    llm_type,
                    model_name,
                    mode=mode,
                    analysis_mode=analysis_mode,
                    anomaly_z_threshold=anomaly_z_threshold,
                    video_path=temp_video_path,
                    video_offset_s=video_offset_s,
                    camera_viewpoint=camera_viewpoint,
                    video_alignment_confidence=video_alignment_confidence,
                    status=status,
                )
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
            if temp_video_path and os.path.exists(temp_video_path):
                try:
                    os.unlink(temp_video_path)
                except OSError:
                    pass

    # セッション状態に結果があれば、ボタン押下状態でなくても常に描画する
    if st.session_state.analysis_results is not None:
        render_results(st.session_state.analysis_results)


if __name__ == "__main__":
    main()
