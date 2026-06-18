import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from drone_app.llm_clients import SUPPORTED_LLM_TYPES, get_model_choices, load_llm_config
from drone_app.video_pipeline import run_video_only_pipeline


WORKSPACE_DIR = "workspace"
OUTPUT_DIR = "output"
LLM_CONFIG_PATH = "llm_config.json"


def write_uploaded_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return temp_file.name


def render_markdown_file(path):
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as markdown_file:
            st.markdown(markdown_file.read())
    else:
        st.info("Markdown file was not generated.")


def render_download_button(label, path, mime, key):
    if not path or not os.path.exists(path):
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
    tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Features", "Events", "Report"])

    with tab1:
        st.subheader("Video Summary")
        st.json({
            "input_video": context.get_artifact("input_video"),
            "analysis_mode": context.get_artifact("analysis_mode"),
            "video_parse_report": context.get_artifact("video_parse_report"),
            "video_feature_summary": context.get_artifact("video_feature_summary"),
        })

    with tab2:
        st.subheader("Video Features")
        features = context.get_data("video_features")
        if features is not None and not features.empty:
            st.dataframe(features, width="stretch")
            numeric = features[["brightness", "blur_score", "frame_diff", "motion_score"]].copy()
            numeric.index = features["video_time_s"]
            st.line_chart(numeric)
        else:
            st.info("動画特徴量は生成されていません。")

    with tab3:
        st.subheader("Video Events")
        events = context.get_data("video_events")
        if events is not None and not events.empty:
            st.dataframe(events, width="stretch")
        else:
            st.info("動画イベントは推定されていません。")

    with tab4:
        st.subheader("Markdown Report")
        render_markdown_file(results["report_path"])
        render_download_button("レポートをダウンロード", results["report_path"], "text/markdown", "download_video_report")
        render_download_button("AI診断をダウンロード", results.get("diagnosis_path"), "text/markdown", "download_video_diagnosis")


def main():
    load_dotenv()
    st.set_page_config(
        page_title="Video-only Drone Footage Analysis",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if "video_analysis_results" not in st.session_state:
        st.session_state.video_analysis_results = None

    try:
        llm_config = load_llm_config(LLM_CONFIG_PATH)
    except Exception as exc:
        llm_config = {}
        st.sidebar.error(f"LLM設定ファイルの読み込みに失敗しました: {exc}")

    default_llm_type = llm_config.get("service", "dummy")
    if default_llm_type not in SUPPORTED_LLM_TYPES:
        default_llm_type = "dummy"
    default_model_name = llm_config.get("model", "")

    st.sidebar.title("解析設定")
    camera_viewpoint = st.sidebar.radio(
        "カメラ視点",
        options=["external", "onboard"],
        index=0,
        horizontal=True,
    )
    sample_interval_s = st.sidebar.number_input(
        "サンプリング間隔秒",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.1,
    )
    enable_llm = st.sidebar.checkbox("AI診断を生成する", value=False)
    llm_type = st.sidebar.selectbox(
        "LLMタイプ",
        options=list(SUPPORTED_LLM_TYPES),
        index=list(SUPPORTED_LLM_TYPES).index(default_llm_type),
        disabled=not enable_llm,
    )
    configured_model = default_model_name if llm_type == default_llm_type else None
    model_choices = get_model_choices(llm_type, configured_model)
    model_index = model_choices.index(configured_model) if configured_model in model_choices else 0
    model_name = st.sidebar.selectbox(
        "モデル",
        options=model_choices,
        index=model_index,
        disabled=not enable_llm,
        help="llm_config.json に独自モデルが設定されている場合は候補に追加されます。",
    )
    mode = st.sidebar.radio(
        "LLMモード",
        options=["api", "export"],
        horizontal=True,
        disabled=not enable_llm,
    )

    st.title("動画単体解析")
    uploaded_video = st.file_uploader(
        "動画ファイルをアップロードしてください",
        type=["mp4", "mov", "avi", "mkv"],
        accept_multiple_files=False,
    )

    if uploaded_video is None:
        st.session_state.video_analysis_results = None

    analyze_button = st.button("動画解析実行", disabled=uploaded_video is None, type="primary")
    if analyze_button and uploaded_video is not None:
        temp_video_path = None
        try:
            temp_video_path = write_uploaded_file(uploaded_video)
            with st.status("動画解析を実行しています...", expanded=True) as status:
                results = run_video_only_pipeline(
                    temp_video_path,
                    camera_viewpoint=camera_viewpoint,
                    sample_interval_s=sample_interval_s,
                    workspace_dir=WORKSPACE_DIR,
                    output_dir=OUTPUT_DIR,
                    enable_llm=enable_llm,
                    llm_type=llm_type if enable_llm else None,
                    model_name=model_name if enable_llm else None,
                    mode=mode if enable_llm else None,
                    llm_config_path=LLM_CONFIG_PATH,
                    status_callback=status.write,
                )
                status.update(label="動画解析が完了しました。", state="complete")
            st.session_state.video_analysis_results = results
            st.success("動画解析が完了しました。")
        except Exception as exc:
            st.error(f"動画解析中にエラーが発生しました: {exc}")
            with st.expander("詳細エラー"):
                st.exception(exc)
        finally:
            if temp_video_path and os.path.exists(temp_video_path):
                try:
                    os.unlink(temp_video_path)
                except OSError:
                    pass

    if st.session_state.video_analysis_results is not None:
        render_results(st.session_state.video_analysis_results)


if __name__ == "__main__":
    main()
