import os
from datetime import datetime

from profilecore.core.context import ProfileCoreContext

from drone_app.llm_clients import create_llm_client, resolve_llm_settings
from drone_app.video_analyzer import VideoAnalyzer
from drone_app.video_interpreter import VideoOnlyInterpreter
from drone_app.video_report_exporter import VideoOnlyReportExporter


def run_video_only_pipeline(
    video_path,
    *,
    camera_viewpoint="external",
    sample_interval_s=1.0,
    workspace_dir="workspace",
    output_dir="output",
    run_output_subdir=True,
    run_id=None,
    report_filename="video_analysis_report.md",
    diagnosis_filename="video_diagnosis.md",
    enable_llm=False,
    llm_type=None,
    model_name=None,
    mode=None,
    llm_config_path="llm_config.json",
    status_callback=None,
):
    if not video_path:
        raise ValueError("video_path is required for video-only analysis.")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    os.makedirs(workspace_dir, exist_ok=True)
    effective_output_dir = _resolve_output_dir(output_dir, video_path, run_output_subdir, run_id)
    os.makedirs(effective_output_dir, exist_ok=True)

    context = ProfileCoreContext(workspace_dir=workspace_dir)
    context.add_log("Video-only pipeline started.")
    context.set_artifact("input_video", _source_metadata(video_path))
    context.set_artifact("analysis_mode", {
        "mode": "video_only",
        "telemetry_available": False,
        "notes": "No telemetry log was provided. Results are based on video features only.",
    })
    context.set_artifact("run_output", {
        "output_dir": effective_output_dir,
        "base_output_dir": output_dir,
        "run_output_subdir": bool(run_output_subdir),
    })

    _status(status_callback, "動画解析中...")
    VideoAnalyzer(context).analyze(
        video_path,
        telemetry_df=None,
        video_offset_s=0.0,
        camera_viewpoint=camera_viewpoint,
        alignment_confidence=0.0,
        sample_interval_s=sample_interval_s,
    )

    diagnosis_path = None
    llm_settings = None
    llm_client = None
    if enable_llm:
        _status(status_callback, "動画AI診断文生成中...")
        llm_settings = resolve_llm_settings(
            service=llm_type,
            model_name=model_name,
            config_path=llm_config_path,
            mode=mode,
        )
        context.set_setting("llm_mode", llm_settings["mode"])
        llm_client = create_llm_client(
            service=llm_settings["service"],
            model_name=llm_settings["model"],
            config_path=llm_config_path,
            require_api_key=llm_settings["mode"] != "export",
        )
        diagnosis_path = os.path.join(effective_output_dir, diagnosis_filename)
        VideoOnlyInterpreter(context, llm_client).run_interpretation(
            diagnosis_path,
            mode=llm_settings["mode"],
        )

    _status(status_callback, "動画Markdownレポート出力中...")
    report_path = VideoOnlyReportExporter(context, output_dir=effective_output_dir).export_markdown(report_filename)
    context.add_log("Video-only pipeline completed.")

    return {
        "context": context,
        "output_dir": effective_output_dir,
        "report_path": report_path,
        "diagnosis_path": diagnosis_path,
        "llm_settings": llm_settings,
        "llm_client": llm_client,
    }


def _resolve_output_dir(base_output_dir, video_path, run_output_subdir, run_id):
    if not run_output_subdir:
        return base_output_dir

    if not run_id:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_stem = os.path.splitext(os.path.basename(video_path))[0] or "video"
        safe_stem = "".join(
            character if character.isalnum() or character in ("-", "_") else "_"
            for character in file_stem
        ).strip("_") or "video"
        run_id = f"{timestamp}_{safe_stem}"

    return os.path.join(base_output_dir, "runs", run_id)


def _source_metadata(path):
    metadata = {
        "path": path,
        "file_name": os.path.basename(path),
    }
    if os.path.exists(path):
        stat = os.stat(path)
        metadata.update({
            "absolute_path": os.path.abspath(path),
            "size_bytes": int(stat.st_size),
            "modified_time": int(stat.st_mtime),
        })
    return metadata


def _status(callback, message):
    if callback:
        callback(message)
