import os
from datetime import datetime

from profilecore.core.context import ProfileCoreContext
from profilecore.core.quality import build_data_quality_summary

from drone_app.analyzer import TelemetryAnalyzer
from drone_app.break_detector import StructuralBreakAnalyzer
from drone_app.csv_loader import CsvTelemetryLoader
from drone_app.history_manager import FlightHistoryManager
from drone_app.interpreter import LLMInterpreter
from drone_app.llm_clients import create_llm_client, resolve_llm_settings
from drone_app.parser import UlgParser
from drone_app.report_exporter import DroneReportExporter
from drone_app.visualizer import TelemetryVisualizer


def run_analysis_pipeline(
    file_path,
    *,
    llm_type=None,
    model_name=None,
    llm_config_path="llm_config.json",
    csv_config_path=None,
    mode=None,
    workspace_dir="workspace",
    output_dir="output",
    run_output_subdir=True,
    run_id=None,
    diagnosis_filename=None,
    report_filename="drone_analysis_report.md",
    status_callback=None,
    anomaly_z_threshold=3.0,
    history_duplicate_policy="append",
    break_min_history=5,
    break_recent_window=2,
    break_threshold_sigma=2.0,
):
    """
    Runs the full drone telemetry analysis pipeline for CLI and UI callers.
    """
    os.makedirs(workspace_dir, exist_ok=True)
    effective_output_dir = _resolve_output_dir(output_dir, file_path, run_output_subdir, run_id)
    os.makedirs(effective_output_dir, exist_ok=True)

    context = ProfileCoreContext(workspace_dir=workspace_dir)
    context.add_log("Pipeline started.")
    context.set_artifact("input_file", _source_metadata(file_path))
    context.set_artifact("run_output", {
        "output_dir": effective_output_dir,
        "base_output_dir": output_dir,
        "run_output_subdir": bool(run_output_subdir),
    })

    _status(status_callback, "データパース中...")
    df = _load_input_dataframe(file_path, csv_config_path, context)
    context.set_data("raw_data", df)
    context.set_artifact("data_quality", build_data_quality_summary(df))
    context.add_log(f"Parsed {len(df)} samples into context with key 'raw_data'")

    csv_path = os.path.join(workspace_dir, "telemetry_data.csv")
    df.to_csv(csv_path)
    context.add_log(f"Raw data saved to: {csv_path}")

    _status(status_callback, "PCA分析中...")
    analyzer = TelemetryAnalyzer(context)
    analyzer.analyze(
        data_key="raw_data",
        n_components=3,
        anomaly_z_threshold=anomaly_z_threshold,
    )
    pca_variance = context.get_data("pca_variance")
    _set_feature_status(context, pca_variance)
    context.set_artifact("summary_insights", build_summary_insights(context, pca_variance))

    _status(status_callback, "グラフ生成中...")
    visualizer = TelemetryVisualizer(context, output_dir=effective_output_dir)
    visualizer.plot_raw_telemetry(filename="raw_telemetry.png")
    visualizer.plot_pca_results(filename="pca_plot.png")
    visualizer.plot_variance(filename="pca_variance.png")
    context.add_log(f"Visualizations generated and saved to '{effective_output_dir}' folder.")

    _status(status_callback, "履歴の記録と経年劣化の検知中...")
    FlightHistoryManager(context).update_history(
        source_path=file_path,
        duplicate_policy=history_duplicate_policy,
    )
    StructuralBreakAnalyzer(
        context,
        min_history=break_min_history,
        recent_window=break_recent_window,
        threshold_sigma=break_threshold_sigma,
    ).detect_breaks()

    _status(status_callback, "AI診断文生成中...")
    llm_settings = resolve_llm_settings(
        service=llm_type,
        model_name=model_name,
        config_path=llm_config_path,
        mode=mode,
    )
    context.set_setting("llm_mode", llm_settings["mode"])
    client = create_llm_client(
        service=llm_settings["service"],
        model_name=llm_settings["model"],
        config_path=llm_config_path,
        require_api_key=llm_settings["mode"] != "export",
    )

    interpreter = LLMInterpreter(context, llm_client=client)
    if diagnosis_filename:
        diagnosis_path = os.path.join(effective_output_dir, diagnosis_filename)
    else:
        safe_model_name = client.model_name.replace("/", "_").replace(":", "_")
        diagnosis_path = os.path.join(effective_output_dir, f"diagnosis_{safe_model_name}.md")

    if not interpreter.run_interpretation(output_file=diagnosis_path):
        raise RuntimeError(f"LLM Interpretation failed using client '{llm_settings['service']}'.")
    context.set_artifact("llm_interpretation", {
        "status": "completed",
        "client": llm_settings["service"],
        "model": client.model_name,
        "output_file": diagnosis_path,
    })

    _status(status_callback, "Markdownレポート出力中...")
    exporter = DroneReportExporter(context, output_dir=effective_output_dir)
    exporter.export_markdown(filename=report_filename)
    context.add_log("Final report generated.")

    return {
        "context": context,
        "llm_settings": llm_settings,
        "llm_client": client,
        "csv_path": csv_path,
        "diagnosis_path": diagnosis_path,
        "output_dir": effective_output_dir,
        "report_path": os.path.join(effective_output_dir, report_filename),
        "raw_telemetry_path": os.path.join(effective_output_dir, "raw_telemetry.png"),
        "pca_plot_path": os.path.join(effective_output_dir, "pca_plot.png"),
        "pca_variance_path": os.path.join(effective_output_dir, "pca_variance.png"),
    }


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


def _load_input_dataframe(file_path, csv_config_path, context):
    if file_path.lower().endswith(".csv"):
        loader = CsvTelemetryLoader(file_path, config_path=csv_config_path)
        context.add_log(f"Reading CSV file: {file_path}")
        df = loader.load()
        context.set_artifact("csv_parse_report", loader.get_parse_report())
        return df

    parser_obj = UlgParser(file_path)
    context.add_log(f"Parsing ULog file: {file_path}")
    df = parser_obj.parse(resample_rate="100ms")
    context.set_artifact("ulg_parse_report", parser_obj.get_parse_report())
    return df


def _set_feature_status(context, pca_variance):
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


def _source_metadata(file_path):
    metadata = {
        "path": file_path,
        "file_name": os.path.basename(file_path),
    }
    if os.path.exists(file_path):
        stat = os.stat(file_path)
        metadata.update({
            "absolute_path": os.path.abspath(file_path),
            "size_bytes": int(stat.st_size),
            "modified_time": int(stat.st_mtime),
        })
    return metadata


def _resolve_output_dir(base_output_dir, file_path, run_output_subdir, run_id):
    if not run_output_subdir:
        return base_output_dir

    if not run_id:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_stem = os.path.splitext(os.path.basename(file_path))[0] or "flight"
        safe_stem = "".join(
            character if character.isalnum() or character in ("-", "_") else "_"
            for character in file_stem
        ).strip("_") or "flight"
        run_id = f"{timestamp}_{safe_stem}"

    return os.path.join(base_output_dir, "runs", run_id)


def _status(callback, message):
    if callback:
        callback(message)
