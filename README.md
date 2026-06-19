# Drone Flight Log Profiling Framework

PX4 ドローンのフライトログ、テレメトリ CSV、動画ファイルを解析する Python ベースの PoC フレームワークです。

主な用途は、フライトログの統計解析、生テレメトリ直接診断、PCA による異常スパイク検出、複数フライト履歴による構造的変化検知、LLM による日本語診断、動画との照合、動画単体解析です。CLI と Streamlit UI の両方から利用できます。

## できること

- PX4 `.ulg` ログの読み込み
- テレメトリ CSV の読み込みと列名マッピング
- ULog トピックの代替解決、複数インスタンス取得、共通時刻グリッド同期
- PCA を使わない生テレメトリ直接要約と AI 診断
- PCA による主成分スコア、寄与率、主成分負荷量の算出
- Z-score による PCA スコアスパイク検出
- ログ側フライトフェーズ推定
- 生テレメトリ、PCA スコア、寄与率のグラフ出力
- 複数フライト履歴の蓄積
- 経年劣化・構造的変化の検知
- Gemini / OpenAI / Anthropic / dummy による LLM 診断
- API を呼ばないプロンプト export モード
- Claude Code / Codex / Agy などのローカルエージェント連携
- 動画付きログ解析
- 動画単体解析
- Markdown レポート出力
- Streamlit Web UI

## 解析モード

このリポジトリには大きく 3 つの解析モードがあります。

| モード | 入力 | エントリポイント | 主な出力 |
| --- | --- | --- | --- |
| ログ解析 | `.ulg` または `.csv` | `main.py` / `dronelog_uiapps.py` | `drone_analysis_report.md` |
| 動画付きログ解析 | `.ulg` / `.csv` + 動画 | `main.py` / `dronelog_uiapps.py` | ログ解析レポート + 動画照合 |
| 動画単体解析 | 動画のみ | `video_main.py` / `video_uiapps.py` | `video_analysis_report.md` |

ログ解析では `--analysis-mode` で解析方式を選択できます。

| 方式 | 内容 | 主な用途 |
| --- | --- | --- |
| `pca` | 従来の PCA 診断。主成分スコア、寄与率、ローディング、PCA スパイクを LLM に渡す | 多変量の変動軸や異常スパイクを見たい場合 |
| `raw` | PCA を使わず、生テレメトリ列の統計、欠損、急変、範囲外候補を LLM に渡す | 元ログに近い形で姿勢、高度、速度、IMU、GPS、バッテリー、モーター出力を確認したい場合 |

## 設計方針

動画とログは混ぜて PCA に投入しません。

```text
ログ解析
  -> PCA / 異常検知 / フライトフェーズ / 経年劣化
  -> または raw モードで生テレメトリ直接診断

動画解析
  -> メタデータ / brightness / blur / motion / video events

最後に照合
  -> Match / Partial Match / Contradiction / No Coverage
```

動画がログ全体の一部しか覆わない場合、動画範囲外のログ異常には `動画による裏付けなし` と明示します。

## ディレクトリ構成

```text
.
├── main.py                         # ログ解析 CLI
├── video_main.py                   # 動画単体解析 CLI
├── dronelog_uiapps.py              # ログ解析 Streamlit UI
├── video_uiapps.py                 # 動画単体解析 Streamlit UI
├── video_log_analysis_usage.md     # 動画+ログ解析の詳しい利用手順
├── llm_config.json                 # 既定 LLM 設定
├── llm_config.dummy.json           # API キー不要の検証用設定
├── requirements.txt
├── drone_app/
│   ├── parser.py                   # ULog 読み込み
│   ├── csv_loader.py               # CSV 読み込み
│   ├── analyzer.py                 # PCA / 異常検知
│   ├── flight_phase_analyzer.py    # ログ側フライトフェーズ推定
│   ├── video_analyzer.py           # 動画特徴量 / 動画イベント
│   ├── pipeline.py                 # ログ解析 pipeline
│   ├── video_pipeline.py           # 動画単体 pipeline
│   ├── report_exporter.py          # ログ解析 Markdown exporter
│   ├── video_report_exporter.py    # 動画単体 Markdown exporter
│   ├── interpreter.py              # ログ解析 LLM 診断
│   ├── raw_interpreter.py          # 生テレメトリ直接 LLM 診断
│   ├── video_interpreter.py        # 動画単体 LLM 診断
│   ├── llm_clients.py              # LLM クライアント
│   ├── visualizer.py               # グラフ生成
│   ├── history_manager.py          # フライト履歴
│   └── break_detector.py           # 構造的変化検知
├── profilecore/                    # 汎用解析基盤
├── tests/                          # テスト
├── workspace/                      # 中間ファイル
└── output/                         # 解析成果物
```

## セットアップ

### 1. Python

Python 3.12 で動作確認しています。

### 2. 依存関係

```powershell
python -m pip install -r requirements.txt
```

動画解析では `opencv-python-headless` を使います。未インストールでもパイプラインは停止せず、動画解析だけ `skipped` としてレポートされます。

### 3. API キー

クラウド LLM API を使う場合は `.env` に設定します。

```text
GEMINI_API_KEY=your_key
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
```

`dummy` または `mode=export` を使う場合、API キーは不要です。

## LLM 設定

`llm_config.json` で LLM サービス、モデル、実行モードを指定できます。

```json
{
  "service": "gemini",
  "model": "gemini-2.5-flash",
  "mode": "api"
}
```

| 項目 | 値 | 説明 |
| --- | --- | --- |
| `service` | `gemini` / `openai` / `anthropic` / `dummy` | 使用する LLM |
| `model` | モデル名 | UI ではプルダウン候補として表示 |
| `mode` | `api` / `export` | API 呼び出しまたはプロンプト書き出し |

`export` は API を呼ばず、ローカルエージェント用のプロンプトを `workspace/` に出力します。

## ログ解析 CLI

基本形:

```powershell
python .\main.py <flight_log.ulg または telemetry.csv>
```

例:

```powershell
python .\main.py .\log_7_2026-3-10-10-46-34.ulg --llm dummy
```

CSV:

```powershell
python .\main.py .\telemetry.csv --llm dummy
```

CSV マッピング付き:

```powershell
python .\main.py .\telemetry.csv --csv-config .\csv_mapping.json --llm dummy
```

PCA を使わずに生テレメトリを直接診断:

```powershell
python .\main.py .\flight.ulg --llm dummy --analysis-mode raw
```

API を呼ばずにプロンプトだけ出す:

```powershell
python .\main.py .\flight.ulg --mode export
```

### ログ解析 CLI オプション

| オプション | 説明 |
| --- | --- |
| `FILE` | `.ulg` または `.csv` |
| `--llm` | `gemini` / `openai` / `anthropic` / `dummy` |
| `--model`, `-m` | モデル名 |
| `--llm-config` | LLM 設定 JSON |
| `--csv-config` | CSV マッピング JSON |
| `--mode` | `api` / `export` |
| `--analysis-mode` | `pca` / `raw`。`raw` は PCA を使わない生テレメトリ直接診断 |
| `--anomaly-z-threshold` | PCA 異常検知 Z-score 閾値 |
| `--break-min-history` | 構造的変化検知に必要な最低履歴数 |
| `--break-threshold-sigma` | 構造的変化検知の閾値倍率 |
| `--flat-output` | `output/runs/...` ではなく `output/` 直下へ出力 |

## 動画付きログ解析 CLI

ログ解析に動画を追加して、ログ異常と動画イベントを照合します。

```powershell
python .\main.py .\flight.ulg `
  --llm dummy `
  --video .\flight_video.mp4 `
  --video-offset-s 1200 `
  --camera-viewpoint external `
  --video-alignment-confidence 0.7
```

1 行で書く場合:

```powershell
python .\main.py .\flight.ulg --llm dummy --video .\flight_video.mp4 --video-offset-s 1200 --camera-viewpoint external --video-alignment-confidence 0.7
```

### 動画付きログ解析オプション

| オプション | 説明 |
| --- | --- |
| `--video` | 照合する動画ファイル |
| `--video-offset-s` | `telemetry_time_s = video_time_s + offset` の offset 秒 |
| `--camera-viewpoint` | `external` または `onboard` |
| `--video-alignment-confidence` | 動画とログの同期信頼度。0.0 から 1.0 |

### カメラ視点

| 値 | 意味 |
| --- | --- |
| `external` | 地上や手持ちカメラから撮影。画面ブレを機体ブレと断定しない |
| `onboard` | 機体搭載カメラ。映像の揺れを機体挙動の補助情報として扱いやすい |

### オフセットの決め方

```text
video_offset_s = telemetry_event_time_s - video_event_time_s
```

例:

- ログ上の離陸開始: 1230 秒
- 動画上の離陸開始: 30 秒

```text
video_offset_s = 1230 - 30 = 1200
```

## 動画単体解析 CLI

動画ファイルだけを解析します。フライトログは不要です。

```powershell
python .\video_main.py .\flight_video.mp4 --camera-viewpoint external
```

サンプリング間隔を指定:

```powershell
python .\video_main.py .\flight_video.mp4 --sample-interval-s 0.5
```

AI 診断も生成:

```powershell
python .\video_main.py .\flight_video.mp4 --llm dummy
```

API を呼ばずに動画診断プロンプトだけ出す:

```powershell
python .\video_main.py .\flight_video.mp4 --mode export --llm dummy
```

### 動画単体 CLI オプション

| オプション | 説明 |
| --- | --- |
| `VIDEO` | 解析対象動画 |
| `--camera-viewpoint` | `external` / `onboard` |
| `--sample-interval-s` | フレームサンプリング間隔秒 |
| `--output-dir` | 出力先 |
| `--workspace-dir` | 作業ディレクトリ |
| `--flat-output` | run ディレクトリを作らず出力 |
| `--llm` | LLM 種別 |
| `--model`, `-m` | モデル名 |
| `--mode` | `api` / `export` |
| `--llm-config` | LLM 設定 JSON |

動画単体解析ではログが無いため、PCA 異常、GPS、IMU、バッテリー異常は判定しません。

## ログ解析 Web UI

起動:

```powershell
python -m streamlit run .\dronelog_uiapps.py
```

画面でできること:

- LLM タイプを選択
- モデルをプルダウンで選択
- `api` / `export` を選択
- 解析方式 `pca` / `raw` を選択
- PCA 異常検知 Z-score 閾値を指定
- 動画同期オフセットを指定
- カメラ視点を指定
- 動画同期信頼度を指定
- `.ulg` / `.csv` をアップロード
- 任意で動画をアップロード
- 解析結果を 4 タブで確認

結果タブ:

- `AI診断結果`
- `統計ビジュアル`
- `詳細データ`
- `経年劣化と飛行履歴`

CSV マッピングを UI で使う場合は、プロジェクト直下に `csv_mapping.json` を置いてください。

## 動画単体 Web UI

起動:

```powershell
python -m streamlit run .\video_uiapps.py
```

画面でできること:

- 動画ファイルをアップロード
- カメラ視点を選択
- サンプリング間隔を指定
- 任意で AI 診断を有効化
- LLM タイプを選択
- モデルをプルダウンで選択
- `api` / `export` を選択
- 動画特徴量、動画イベント、Markdown レポートを確認

結果タブ:

- `Summary`
- `Features`
- `Events`
- `Report`

## CSV 入力

CSV は `CsvTelemetryLoader` で読み込みます。

対応内容:

- timestamp 列の自動推定
- `TimedeltaIndex` への変換
- JSON による列名マッピング
- 数値列の自動変換
- 欠損、定数列、マッピング結果のレポート

timestamp 候補:

- `timestamp`
- `time`
- `time_s`
- `time_sec`
- `time_ms`
- `elapsed_time`
- `datetime`
- `date_time`

マッピング例:

```json
{
  "timestamp_column": "time_s",
  "timestamp_unit": "s",
  "columns": {
    "acc_x": "sensor_combined_accelerometer_m_s2[0]",
    "motor_1": "actuator_outputs_output[0]"
  }
}
```

CLI では `--csv-config` を指定します。

```powershell
python .\main.py .\telemetry.csv --csv-config .\csv_mapping.json --llm dummy
```

UI ではプロジェクト直下の `csv_mapping.json` を自動的に読み込みます。

## 出力物

既定では run ごとに出力ディレクトリを作ります。

```text
output/runs/YYYYMMDD_HHMMSS_<input_stem>/
```

ログ解析の主な出力:

| ファイル | 内容 |
| --- | --- |
| `drone_analysis_report.md` | 統合 Markdown レポート |
| `diagnosis_<model>.md` / `diagnosis.md` | LLM 診断 |
| `raw_telemetry_diagnosis_<model>.md` | raw モードの生テレメトリ直接 LLM 診断 |
| `raw_telemetry.png` | 生テレメトリ時系列 |
| `pca_plot.png` | PCA スコア時系列 |
| `pca_variance.png` | PCA 寄与率 |

動画単体解析の主な出力:

| ファイル | 内容 |
| --- | --- |
| `video_analysis_report.md` | 動画単体 Markdown レポート |
| `video_diagnosis.md` | 動画単体 LLM 診断 |

共有 workspace 出力:

| ファイル | 内容 |
| --- | --- |
| `workspace/<入力名>_telemetry_data_<実行ID>.csv` | 同期済みテレメトリ |
| `workspace/flight_history.csv` | フライト履歴 |
| `workspace/llm_prompt.txt` | ログ解析 export プロンプト |
| `workspace/raw_telemetry_llm_prompt.txt` | raw モードのログ直接診断 export プロンプト |
| `workspace/video_llm_prompt.txt` | 動画単体 export プロンプト |

## Markdown レポートの主なセクション

ログ解析レポート:

- Executive Summary
- Data Quality
- Figures
- PCA Summary
- PCA Loadings
- ULog Parse Report
- CSV Parse Report
- Raw Telemetry Direct Summary
- PCA Anomaly Detection Report
- Structural Break Report
- Telemetry Flight Phases
- Video Summary
- Video Coverage
- Video Events
- Telemetry vs Video
- Warnings
- Appendix

動画単体レポート:

- 概要
- 動画メタデータ
- 動画同期情報
- カバレッジ
- 特徴量統計
- 動画イベント
- AIによる解釈
- 警告
- 付録

## ローカルエージェント連携

Claude Code、Codex、Agy を使う場合は `export` モードを使います。これにより、クラウド API を呼ばずに診断プロンプトをファイルへ出力できます。

### ログ解析プロンプト

```powershell
python .\main.py .\flight.ulg --mode export
```

出力:

```text
workspace/llm_prompt.txt
```

raw モードでは PCA を使わない直接診断プロンプトを出力します。

```powershell
python .\main.py .\flight.ulg --analysis-mode raw --mode export
```

出力:

```text
workspace/raw_telemetry_llm_prompt.txt
```

Claude Code:

```powershell
claude "workspace/llm_prompt.txt を読み込み、指示に従ってドローン解析診断レポートを日本語Markdownで作成し、output/diagnosis_claude.md に保存してください。"
```

Agy:

```powershell
agy run "workspace/llm_prompt.txt を読み込み、ドローン解析診断レポートを日本語Markdownで output/diagnosis_agy.md に作成してください。"
```

Codex:

```text
workspace/llm_prompt.txt を読み込み、ドローン飛行解析の診断レポートを日本語Markdownで作成し、output/diagnosis_codex.md に保存してください。
```

### 動画単体プロンプト

```powershell
python .\video_main.py .\flight_video.mp4 --mode export --llm dummy
```

出力:

```text
workspace/video_llm_prompt.txt
```

Claude Code:

```powershell
claude "workspace/video_llm_prompt.txt を読み込み、動画単体解析レポートを日本語Markdownで output/video_diagnosis_claude.md に保存してください。"
```

Agy:

```powershell
agy run "workspace/video_llm_prompt.txt を読み込み、動画単体解析レポートを日本語Markdownで output/video_diagnosis_agy.md に作成してください。"
```

Codex:

```text
workspace/video_llm_prompt.txt を読み込み、動画単体解析レポートを日本語Markdownで output/video_diagnosis_codex.md に保存してください。
```

## 経年劣化・構造的変化検知

解析ごとに PCA スコア統計を `workspace/flight_history.csv` に追記します。

履歴が十分に蓄積されると、`StructuralBreakAnalyzer` が直近フライト群の分散変化を過去平均との差分で評価します。検知された場合は、Markdown レポートと LLM プロンプトに警告が入ります。

主な調整オプション:

```powershell
python .\main.py .\flight.ulg --break-min-history 5 --break-threshold-sigma 2.0
```

## 検証

全テスト:

```powershell
python -m pytest
```

構文確認:

```powershell
python -m compileall .\main.py .\video_main.py .\dronelog_uiapps.py .\video_uiapps.py .\drone_app .\tests
```

ログ解析 smoke:

```powershell
python .\main.py .\telemetry.csv --llm dummy
```

動画単体 smoke:

```powershell
python .\video_main.py .\flight_video.mp4 --camera-viewpoint external
```

## 注意事項

- Gemini / OpenAI / Anthropic の API 利用には料金が発生する可能性があります。
- `dummy` は動作確認用です。実診断としては使いません。
- `mode=export` は API キーなしで使えます。
- 動画解析には OpenCV が必要です。
- 動画単体解析では、ログ由来の機体異常は判定できません。
- 外部カメラ映像では、手ブレや撮影者操作を機体挙動と断定しないでください。
- 動画とログの同期がずれると、照合結果もずれます。`video_alignment_confidence` を適切に設定してください。

## 参考ドキュメント

- [動画とフライトログ統合解析の利用手順](video_log_analysis_usage.md)
- [動画単体解析 実装プラン](devai/plan20260618_video_only_analysis_codex.md)
- [動画単体解析 実装レポート](devai/implementation20260618_video_only_analysis_codex.md)
