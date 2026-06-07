# ドローン・フライトログ解析パイプライン

PX4 ドローンのフライトログ（`.ulg`）およびテレメトリCSVを解析する PoC プロジェクトです。

`.ulg` ファイルまたはCSVファイルからテレメトリデータを抽出し、PCA による統計解析、グラフ生成、LLM による日本語診断、Markdown レポート出力を行います。CLI 実行に加えて、Streamlit によるブラウザ UI からも解析できます。

## 主な機能

- `.ulg` ファイルからテレメトリトピックを抽出
- CSVファイルを標準テレメトリDataFrameへ正規化
- ULogの代替トピック解決、複数インスタンス取得、共通時刻グリッド同期
- 低頻度・単発トピックを全期間へ広げない限定補完
- CSVのtimestamp推定、列名マッピング、数値列検証
- PCA による主成分スコア、寄与率、主成分負荷量の算出
- Z-score による PCA スコアのスパイク時刻検出
- 生テレメトリ、PCA スコア、寄与率の PNG グラフ生成
- Gemini / OpenAI / Anthropic / dummy クライアントによる診断文生成
- `profilecore` の `ProfileCoreContext` と `ReportExporter` を使った Markdown レポート出力
- `ULog Parse Report` / `CSV Parse Report` による読み込み品質の可視化
- Streamlit Web UI によるアップロード、解析実行、3タブ結果表示
- `llm_config.json` による LLM サービス・モデル指定
- 複数フライトにまたがる「経年劣化（構造的変化）」の自動検知と履歴追記管理（CSV保存）
- 経年劣化（構造的変化）検知時のLLMプロンプト警告注入と長期予知保全アドバイスの自動要求

## プロジェクト構成

```text
drone_poc/
├── main.py                    # CLI エントリポイント
├── dronelog_uiapps.py         # Streamlit Web UI
├── csv_mapping.example.json   # CSV列名マッピング設定例
├── llm_config.json            # 既定の LLM 設定
├── llm_config.dummy.json      # APIキー不要の検証用 LLM 設定
├── requirements.txt
├── drone_app/
│   ├── parser.py              # .ulg から DataFrame への変換・同期
│   ├── csv_loader.py          # CSV から DataFrame への変換・正規化
│   ├── analyzer.py            # PCA と異常時刻検出
│   ├── visualizer.py          # PNG グラフ生成
│   ├── interpreter.py         # LLM 診断文生成
│   ├── history_manager.py     # フライトごとの統計履歴追記 (FlightHistoryManager)
│   ├── break_detector.py      # 複数フライト経年劣化検知 (StructuralBreakAnalyzer)
│   └── llm_clients.py         # LLM クライアントと JSON 設定読み込み
├── profilecore/               # 解析基盤ライブラリ
├── tests/                     # 回帰テスト
├── workspace/                 # CSV などの中間成果物
└── output/                    # グラフ、診断、レポート
```

## セットアップ

### 1. 依存関係

Python 3.8 以上を想定しています。

```powershell
python -m pip install -r requirements.txt
```

依存関係は `>=,<次メジャー` 形式で指定しています。再現性を保ちつつ、パッチ・マイナー更新を許容する方針です。

### 2. APIキー

LLM API を使う場合は、プロジェクト直下の `.env` に必要なキーを設定します。

```text
GEMINI_API_KEY=your_key
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
```

`dummy` を使う場合は API キー不要です。

## LLM設定

利用する LLM サービス、モデル、および連携モードは `llm_config.json` で指定できます。

```json
{
  "service": "gemini",
  "model": "gemini-2.5-flash",
  "mode": "api"
}
```

### 設定項目:

- **`service`**: 利用するLLMサービス。`gemini`, `openai`, `anthropic`, `dummy` から指定します。
- **`model`**: 利用するモデル名。省略または空欄の場合、各サービスのデフォルトモデルが使用されます。
- **`mode`**: LLM連携および評価の実行モードを指定します。
  - **`api`** (デフォルト): 直接クラウドLLM APIを呼び出し、自動で診断文を生成して `output/diagnosis_<model>.md` に書き出します。
  - **`export`**: クラウドAPIを呼び出さず、診断用プロンプトを `workspace/llm_prompt.txt` にエクスポートします。出力ファイルには、ローカルエージェントツール等でこのプロンプトを実行するための手順とコマンド例（プレースホルダー）が保存されます。

ネスト形式も利用できます。

```json
{
  "llm": {
    "service": "dummy",
    "model": "dummy-model",
    "mode": "export"
  }
}
```

優先順位は次の通りです。

1. CLI 引数または Web UI の画面入力
2. `llm_config.json`
3. クライアントの組み込みデフォルト

API キーなしで検証する場合は `llm_config.dummy.json` を使います。

## CLIで実行

基本形式:

```powershell
python .\main.py <ulg_or_csv_file_path> [options]
```

主なオプション:

- `--llm`: 使用する LLM サービス。`gemini`, `openai`, `anthropic`, `dummy` から選択
- `--model`, `-m`: 使用するモデル名
- `--mode`: LLM評価モード。`api` (直接APIコール) または `export` (プロンプトファイル書出) から選択
- `--llm-config`: LLM 設定 JSON のパス。既定値は `llm_config.json`
- `--csv-config`: CSVのtimestamp列や列名マッピングを指定するJSONファイル

実行例:

```powershell
# llm_config.json の設定で実行 (デフォルトは API モード)
python .\main.py .\log_7_2026-3-10-10-46-34.ulg

# プロンプトエクスポートモードで実行 (APIを呼び出さずローカル評価用プロンプトを書き出し)
python .\main.py .\log_7_2026-3-10-10-46-34.ulg --mode export

# APIキー不要の dummy 設定で実行
python .\main.py .\log_7_2026-3-10-10-46-34.ulg --llm-config .\llm_config.dummy.json

# CLI引数で LLM サービスを上書き
python .\main.py .\log_7_2026-3-10-10-46-34.ulg --llm dummy

# Anthropic のモデルを明示指定
python .\main.py .\log_7_2026-3-10-10-46-34.ulg --llm anthropic --model claude-3-haiku-20240307

# CSVを列名マッピング付きで実行
python .\main.py .\telemetry.csv --csv-config .\csv_mapping.example.json --llm dummy
```

CLI では LLM 診断結果をモデル名付きで出力します。

```text
output/diagnosis_<model>.md
```

## 入力処理

このプロジェクトでは、入力形式ごとに専用の読み込み層を使い、後段のPCA・可視化・LLM診断には `pandas.DataFrame` として渡します。

```text
.ulg
  -> UlgParser
  -> 共通時刻グリッドへ同期
  -> raw_data
  -> Analyzer / Visualizer / Report

.csv
  -> CsvTelemetryLoader
  -> timestamp正規化・列名マッピング・数値列検証
  -> raw_data
  -> Analyzer / Visualizer / Report
```

### ULog入力

`.ulg` 入力では `drone_app.parser.UlgParser` が以下を行います。

- `DEFAULT_TOPICS` に基づく主要テレメトリトピックの抽出
- `vehicle_global_position` が無い場合の `vehicle_gps_position` / `sensor_gps` への代替解決
- `input_rc` が無い場合の `manual_control_setpoint` への代替解決
- `estimator_states` など同名トピックの複数インスタンス取得
- 全トピックを共通時刻グリッドへリサンプリング
- `fill_strategy="bounded"` による限定的な前方補完
- `ulg_parse_report` の生成

`fill_strategy` は必要に応じて以下を使い分けられます。

| 値 | 挙動 |
|---|---|
| `bounded` | デフォルト。実サンプル間隔に基づいて限定的に前方補完します。1サンプルだけのトピックは全期間へ補完しません。 |
| `none` | 共通グリッドへ揃えるだけで補完しません。 |
| `unbounded` | 従来相当の `ffill().bfill()` を行います。 |

`ulg_parse_report` は最終Markdownの `ULog Parse Report` セクションに出力されます。

## Web UIで実行

Streamlit アプリを起動します。

```powershell
python -m streamlit run .\dronelog_uiapps.py
```

ブラウザで表示された画面から `.ulg` または `.csv` ファイルをアップロードし、`解析実行 (Analyze)` を押します。

Web UI の主な操作:

- サイドバーで LLM タイプを選択
- サイドバーでモデル名を任意指定
- `.ulg` または `.csv` ファイルをアップロード
- 解析実行後、以下の3タブで結果を確認
  - `AI診断結果`
  - `統計ビジュアル`
  - `詳細データ`

Web UI は `st.session_state` で解析結果を保持します。タブ切り替えやサイドバー操作による Streamlit の再実行後も、アップロードファイルが維持されている限り結果表示は残ります。アップロードファイルをクリアすると、古い解析結果もクリアされます。

Web UI では診断結果を固定名で出力します。

```text
output/diagnosis.md
```

CSVの列名マッピングを使う場合は、プロジェクト直下に `csv_mapping.json` を置くとWeb UIが自動で読み込みます。

## CSV入力

CSV入力では、`drone_app.csv_loader.CsvTelemetryLoader` が以下を行います。

- timestamp列の自動推定
- timestampを経過時間の `TimedeltaIndex` へ正規化
- JSON設定による列名マッピング
- 数値化できる文字列列の自動変換
- 行数、数値列数、欠損、定数列、マッピング結果の `csv_parse_report` 生成

timestamp列の自動候補:

- `timestamp`
- `time`
- `time_s`
- `time_sec`
- `time_ms`
- `elapsed_time`
- `datetime`
- `date_time`

先頭列が `Unnamed: 0` の場合は、`DataFrame.to_csv()` で保存されたインデックス列とみなし、timestampとして優先的に扱います。timestamp列が見つからない場合は、行番号を秒として `TimedeltaIndex` を作ります。

マッピング設定例:

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

`timestamp_unit` は `s`, `ms`, `us`, `ns` を指定できます。未指定時は列名から推定し、推定できない数値timestampは秒として扱います。

CSVは列名やフォーマットが現場ごとに変わる前提です。標準的な列名へ寄せたい場合は、CLIでは `--csv-config` を指定し、Web UIではプロジェクト直下に `csv_mapping.json` を置いてください。

`csv_parse_report` は最終Markdownの `CSV Parse Report` セクションに出力されます。

## 経年劣化（構造的変化）検知と飛行履歴管理

フライトごとに主要な統計量を記録・追記し、長期的な変化（経年劣化の兆候など）を自動検知する仕組みを搭載しています。

### 動作フロー

1. **フライト履歴の記録 (`FlightHistoryManager`)**:
   解析実行時に、現在のフライトのPCAスコアから各PCの「分散 (Variance)」「トレンド (Trend)」「異常スパイク回数 (Anomaly count)」を計算し、`workspace/flight_history.csv` に実行日時（タイムスタンプ）をインデックスとして1行追加・保存します。

2. **構造的変化の検知 (`StructuralBreakAnalyzer`)**:
   `flight_history.csv` に蓄積されたデータが **3フライト分以上** ある場合に判定が自動実行されます（3フライト未満の場合は「データ不足」としてスキップされます）。
   - 各PCの分散 (`PC*_variance`) カラムに対して、直近2フライトを除く過去データの「平均値 + 2標準偏差」を閾値とします。
   - 直近2フライトの分散値が連続してこの閾値を超過していた場合、「構造的変化（劣化の兆候）あり」として検知フラグを立て、変化検出日時とともに `structural_break` データとしてContextに保存します。

3. **LLMプロンプトへの警告・指示の注入 (`LLMInterpreter`)**:
   経年劣化や構造的変化が検知された場合、LLMへ送信されるプロンプトに以下の情報が自動的に注入されます。
   - 過去のフライト数の推移履歴情報
   - 構造的変化（経年劣化の兆候）が検知された旨の警告文と、詳細数値（過去平均、閾値、直近2フライトの変動値）
   - モーターやローター等の摩耗を考慮した「長期的な予知保全の観点からの保守点検アドバイス」のLLMへの要求指示

## ローカルエージェントを用いた評価手順（プロンプト・エクスポート）

クラウドLLM API（有料接続）を使用せず、ローカル環境で動作する開発支援エージェント（Claude Code / Agy / Codex 等）を利用してドローンの飛行診断レポート（評価）を行う手順は以下の通りです。

### ステップ 1: 解析パイプラインの実行（プロンプトのエクスポート）
コマンドライン引数で `--mode export` を指定してパイプラインを実行します。

```powershell
python .\main.py <解析対象フライトログファイル.ulg> --mode export
```

これにより、APIを呼び出さずに、今回の解析データ（PCA統計、Z-scoreスパイク、過去の経年劣化判定結果などを含む）から構築された最終プロンプトがファイルにエクスポートされます。

- **エクスポートされたプロンプトファイル**: `workspace/llm_prompt.txt`
- **手順書付きプレースホルダーファイル**: `output/diagnosis_<使用モデル名>.md`

### ステップ 2: ローカルエージェントによる診断レポートの生成

エクスポートされたプロンプトテキストを、普段ご利用のローカルAIエージェントに読み込ませて診断を実行します。

#### 例A: Claude Code (Anthropic CLI) を使用する場合
ターミナル上で `claude` コマンドを使用し、以下のように指示を出します。

```bash
claude "workspace/llm_prompt.txt を読み取り、その指示に従ってドローンの詳細な飛行診断・予知保全レポートを作成し、output/diagnosis_claude.md に保存してください。"
```

Claude Codeが指示内容とデータを理解し、ローカル環境で自動的に適切な Markdown レポートを書き出します。

#### 例B: Agy (Antigravity CLI) を使用する場合
Agyの自律型タスクランナーを使用し、診断レポートの出力を依頼します。

```bash
agy run "workspace/llm_prompt.txt に基づいて、フライトログの評価診断レポートを日本語で output/diagnosis_agy.md に作成せよ。"
```

#### 例C: Codex (Desktop App / CLI) を使用する場合
Codexのチャットあるいはタスク実行コンソールを開き、次のプロンプトを入力してファイルを生成させます。

> 「`workspace/llm_prompt.txt` の内容を読み込んでドローン飛行解析の評価をまとめ、`output/diagnosis_codex.md` にMarkdownファイルとして保存してください。」

### ステップ 3: 最終レポートの確認
エージェントによって診断結果が保存されたら、通常のパイプライン出力と組み合わせて `output/drone_analysis_report.md` 等の最終レポートを確認します。

## 出力物

共通出力:

- `workspace/telemetry_data.csv`: 同期済みテレメトリデータ
- `workspace/flight_history.csv`: 複数フライトの統計量推移履歴
- `output/raw_telemetry.png`: 生テレメトリの時系列グラフ
- `output/pca_plot.png`: PCA スコアの時系列グラフ
- `output/pca_variance.png`: 主成分寄与率
- `output/drone_analysis_report.md`: 統合 Markdown レポート

Markdownレポートには、入力形式に応じて以下の読み込みレポートが追加されます。

- ULog入力: `ULog Parse Report`
- CSV入力: `CSV Parse Report`

LLM 診断出力:

- CLI: `output/diagnosis_<model>.md`
- Web UI: `output/diagnosis.md`

`workspace/` と `output/` は固定出力先です。複数回実行すると前回の同名ファイルは上書きされます。

## 検証

構文チェック:

```powershell
python -m compileall .\main.py .\dronelog_uiapps.py .\drone_app .\profilecore .\tests
```

ユニットテスト:

```powershell
# すべてのユニットテストを実行
python -m unittest discover -s tests

# または pytest を使用して経年劣化検知テストを実行
pytest tests/test_degradation.py
```

CLI 回帰確認:

```powershell
python .\main.py .\log_7_2026-3-10-10-46-34.ulg --llm-config .\llm_config.dummy.json
```

CSV入力確認:

```powershell
python .\main.py .\workspace\telemetry_data.csv --llm dummy
```

Web UI 起動確認:

```powershell
python -m streamlit run .\dronelog_uiapps.py
```

## 注意事項

- Gemini 以外の OpenAI / Anthropic 連携は、有料 API の利用が発生する可能性があります。
- API キー未設定時は、CLI ではエラー終了し、Web UI では画面上にエラーを表示します。
- `dummy` は API キーなしの動作確認用です。実際の診断文は生成しません。
- `llm_config.json` の `service` と CLI の `--llm` が異なる場合、CLI の `--llm` が優先されます。この場合、`--model` を指定しなければ選択サービスのデフォルトモデルを使います。
