# ドローン・フライトログ解析パイプライン

PX4 ドローンのフライトログ（`.ulg`）を解析する PoC プロジェクトです。

`.ulg` ファイルからテレメトリデータを抽出し、PCA による統計解析、グラフ生成、LLM による日本語診断、Markdown レポート出力を行います。CLI 実行に加えて、Streamlit によるブラウザ UI からも解析できます。

## 主な機能

- `.ulg` ファイルから `sensor_combined` と `actuator_outputs` を抽出
- 異なる周期のテレメトリを `100ms` 間隔へリサンプリング・同期
- PCA による主成分スコア、寄与率、主成分負荷量の算出
- Z-score による PCA スコアのスパイク時刻検出
- 生テレメトリ、PCA スコア、寄与率の PNG グラフ生成
- Gemini / OpenAI / Anthropic / dummy クライアントによる診断文生成
- `profilecore` の `ProfileCoreContext` と `ReportExporter` を使った Markdown レポート出力
- Streamlit Web UI によるアップロード、解析実行、3タブ結果表示
- `llm_config.json` による LLM サービス・モデル指定

## プロジェクト構成

```text
drone_poc/
├── main.py                    # CLI エントリポイント
├── dronelog_uiapps.py         # Streamlit Web UI
├── llm_config.json            # 既定の LLM 設定
├── llm_config.dummy.json      # APIキー不要の検証用 LLM 設定
├── requirements.txt
├── drone_app/
│   ├── parser.py              # .ulg から DataFrame への変換
│   ├── analyzer.py            # PCA と異常時刻検出
│   ├── visualizer.py          # PNG グラフ生成
│   ├── interpreter.py         # LLM 診断文生成
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

利用する LLM サービスとモデルは `llm_config.json` で指定できます。

```json
{
  "service": "gemini",
  "model": "gemini-2.5-flash"
}
```

`service` に指定できる値:

- `gemini`
- `openai`
- `anthropic`
- `dummy`

`model` を省略するか空欄にすると、各クライアントのデフォルトモデルを使用します。

ネスト形式も利用できます。

```json
{
  "llm": {
    "service": "dummy",
    "model": "dummy-model"
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
python .\main.py <ulg_file_path> [options]
```

主なオプション:

- `--llm`: 使用する LLM サービス。`gemini`, `openai`, `anthropic`, `dummy` から選択
- `--model`, `-m`: 使用するモデル名
- `--llm-config`: LLM 設定 JSON のパス。既定値は `llm_config.json`

実行例:

```powershell
# llm_config.json の設定で実行
python .\main.py .\log_7_2026-3-10-10-46-34.ulg

# APIキー不要の dummy 設定で実行
python .\main.py .\log_7_2026-3-10-10-46-34.ulg --llm-config .\llm_config.dummy.json

# CLI引数で LLM サービスを上書き
python .\main.py .\log_7_2026-3-10-10-46-34.ulg --llm dummy

# Anthropic のモデルを明示指定
python .\main.py .\log_7_2026-3-10-10-46-34.ulg --llm anthropic --model claude-3-haiku-20240307
```

CLI では LLM 診断結果をモデル名付きで出力します。

```text
output/diagnosis_<model>.md
```

## Web UIで実行

Streamlit アプリを起動します。

```powershell
python -m streamlit run .\dronelog_uiapps.py
```

ブラウザで表示された画面から `.ulg` ファイルをアップロードし、`解析実行 (Analyze)` を押します。

Web UI の主な操作:

- サイドバーで LLM タイプを選択
- サイドバーでモデル名を任意指定
- `.ulg` ファイルをアップロード
- 解析実行後、以下の3タブで結果を確認
  - `AI診断結果`
  - `統計ビジュアル`
  - `詳細データ`

Web UI は `st.session_state` で解析結果を保持します。タブ切り替えやサイドバー操作による Streamlit の再実行後も、アップロードファイルが維持されている限り結果表示は残ります。アップロードファイルをクリアすると、古い解析結果もクリアされます。

Web UI では診断結果を固定名で出力します。

```text
output/diagnosis.md
```

## 出力物

共通出力:

- `workspace/telemetry_data.csv`: 同期済みテレメトリデータ
- `output/raw_telemetry.png`: 生テレメトリの時系列グラフ
- `output/pca_plot.png`: PCA スコアの時系列グラフ
- `output/pca_variance.png`: 主成分寄与率
- `output/drone_analysis_report.md`: 統合 Markdown レポート

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
python -m unittest discover -s tests
```

CLI 回帰確認:

```powershell
python .\main.py .\log_7_2026-3-10-10-46-34.ulg --llm-config .\llm_config.dummy.json
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
