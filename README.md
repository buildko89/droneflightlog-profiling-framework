# ドローン・テレメトリ解析パイプライン

このプロジェクトは、PX4ドローンのフライトログ（`.ulg`ファイル）を自動でパースし、統計解析およびLLMによる高度な分析解釈を行うためのパイプラインを提供します。
内部ライブラリである `profilecore` をベースとして利用し、コンテキスト管理やレポート出力を継承しつつ、ドローンデータ処理に特化したモジュールを実装しています。

## 主な機能

- **ULogパース**: `.ulg`ファイルから `sensor_combined` や `actuator_outputs` などのトピックを抽出。
- **データ同期**: 異なる周期で記録されたテレメトリデータを `100ms` 間隔で自動的にリサンプリング・同期。
- **統計解析**: `scikit-learn` を使用した主成分分析（PCA）を実行し、飛行データのパターンを抽出。
- **LLM自動解析**: Google Gemini API（1.5 Pro/Flash等）を用いて、PCA結果からドローンの挙動や異常を言語化。
- **自動可視化**: PCAスコア、寄与率、および生データの波形をグラフとして生成。
- **Markdownレポート出力**: 解析ログ、統計サマリー、LLM診断結果を含む包括的なレポートを自動生成。

## プロジェクト構成

- `main.py`: パイプライン全体を制御するメインエントリポイント。
- `drone_app/`: アプリケーションのコアロジック。
    - `parser.py`: `.ulg` から `pandas.DataFrame` への変換を担当。
    - `analyzer.py`: PCAを用いた統計解析を実装。
    - `visualizer.py`: matplotlib/seaborn を用いたグラフ生成を管理。
    - `interpreter.py`: Gemini APIを用いた解析結果の言語化（LLM診断）を担当。
- `profilecore/`: 基盤となる解析ライブラリ（コンテキスト管理、基底クラス、レポート出力）。
- `workspace/`: 処理済みのCSVデータなどの一時保存先。
- `output/`: 生成された画像、LLM診断結果、およびMarkdownレポートの出力先。

## セットアップ

### 1. 依存関係のインストール
Python 3.8以上がインストールされていることを確認してください。以下のコマンドで必要な依存関係をインストールできます。

```bash
pip install -r requirements.txt
```

### 2. 環境変数の設定
LLMによる自動解析機能を利用するには、各サービスのAPIキーが必要です。プロジェクト直下に `.env` ファイルを作成し、使用するサービスに合わせて以下の内容を記述してください。

```text
# 使用するLLMに応じて設定
GEMINI_API_KEY=あなたのAPIキー
OPENAI_API_KEY=あなたのAPIキー
ANTHROPIC_API_KEY=あなたのAPIキー
```

## 使用方法

以下のコマンドを実行し、解析対象の `.ulg` ファイルを引数として指定します。

```bash
python main.py <ulg_file_path> [options]
```

### コマンドライン引数

- `ulg_file_path`: 解析対象の `.ulg` ファイルへのパス（必須）。
- `--llm`: 使用するLLMサービスを選択（任意、デフォルト: `gemini`）。
    - 選択肢: `gemini`, `openai`, `anthropic`, `dummy`
- `--model`, `-m`: 使用するモデル名（任意）。指定しない場合は各クライアントのデフォルトが使用されます。
    - **Gemini (デフォルト)**: `gemini-2.5-flash`
    - **OpenAI**: `gpt-4o`
    - **Anthropic**: `claude-3-5-sonnet-20240620`

### 実行例

```bash
# デフォルト（Gemini / gemini-2.5-flash）で実行
python main.py log_7_2026-3-10-10-46-34.ulg

# OpenAI を使用して実行
python main.py log_7_2026-3-10-10-46-34.ulg --llm openai

# 特定のモデルを指定して Anthropic で実行
python main.py log_7_2026-3-10-10-46-34.ulg --llm anthropic -m claude-3-haiku-20240307

# オフライン・テスト用（APIキー不要）
python main.py log_7_2026-3-10-10-46-34.ulg --llm dummy
```

### 出力物

- **統合レポート**: `output/drone_analysis_report.md`
- **LLM診断結果**: `output/diagnosis.md`（LLMによる詳細な日本語解説）
- **グラフ**:
    - `output/raw_telemetry.png`: センサー値とアクチュエータ出力の時系列グラフ。
    - `output/pca_plot.png`: 主成分スコアの時系列推移。
    - `output/pca_variance.png`: 各主成分の寄与率。
- **データ**: `workspace/telemetry_data.csv`: クレンジングおよび同期済みのデータセット。

## アーキテクチャ

本アプリケーションは、モジュール化されたオブジェクト指向設計を採用しています。
1. **コンテキスト管理**: `ProfileCoreContext` を通じて、モジュール間でデータとログを共有します。
2. **動的なモデル選択**: `LLMInterpreter` はモデル名を引数として受け取り、指定されたAIモデルで柔軟に解析を実行します。
3. **安全な設計**: APIキーの欠損時もパイプライン全体を停止させず、エラーログを出力して処理を継続する頑健性を備えています。

## openai,AnthropicのAPI利用について
本リポジトリでは、Geminiの無料枠でしか動作確認していません。有料となる部分は動作未確認です。
