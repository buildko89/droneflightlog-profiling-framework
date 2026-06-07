# 依頼内容
ドローンフライトログ解析パイプラインのWeb UIアプリケーション（Streamlit）を実装するためのPythonコードを記述してください。
既存の `profilecore` ライブラリおよび `drone_app` のモジュール群（`UlgParser`, `TelemetryAnalyzer`, `TelemetryVisualizer`, `LLMInterpreter`）のコードは一切変更せず、これらをインポートして利用する設計にしてください。

# 1. 作成するファイル名: `dronelog_uiapps.py`

# 2. UI/UX 構成要件
Streamlitを用いて、以下の要素を持つプロフェッショナルなダッシュボードを構築してください。

### ① サイドバー設定 (Sidebar)
- タイトル: 「profilecore 解析設定」
- LLMタイプの選択: `st.selectbox` を用いて、`gemini`, `openai`, `anthropic`, `dummy` から選択できるようにする（デフォルト: `gemini`）。
- モデル名の指定: `st.text_input` を用いて、使用するモデル名を動的に入力できるようにする（空欄の場合はクライアントのデフォルト値を使用）。

### ② メイン画面 (Main Body)
- ヘッダー: 「profilecore — ドローンフライトログ AI×統計解析ダッシュボード」
- ファイルアップローダー: `st.file_uploader` を配置し、`.ulg` ファイルのみを受け付けるようにする。
- 解析ボタン: 「解析実行 (Analyze)」ボタンを配置する。

### ③ 解析実行時の処理とインターフェース
「解析実行」ボタンが押されたら、以下のパイプライン処理を一気通貫で実行してください。
1. `st.spinner` やプログレス表示を用いて、現在の処理状況（データパース中... -> PCA分析中... -> グラフ生成中... -> AI診断文生成中...）をユーザーに視覚的に伝える。
2. アップロードされた一時ファイルを `UlgParser` に渡し、`workspace/` 内にデータをパース、`output/` 内にグラフ（PNG）や診断書（MD）を出力する。
3. すべての処理が正常に完走したら、画面を以下の3つのタブ（`st.tabs`）に分けて結果を表示する。

### ④ 結果表示の3大タブ構成
- **【タブ1: AI診断結果】**
  - 最上部に配置。生成された `output/diagnosis.md` の内容を `st.markdown()` を用いてリッチに表示する。
- **【タブ2: 統計ビジュアル】**
  - `output/pca_plot.png`, `output/pca_variance.png`, `output/raw_telemetry.png` の3つの画像を `st.image()` を使って綺麗に並べて表示する。
- **【タブ3: 詳細データ】**
  - `ProfileCoreContext` から `pca_variance`、`pca_loadings`、`anomaly_timestamps` などのデータを取得し、`st.dataframe()` や `st.table()` を用いて、見やすい表形式で展開する。

# 3. 安全・堅牢性の要件
- 処理の最初に `python-dotenv` の `load_dotenv()` を呼び出し、`.env` ファイルから各LLMのAPIキーを自動で読み込むようにしてください。
- 必要なAPIキーが環境変数に設定されていない場合や、パースエラーが起きた場合は、画面上に `st.error()` で分かりやすくエラーメッセージを表示し、アプリがクラッシュしないように例外処理（try-except）を徹底してください。

# 出力形式
- `dronelog_uiapps.py` の完全なPythonコードを出力してください。
