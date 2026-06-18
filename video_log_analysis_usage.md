# 動画とフライトログ統合解析の利用手順

このドキュメントは、ドローンのフライトログ `.ulg` / `.csv` と動画ファイル `.mp4` / `.mov` などを同時に解析し、ログ異常と動画イベントの一致・矛盾を確認するための手順です。

## 1. 機能の位置づけ

この実装では、動画特徴量をログの PCA に混ぜません。

解析は次の別トラックで実行されます。

1. ログ解析
   - ULog / CSV の読み込み
   - PCA
   - PCA スコア異常検知
   - ログ側フライトフェーズ推定
   - グラフ生成
   - LLM 診断

2. 動画解析
   - 動画メタデータ取得
   - フレームサンプリング
   - 輝度、ブレ、フレーム差分、モーション量などの抽出
   - 動画イベント推定

3. 照合
   - ログ異常と動画イベントの照合
   - ログ側フライトフェーズと動画イベントの照合
   - 動画カバレッジ外の異常は `動画による裏付けなし` として扱う

## 2. 前提条件

Python 環境に依存関係をインストールします。

```powershell
python -m pip install -r requirements.txt
```

動画解析には `opencv-python-headless` を使用します。

インストールされていない場合でもログ解析自体は停止しませんが、動画解析は `skipped` としてレポートされます。

## 3. 入力ファイル

### 必須

- フライトログ
  - `.ulg`
  - `.csv`

### 任意

- 動画ファイル
  - `.mp4`
  - `.mov`
  - `.avi`
  - `.mkv`

動画ファイルを指定しない場合、従来どおりログ単体解析として動作します。

## 4. 重要な考え方

### カメラ視点

最初に動画の撮影視点を指定します。

| 値 | 意味 | 解釈 |
| --- | --- | --- |
| `external` | 地上や手持ちカメラから撮影 | 動画の揺れは機体の揺れと断定しない |
| `onboard` | 機体搭載カメラ | 動画の揺れを機体挙動の補助情報として扱いやすい |

デフォルトは `external` です。

### 同期オフセット

現時点の実装は手動同期です。

同期式:

```text
telemetry_time_s = video_time_s + video_offset_s
```

例:

- 動画の 0 秒地点が、ログ開始から 1200 秒後に対応する
- この場合 `--video-offset-s 1200` を指定する

### 同期信頼度

一致・矛盾判定は同期精度に依存します。

そのため、同期の確からしさを `0.0` から `1.0` で指定できます。

| 値 | 目安 |
| --- | --- |
| `1.0` | 動画とログの開始時刻が高精度に分かっている |
| `0.7` | 目視や記録から数秒以内で合わせられている |
| `0.5` | おおよその区間は分かるがズレの可能性がある |
| `0.0` | 同期根拠が弱い |

低信頼度の場合、矛盾判定は `Contradiction(low-sync-confidence)` のように表示されます。

## 5. CLI で実行する

### ログ単体解析

```powershell
python .\main.py .\path\to\flight.ulg --llm dummy
```

CSV の場合:

```powershell
python .\main.py .\path\to\telemetry.csv --llm dummy
```

### 動画付き解析

```powershell
python .\main.py .\path\to\flight.ulg `
  --llm dummy `
  --video .\path\to\flight_video.mp4 `
  --video-offset-s 1200 `
  --camera-viewpoint external `
  --video-alignment-confidence 0.7
```

1 行で書く場合:

```powershell
python .\main.py .\path\to\flight.ulg --llm dummy --video .\path\to\flight_video.mp4 --video-offset-s 1200 --camera-viewpoint external --video-alignment-confidence 0.7
```

### LLM API を使わずプロンプトだけ出力する

```powershell
python .\main.py .\path\to\flight.ulg `
  --mode export `
  --video .\path\to\flight_video.mp4 `
  --video-offset-s 1200 `
  --camera-viewpoint external
```

この場合、API は呼ばずに `workspace/llm_prompt.txt` に診断用プロンプトが出力されます。

## 6. CLI オプション

| オプション | 必須 | デフォルト | 説明 |
| --- | --- | --- | --- |
| `FILE` | 必須 | なし | 解析対象の `.ulg` または `.csv` |
| `--video` | 任意 | なし | 照合する動画ファイル |
| `--video-offset-s` | 任意 | `0.0` | 動画 0 秒地点がログ開始から何秒後か |
| `--camera-viewpoint` | 任意 | `external` | `external` または `onboard` |
| `--video-alignment-confidence` | 任意 | `0.5` | 動画とログの同期信頼度 |
| `--llm` | 任意 | 設定ファイル依存 | `gemini` / `openai` / `anthropic` / `dummy` |
| `--mode` | 任意 | 設定ファイル依存 | `api` または `export` |
| `--csv-config` | 任意 | なし | CSV 列マッピング設定 |
| `--anomaly-z-threshold` | 任意 | `3.0` | PCA 異常検知の Z-score 閾値 |
| `--flat-output` | 任意 | 無効 | `output/runs/...` ではなく `output/` 直下に出力 |

## 7. Streamlit UI で実行する

Streamlit アプリを起動します。

```powershell
python -m streamlit run .\dronelog_uiapps.py
```

ブラウザ画面で次を設定します。

1. サイドバーで LLM タイプとモードを選択
2. `異常検知Z-score閾値` を設定
3. `動画同期オフセット秒` を設定
4. `動画カメラ視点` を `external` または `onboard` から選択
5. `動画同期信頼度` を設定
6. フライトログファイルをアップロード
7. 必要に応じて動画ファイルをアップロード
8. `解析実行 (Analyze)` を押す

動画ファイルをアップロードしない場合は、ログ単体解析として実行されます。

## 8. 出力先

通常は次のディレクトリに run ごとの成果物が出力されます。

```text
output/runs/<timestamp>_<input_file_stem>/
```

主な出力:

| ファイル | 内容 |
| --- | --- |
| `drone_analysis_report.md` | 統合 Markdown レポート |
| `diagnosis_*.md` または `diagnosis.md` | LLM 診断結果 |
| `raw_telemetry.png` | ログ時系列グラフ |
| `pca_plot.png` | PCA スコア推移 |
| `pca_variance.png` | PCA 寄与率 |

`--flat-output` を指定した場合は `output/` 直下に出力されます。

## 9. レポートの読み方

### Video Summary

動画メタデータと解析状態を確認します。

主な項目:

- `status`
- `file_name`
- `camera_viewpoint`
- `duration_s`
- `fps`
- `width`
- `height`
- `codec`
- `feature_rows`
- `event_count`

`status` が `skipped` の場合、動画解析は実行されていません。

### Video Coverage

動画がログ全体のどの範囲をカバーしているかを確認します。

主な項目:

- `start_elapsed_s`
- `end_elapsed_s`
- `duration_s`
- `telemetry_duration_s`
- `coverage_ratio`

例:

```text
start_elapsed_s = 1200.0
end_elapsed_s = 1407.0
coverage_ratio = 0.047
```

この場合、ログ開始から 1200 秒から 1407 秒までが動画範囲です。

### Video Events

動画特徴量から推定されたイベントです。

主なイベント:

- `hover`
- `rapid_movement`
- `lateral_motion`
- `forward_motion`
- `visibility_loss`
- `severe_blur`

現時点ではローカル CV による軽量ヒューリスティックです。

確定的な意味ではなく、ログとの照合に使う補助信号として見てください。

### Telemetry Flight Phases

ログ側から推定されたフライトフェーズです。

主なフェーズ:

- `ground`
- `takeoff`
- `hover`
- `moving`
- `landing`
- `unknown`

高度、垂直速度、スロットル推定から粗く分類します。

### Telemetry vs Video

ログ解析と動画解析の照合結果です。

主な列:

| 列 | 意味 |
| --- | --- |
| `Time` | ログ上の時刻 |
| `Log` | ログ側の異常またはフェーズ |
| `Video` | 対応する動画イベント |
| `Result` | 照合結果 |
| `Alignment Confidence` | 同期信頼度 |
| `Tolerance Window s` | 照合に使った時間窓 |
| `Comment` | 補足 |

主な `Result`:

| 値 | 意味 |
| --- | --- |
| `Match` | ログと動画が整合 |
| `Partial Match` | 一部整合 |
| `Contradiction` | ログと動画が矛盾する可能性 |
| `Contradiction(low-sync-confidence)` | 同期信頼度が低いため要確認の矛盾 |
| `Undetermined` | 動画範囲内だが対応イベントなし |
| `Undetermined(low-sync-confidence)` | 同期信頼度が低く判定保留 |
| `No Coverage` | 動画範囲外 |

`No Coverage` の場合は、動画による裏付けはありません。

## 10. オフセットの決め方

現時点では手動で `--video-offset-s` を指定します。

### 方法 1: 撮影メモから決める

飛行開始から何秒後に動画撮影を始めたか分かる場合、その秒数を指定します。

例:

```powershell
--video-offset-s 1200
```

### 方法 2: ログのイベント時刻と動画の目視イベントを合わせる

1. レポートやグラフで、離陸、着陸、大きな操作などのログ時刻を確認
2. 動画内で同じ出来事が見える動画時刻を確認
3. 次の式でオフセットを計算

```text
video_offset_s = telemetry_event_time_s - video_event_time_s
```

例:

- ログ上の離陸開始: 1230 秒
- 動画上の離陸開始: 30 秒

```text
video_offset_s = 1230 - 30 = 1200
```

この場合:

```powershell
--video-offset-s 1200
```

## 11. 推奨ワークフロー

1. まずログ単体で解析する

```powershell
python .\main.py .\path\to\flight.ulg --llm dummy
```

2. PCA 異常時刻やフライトフェーズを確認する

3. 動画の該当区間を目視し、`video_offset_s` を決める

4. 動画付きで再解析する

```powershell
python .\main.py .\path\to\flight.ulg --llm dummy --video .\path\to\flight_video.mp4 --video-offset-s 1200 --camera-viewpoint external --video-alignment-confidence 0.7
```

5. `Telemetry vs Video` を確認する

6. `No Coverage` の異常には動画コメントを付けず、ログ側の根拠で評価する

7. `Contradiction` または `Contradiction(low-sync-confidence)` は追加調査対象にする

## 12. 注意点

- 動画はログ全体の一部だけをカバーすることが多いです。
- 動画範囲外の異常は、必ず `動画による裏付けなし` として扱います。
- `external` 視点では、手持ちカメラの揺れと機体の揺れを混同しないでください。
- ローカル CV の動画イベントは軽量な推定です。確定診断ではありません。
- 同期オフセットがずれると、一致・矛盾判定もずれます。
- 低い同期信頼度で出た矛盾は、真の矛盾ではなく同期ズレの可能性があります。
- LLM 診断ではログ数値を基準にしつつ、動画との矛盾は調査フラグとして扱います。

## 13. トラブルシュート

### 動画解析が `skipped` になる

原因候補:

- 動画ファイルのパスが間違っている
- OpenCV がインストールされていない
- 動画ファイルを OpenCV が開けない

確認:

```powershell
python -m pip install -r requirements.txt
```

### `No Coverage` が多い

動画がログの一部しかカバーしていない可能性があります。

`Video Coverage` の `start_elapsed_s` / `end_elapsed_s` / `coverage_ratio` を確認してください。

### 矛盾判定が多い

原因候補:

- `--video-offset-s` がずれている
- `--video-alignment-confidence` が実態より高すぎる
- 外部カメラの動きを機体挙動として過剰に解釈している

対策:

- オフセットを再計算する
- 同期信頼度を下げる
- `camera_viewpoint` が正しいか確認する

### LLM API を使いたくない

`dummy` または `export` モードを使います。

```powershell
python .\main.py .\path\to\flight.ulg --llm dummy --video .\path\to\flight_video.mp4 --video-offset-s 1200
```

または:

```powershell
python .\main.py .\path\to\flight.ulg --mode export --video .\path\to\flight_video.mp4 --video-offset-s 1200
```

## 14. 現時点で未対応の範囲

次の機能は設計上の将来拡張です。

- 動画 `creation_time` と ULog GPS UTC による絶対時刻同期
- 離陸イベントによる自動同期
- Vision LLM による高信頼な動画イベント説明
- FPV 向けの Optical Flow / IMU / Audio 相関
- 動画特徴量を PCA に入れる明示オプトイン

現時点では、手動オフセットを指定したうえで、ログ解析結果と動画イベントを別トラックで照合する使い方が基本です。
