import os
import pandas as pd
from profilecore.core.module import AnalysisModule

class LLMInterpreter(AnalysisModule):
    """
    LLM based analysis interpretation module.
    Delegates text generation to an injected LLM client.
    """
    def __init__(self, context, llm_client):
        super().__init__(context)
        self.llm_client = llm_client

    def run_interpretation(self, output_file="output/diagnosis.md"):
        """
        Retrieves PCA results from context, generates summary statistics,
        and uses the LLM client to interpret the data.
        Returns True if successful, False if error occurs.
        """
        self.log(f"Starting LLM interpretation using client: {self.llm_client.__class__.__name__}...")

        # 1. Retrieve data from context
        pca_variance = self.context.get_data('pca_variance')
        pca_scores = self.context.get_data('pca_scores')
        pca_loadings = self.context.get_data('pca_loadings')
        anomaly_timestamps = self.context.get_data('anomaly_timestamps')
        pca_preprocessing = self.context.get_artifact('pca_preprocessing_report')
        anomaly_config = self.context.get_artifact('anomaly_detection_config')
        video_parse_report = self.context.get_artifact('video_parse_report')
        video_alignment = self.context.get_artifact('video_alignment')
        video_coverage = self.context.get_artifact('video_coverage')
        video_comparison = self.context.get_artifact('telemetry_video_comparison')
        video_events = self.context.get_data('video_events')
        flight_phase_report = self.context.get_artifact('flight_phase_report')
        flight_phases = self.context.get_data('flight_phases')

        # 新たに structural_break と flight_history をContextから取得する（存在する場合）
        structural_break = self.context.get_data('structural_break')
        flight_history = self.context.get_data('flight_history')

        if pca_variance is None or pca_scores is None:
            self.log("Error: PCA results not found in context.")
            return False

        # 2. Calculate basic statistics
        stats_summary = self._calculate_stats(pca_variance, pca_scores, pca_loadings, anomaly_timestamps)
        
        # Add history-related data to stats summary
        stats_summary['structural_break'] = structural_break
        stats_summary['flight_history'] = flight_history
        stats_summary['pca_preprocessing'] = pca_preprocessing
        stats_summary['anomaly_config'] = anomaly_config
        stats_summary['video_parse_report'] = video_parse_report
        stats_summary['video_alignment'] = video_alignment
        stats_summary['video_coverage'] = video_coverage
        stats_summary['video_events'] = video_events
        stats_summary['telemetry_video_comparison'] = video_comparison
        stats_summary['flight_phase_report'] = flight_phase_report
        stats_summary['flight_phases'] = flight_phases

        # 3. Create prompt
        prompt = self._create_prompt(stats_summary)
        
        # Check execution mode: api vs export
        mode = self.context.settings.get('llm_mode', 'api')
        if mode == 'export':
            self.log("LLM Mode is 'export'. Exporting prompt to file...")
            workspace_dir = getattr(self.context, 'workspace_dir', 'workspace')
            prompt_file = os.path.join(workspace_dir, "llm_prompt.txt")
            try:
                os.makedirs(os.path.dirname(prompt_file), exist_ok=True)
                with open(prompt_file, 'w', encoding='utf-8') as f:
                    f.write(prompt)
                self.log(f"Prompt successfully exported to: {prompt_file}")
            except Exception as e:
                self.log(f"Error exporting prompt to {prompt_file}: {str(e)}")
                return False
            
            # Write instructions to the output markdown file
            client_type = self.llm_client.__class__.__name__.replace("Client", "")
            model_name = self.llm_client.model_name
            instructions = f"""# LLMによるドローン解析結果の解釈 (プロンプト・エクスポート)

- **想定クライアント**: {client_type}
- **想定モデル**: {model_name}
- **ステータス**: プロンプト出力済み

本フライト解析の評価用プロンプトがファイルにエクスポートされました。

### 実行方法

ローカルの自律型エージェントツール (Claude Code / Agy / Codex 等) を使用して、以下のファイルに記載されているプロンプトを実行してください。

- **プロンプトファイルパス**: `{prompt_file}`

#### 実行コマンド例（Claude Codeの場合）:
```bash
claude "Read {prompt_file} and generate a detailed flight diagnosis report based on its instructions, saving the result in output/diagnosis_{model_name.replace("/", "_").replace(":", "_")}.md"
```
"""
            text_df = pd.DataFrame({'interpretation': [instructions]})
            self.context.set_data('llm_diagnosis', text_df)
            
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(instructions)
            
            self.log(f"Export placeholder saved to: {output_file}")
            return True

        # 4. Call LLM Client
        try:
            diagnosis_text = self.llm_client.generate_text(prompt)
            self.log("LLM interpretation generated successfully.")
        except Exception as e:
            self.log(f"Error calling LLM client: {str(e)}")
            return False

        # 5. Save results
        client_type = self.llm_client.__class__.__name__.replace("Client", "")
        model_name = self.llm_client.model_name

        # Prepare diagnosis text with metadata
        header_text = f"# LLMによるドローン解析結果の解釈\n\n- **使用AI**: {client_type}\n- **使用モデル**: {model_name}\n\n"
        full_diagnosis = header_text + diagnosis_text

        text_df = pd.DataFrame({'interpretation': [full_diagnosis]})
        self.context.set_data('llm_diagnosis', text_df)
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_diagnosis)
        
        self.log(f"Interpretation saved to: {output_file}")
        return True

    def _calculate_stats(self, variance_df, scores_df, loadings_df=None, anomaly_timestamps=None):
        """
        Calculates basic statistics for PCA results.
        """
        stats = {}
        stats['explained_variance'] = variance_df.to_dict('records')
        
        score_stats = {}
        for col in scores_df.columns:
            series = scores_df[col]
            
            # Extract top 3 features by absolute loading value
            top_features = []
            if loadings_df is not None and col in loadings_df.index:
                pc_loadings = loadings_df.loc[col]
                # Sort by absolute value descending and take top 3
                top_3 = pc_loadings.abs().sort_values(ascending=False).head(3)
                for feat, _ in top_3.items():
                    val = pc_loadings[feat]
                    top_features.append({
                        'feature': feat,
                        'loading': val,
                        'description': self._describe_feature(feat),
                    })

            score_stats[col] = {
                'mean': series.mean(),
                'std': series.std(),
                'min': series.min(),
                'max': series.max(),
                'start_value': series.iloc[0],
                'end_value': series.iloc[-1],
                'trend': series.iloc[-1] - series.iloc[0],
                'top_features': top_features,
                'anomaly_times': anomaly_timestamps.get(col, []) if anomaly_timestamps else []
            }
        stats['score_stats'] = score_stats
        
        return stats

    def _create_prompt(self, stats):
        """
        Generates a prompt for the Gemini API.
        """
        prompt = f"""
あなたはドローンのテレメトリデータ解析のエキスパートです。
以下のPCA（主成分分析）の結果から得られた統計情報に基づき、ドローンの飛行状態や異常の有無、推移について日本語で解説してください。

### 統計情報
1. 寄与率（Explained Variance Ratio）:
{stats['explained_variance']}

### 解析時の前提と注意
- 入力はPX4 ULogまたはテレメトリCSV由来の時系列データです。列名に単位が含まれる場合がありますが、CSVマッピングやログ設定により意味が変わる可能性があります。
- PX4系の代表的な座標は機体座標系またはNED座標系で表現されることがあります。列名だけで座標系を断定しないでください。
- PCAの主成分は統計的な合成軸であり、主成分名だけで故障や物理現象を確定しないでください。ローディング上位特徴量と元系列確認を前提に、可能性として表現してください。
- 異常スパイクはPCAスコアのZ-scoreしきい値による検出です。瞬間的な操作、飛行フェーズ遷移、ログ欠損補完、センサー外乱でも発生し得ます。

"""
        preprocessing = stats.get('pca_preprocessing')
        if preprocessing:
            prompt += "### PCA前処理情報\n"
            prompt += f"- PCA投入列数: {preprocessing.get('selected_column_count')}\n"
            prompt += f"- 要求主成分数/実効主成分数: {preprocessing.get('requested_n_components')} / {preprocessing.get('effective_n_components')}\n"
            prompt += f"- 除外された全NaN列数: {len(preprocessing.get('all_nan_columns', []))}\n"
            prompt += f"- 除外された定数列数: {len(preprocessing.get('constant_columns', []))}\n"
            if preprocessing.get('component_adjustment_reason'):
                prompt += f"- 主成分数調整理由: {preprocessing.get('component_adjustment_reason')}\n"

        anomaly_config = stats.get('anomaly_config')
        if anomaly_config:
            prompt += "\n### 異常検知設定\n"
            prompt += f"- 方法: {anomaly_config.get('method')}\n"
            prompt += f"- Z-score閾値: {anomaly_config.get('z_threshold')}\n"
            prompt += f"- 判定条件: {anomaly_config.get('comparison')}\n"

        video_parse_report = stats.get('video_parse_report')
        if video_parse_report:
            prompt += "\n### 動画解析情報\n"
            prompt += "- 動画はログ解析とは独立した補助情報です。動画範囲外のログ異常について動画から判断しないでください。\n"
            prompt += "- ログ数値を優先し、動画との一致・部分一致・矛盾・判定不能を明示してください。\n"
            prompt += f"- ステータス: {video_parse_report.get('status')}\n"
            if video_parse_report.get('reason'):
                prompt += f"- スキップ理由: {video_parse_report.get('reason')}\n"
            prompt += f"- カメラ視点: {video_parse_report.get('camera_viewpoint', 'external')}\n"
            prompt += f"- 長さ/FPS/解像度: {video_parse_report.get('duration_s')} 秒 / {video_parse_report.get('fps')} / {video_parse_report.get('width')}x{video_parse_report.get('height')}\n"

            video_alignment = stats.get('video_alignment') or {}
            video_coverage = stats.get('video_coverage') or {}
            prompt += (
                f"- 同期方式: {video_alignment.get('mode')} "
                f"(offset={video_alignment.get('video_offset_s')}秒, "
                f"confidence={video_alignment.get('confidence')}, "
                f"window=±{video_alignment.get('event_window_s')}秒)\n"
            )
            prompt += (
                "- 動画カバレッジ: "
                f"{video_coverage.get('start_elapsed_s')}秒 - {video_coverage.get('end_elapsed_s')}秒 "
                f"(coverage_ratio={video_coverage.get('coverage_ratio')})\n"
            )

            video_events = stats.get('video_events')
            if video_events is not None and not video_events.empty:
                prompt += "- 動画イベント（先頭30件）:\n"
                for row in video_events.head(30).to_dict('records'):
                    prompt += f"  - {row}\n"

            comparison = stats.get('telemetry_video_comparison')
            if comparison:
                prompt += "- ログ異常と動画イベントの照合:\n"
                for row in comparison[:30]:
                    prompt += f"  - {row}\n"

            flight_phase_report = stats.get('flight_phase_report')
            if flight_phase_report:
                prompt += "- ログ側フライトフェーズ抽出:\n"
                prompt += f"  - {flight_phase_report}\n"
            flight_phases = stats.get('flight_phases')
            if flight_phases is not None and not flight_phases.empty:
                prompt += "- ログ側フライトフェーズサンプル（先頭30件）:\n"
                for row in flight_phases.head(30).to_dict('records'):
                    prompt += f"  - {row}\n"

        prompt += """
2. 主成分スコアの基本統計量:
"""
        for pc, s in stats['score_stats'].items():
            prompt += f"- {pc}:\n"
            if s['top_features']:
                feats_str = ", ".join([
                    f"{f['feature']} ({f['loading']:.4f}; {f['description']})"
                    for f in s['top_features']
                ])
                prompt += f"  - 主要な構成要素（ローディング上位3つ）: [{feats_str}]\n"
            
            # Add anomaly timestamps
            anomaly_info = ", ".join(s['anomaly_times']) if s['anomaly_times'] else "なし"
            prompt += f"  - 異常（スパイク）が検出された時刻: [{anomaly_info}]\n"
            
            prompt += f"  - 平均: {s['mean']:.4f}\n"
            prompt += f"  - 標準偏差: {s['std']:.4f}\n"
            prompt += f"  - 最小/最大: {s['min']:.4f} / {s['max']:.4f}\n"
            prompt += f"  - 開始/終了値: {s['start_value']:.4f} / {s['end_value']:.4f}\n"
            prompt += f"  - 全体的なトレンド (終了-開始): {s['trend']:.4f}\n"

        # 過去のフライト履歴との比較セクションの追加
        flight_history = stats.get('flight_history')
        structural_break = stats.get('structural_break')

        if flight_history is not None:
            prompt += "\n### 過去のフライト履歴との比較\n"
            prompt += f"- 過去のフライト数: {len(flight_history)}\n"
            prompt += "- 各主成分の履歴データ（分散の直近の推移）:\n"
            var_cols = [c for c in flight_history.columns if c.endswith('_variance')]
            for col in var_cols:
                # Show up to last 10 historical values
                vals = [f"{v:.4f}" for v in flight_history[col].tail(10)]
                vals_str = " -> ".join(vals)
                prompt += f"  - {col}: {vals_str}\n"

        # 構造的変化の警告情報の注入
        break_warning_info = ""
        if isinstance(structural_break, dict):
            if structural_break.get('detected', False):
                break_warning_info = (
                    "\n【警告：構造的変化（経年劣化の兆候）の検出】\n"
                    "過去の飛行履歴と比較して、構造的な変化（経年劣化の兆候）が検知されました。モーター等の摩耗が疑われます。\n"
                    "検知された項目と詳細:\n"
                )
                for col in structural_break.get('detected_columns', []):
                    details = structural_break.get('break_details', {}).get(col, {})
                    recent_values = details.get('recent_values')
                    if recent_values:
                        recent_values_text = ", ".join([f"{value:.4f}" for value in recent_values])
                    else:
                        recent_values_text = f"{details.get('prev_value', 0.0):.4f}, {details.get('last_value', 0.0):.4f}"
                    config = structural_break.get('config', {})
                    sigma = config.get('threshold_sigma', 2.0)
                    break_warning_info += (
                        f"- {col}: 直近フライト群の値 ({recent_values_text}) が、"
                        f"閾値 {details.get('threshold', 0.0):.4f}（過去平均 {details.get('mean', 0.0):.4f} + {sigma}標準偏差）を連続して超えました。\n"
                    )
                break_warning_info += f"変化検出フライト日時: {structural_break.get('timestamp', 'N/A')}\n"
            elif structural_break.get('status') == 'skipped':
                prompt += f"\n- 経年劣化分析ステータス: スキップ ({structural_break.get('reason', '')})\n"
            else:
                prompt += "\n- 経年劣化分析ステータス: 特記すべき構造的変化（経年劣化の兆候）は検出されませんでした。\n"

        prompt += break_warning_info

        prompt += """
### 依頼内容
- 上記の数値から読み取れるドローンの挙動（安定性、特異な変化、トレンドなど）を専門家として分析してください。
- 各主成分（PC1, PC2...）の「主要な構成要素」に着目し、PC1などの抽象的な言葉だけでなく、構成要素のセンサー名を挙げて具体的な物理現象（例：「PC1はZ軸加速度とピッチ角の変動を強く反映しており、高度維持の不安定さを示唆している」など）として解説してください。
- 特定の時刻に異常なスパイクが検出されている場合は、そのタイミングでドローンにどのような物理的衝撃や操作が行われたと推測されるか、時刻を明記して考察してください。
- 主成分スコアの変動（標準偏差の大きさやトレンド）が具体的にどのセンサー群のどのような振る舞いを意味している可能性があるか考察してください。
- 異常が疑われる場合はその旨を指摘しつつ、確定診断ではなく、元系列グラフ・該当時刻前後・機体ログイベントの確認が必要であることを明記してください。
"""

        # 構造的変化がある場合、予知保全のアドバイスを要求
        if isinstance(structural_break, dict) and structural_break.get('detected', False):
            prompt += """- 過去の飛行履歴との比較において構造的変化（経年劣化の兆候）が検知されているため、長期的な予知保全の観点（モーター、ローター、ギアなどの磨耗・劣化）から、今後の保守点検項目や推奨されるアクションのアドバイスを詳しく提示してください。
"""
        else:
            prompt += """- 長期的な安定運用のための予知保全の観点からのアドバイス（点検推奨項目など）があれば含めてください。
"""

        prompt += """- 解説は丁寧な日本語で行ってください。
"""
        return prompt

    def _describe_feature(self, feature):
        lowered = str(feature).lower()
        descriptions = []
        if "accelerometer" in lowered or "accel" in lowered:
            descriptions.append("加速度。列名にm_s2があればm/s^2相当")
        if "gyro" in lowered or "angular_velocity" in lowered:
            descriptions.append("角速度。rad/s系の可能性")
        if "actuator_outputs" in lowered or "output[" in lowered:
            descriptions.append("アクチュエータまたはモーター出力")
        if "vehicle_attitude" in lowered:
            descriptions.append("機体姿勢。クォータニオンまたは姿勢関連値")
        if "local_position" in lowered:
            descriptions.append("ローカル位置/速度。PX4ではNED系の可能性")
        if "global_position" in lowered or "gps" in lowered:
            descriptions.append("GPSまたはグローバル位置情報")
        if "manual_control" in lowered or "input_rc" in lowered:
            descriptions.append("操縦入力またはRC入力")
        if "setpoint" in lowered:
            descriptions.append("制御目標値")
        if not descriptions:
            descriptions.append("列名から用途を明確に断定できない特徴量")
        return "; ".join(descriptions)
