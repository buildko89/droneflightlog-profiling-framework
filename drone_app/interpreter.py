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

        if pca_variance is None or pca_scores is None:
            self.log("Error: PCA results not found in context.")
            return False

        # 2. Calculate basic statistics
        stats_summary = self._calculate_stats(pca_variance, pca_scores, pca_loadings, anomaly_timestamps)
        
        # 3. Create prompt
        prompt = self._create_prompt(stats_summary)
        
        # 4. Call LLM Client
        try:
            diagnosis_text = self.llm_client.generate_text(prompt)
            self.log("LLM interpretation generated successfully.")
        except Exception as e:
            self.log(f"Error calling LLM client: {str(e)}")
            return False

        # 5. Save results
        text_df = pd.DataFrame({'interpretation': [diagnosis_text]})
        self.context.set_data('llm_diagnosis', text_df)
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# LLMによるドローン解析結果の解釈\n\n")
            f.write(diagnosis_text)
        
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
                    top_features.append({'feature': feat, 'loading': val})

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

2. 主成分スコアの基本統計量:
"""
        for pc, s in stats['score_stats'].items():
            prompt += f"- {pc}:\n"
            if s['top_features']:
                feats_str = ", ".join([f"{f['feature']} ({f['loading']:.4f})" for f in s['top_features']])
                prompt += f"  - 主要な構成要素（ローディング上位3つ）: [{feats_str}]\n"
            
            # Add anomaly timestamps
            anomaly_info = ", ".join(s['anomaly_times']) if s['anomaly_times'] else "なし"
            prompt += f"  - 異常（スパイク）が検出された時刻: [{anomaly_info}]\n"
            
            prompt += f"  - 平均: {s['mean']:.4f}\n"
            prompt += f"  - 標準偏差: {s['std']:.4f}\n"
            prompt += f"  - 最小/最大: {s['min']:.4f} / {s['max']:.4f}\n"
            prompt += f"  - 開始/終了値: {s['start_value']:.4f} / {s['end_value']:.4f}\n"
            prompt += f"  - 全体的なトレンド (終了-開始): {s['trend']:.4f}\n"

        prompt += """
### 依頼内容
- 上記の数値から読み取れるドローンの挙動（安定性、特異な変化、トレンドなど）を専門家として分析してください。
- 各主成分（PC1, PC2...）の「主要な構成要素」に着目し、PC1などの抽象的な言葉だけでなく、構成要素のセンサー名を挙げて具体的な物理現象（例：「PC1はZ軸加速度とピッチ角の変動を強く反映しており、高度維持の不安定さを示唆している」など）として解説してください。
- 特定の時刻に異常なスパイクが検出されている場合は、そのタイミングでドローンにどのような物理的衝撃や操作が行われたと推測されるか、時刻を明記して考察してください。
- 主成分スコアの変動（標準偏差の大きさやトレンド）が具体的にどのセンサー群のどのような振る舞いを意味している可能性があるか考察してください。
- 異常が疑われる場合はその旨を指摘してください。
- 解説は丁寧な日本語で行ってください。
"""
        return prompt
