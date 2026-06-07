import os
import pandas as pd
from profilecore.core.module import AnalysisModule

class FlightHistoryManager(AnalysisModule):
    """
    Manages flight history by calculating representative statistics for each flight
    and appending them to a cumulative history CSV.
    """
    def __init__(self, context):
        super().__init__(context)

    def update_history(self):
        """
        Retrieves pca_scores and anomaly_timestamps from context, calculates metrics,
        appends to flight_history.csv, and sets 'flight_history' in the context.
        """
        self.log("Starting FlightHistoryManager analysis...")
        
        # 1. Contextから pca_scores と anomaly_timestamps を取得する。
        pca_scores = self.context.get_data('pca_scores')
        anomaly_timestamps = self.context.get_data('anomaly_timestamps')
        
        if pca_scores is None:
            self.log("Error: pca_scores not found in context. Skipping history update.")
            return None

        # 2. 各PCの「分散（Variance）」、「トレンド（終了値 - 開始値）」、「異常（スパイク）の発生回数」を計算する。
        metrics = {}
        for col in pca_scores.columns:
            series = pca_scores[col]
            if len(series) == 0:
                variance = 0.0
                trend = 0.0
            else:
                variance = float(series.var())
                trend = float(series.iloc[-1] - series.iloc[0])
            
            anomaly_count = 0
            if isinstance(anomaly_timestamps, dict) and col in anomaly_timestamps:
                anomaly_count = len(anomaly_timestamps[col])
            
            metrics[f"{col}_variance"] = variance
            metrics[f"{col}_trend"] = trend
            metrics[f"{col}_anomaly_count"] = anomaly_count
        
        # 3. 実行日時をインデックスとして、これらを1行のデータ（DataFrame）にまとめる。
        now = pd.Timestamp.now()
        new_row = pd.DataFrame([metrics], index=[now])
        new_row.index.name = 'timestamp'
        
        # 4. workspace/flight_history.csv が存在すれば読み込んで結合し、なければ新規作成して保存する。
        workspace_dir = getattr(self.context, 'workspace_dir', 'workspace')
        history_file = os.path.join(workspace_dir, "flight_history.csv")
        
        if os.path.exists(history_file):
            try:
                self.log(f"Reading existing flight history from {history_file}")
                history_df = pd.read_csv(history_file, index_col=0, parse_dates=True)
                # Ensure the index name remains consistent
                history_df.index.name = 'timestamp'
                history_df = pd.concat([history_df, new_row])
            except Exception as e:
                self.log(f"Warning: Failed to read {history_file} ({e}). Creating new history file.")
                history_df = new_row
        else:
            self.log(f"Flight history file not found. Creating a new one at {history_file}")
            history_df = new_row
        
        try:
            os.makedirs(os.path.dirname(history_file), exist_ok=True)
            history_df.to_csv(history_file)
            self.log(f"Flight history saved to {history_file}")
        except Exception as e:
            self.log(f"Error saving flight history to {history_file}: {e}")
            
        # 5. 結合された履歴データを self.context.set_data('flight_history', history_df) でContextに保存する。
        self.context.set_data('flight_history', history_df)
        self.log("FlightHistoryManager analysis completed.")
        return history_df

    def analyze(self):
        return self.update_history()

