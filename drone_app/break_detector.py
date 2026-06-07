import pandas as pd
from profilecore.core.module import AnalysisModule

class StructuralBreakAnalyzer(AnalysisModule):
    """
    Analyzes flight history time-series data to detect structural breaks
    (such as irreversible physical degradation or wearing out of motors).
    """
    def __init__(self, context):
        super().__init__(context)

    def detect_breaks(self):
        """
        Retrieves flight_history from context, performs structural break detection
        on variance columns, and saves detection results to the context.
        """
        self.log("Starting StructuralBreakAnalyzer analysis...")
        
        # 1. Contextから flight_history を取得する
        flight_history = self.context.get_data('flight_history')
        
        # データが3フライト分未満の場合は「データ不足」として処理をスキップ
        if flight_history is None or len(flight_history) < 3:
            self.log("Data insufficient: Less than 3 flights in flight history. Skipping break detection.")
            result_dict = {
                'detected': False,
                'status': 'skipped',
                'reason': 'Data insufficient (less than 3 flights)'
            }
            self.context.set_data('structural_break', result_dict)
            return result_dict

        # 2. 対象カラムに対して、簡易的な変化点検知を行う
        # Target columns: any columns that end with '_variance' (e.g. PC1_variance)
        target_cols = [c for c in flight_history.columns if c.endswith('_variance')]
        
        detected = False
        break_details = {}
        detected_columns = []
        break_timestamps = []
        
        for col in target_cols:
            series = flight_history[col]
            
            # Baseline is everything except the last 2 flights being evaluated
            baseline = series.iloc[:-2]
            mean_val = float(baseline.mean())
            std_val = float(baseline.std())
            if pd.isna(std_val):
                std_val = 0.0
                
            threshold = mean_val + 2.0 * std_val
            
            # Get the two most recent values (guaranteed to exist since len >= 5)
            val_last = float(series.iloc[-1])
            val_prev = float(series.iloc[-2])
            
            # Check if they exceed the threshold consecutively (2 times in a row)
            is_break = (val_last > threshold) and (val_prev > threshold)
            
            break_details[col] = {
                'mean': mean_val,
                'std': std_val,
                'threshold': threshold,
                'last_value': val_last,
                'prev_value': val_prev,
                'detected': is_break
            }
            
            if is_break:
                detected = True
                detected_columns.append(col)
                # The break is confirmed at the latest flight timestamp.
                ts_str = str(series.index[-1])
                break_timestamps.append(ts_str)

        # 3. 検知結果と、変化が起きた日時を辞書にまとめ、self.context.set_data('structural_break', result_dict) として保存する。
        result_dict = {
            'detected': detected,
            'status': 'success',
            'detected_columns': detected_columns,
            'break_details': break_details,
            # Use the latest flight's timestamp as the detection timestamp
            'timestamp': str(flight_history.index[-1]),
            # If multiple columns have breaks, use the latest break timestamp
            'break_timestamp': break_timestamps[-1] if break_timestamps else None
        }
        
        self.context.set_data('structural_break', result_dict)
        self.log(f"StructuralBreakAnalyzer completed. Detected: {detected}")
        return result_dict

    def analyze(self):
        return self.detect_breaks()

