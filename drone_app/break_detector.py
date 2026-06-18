import pandas as pd
from profilecore.core.module import AnalysisModule

class StructuralBreakAnalyzer(AnalysisModule):
    """
    Analyzes flight history time-series data to detect structural breaks
    (such as irreversible physical degradation or wearing out of motors).
    """
    def __init__(
        self,
        context,
        min_history=5,
        recent_window=2,
        threshold_sigma=2.0,
    ):
        super().__init__(context)
        self.min_history = int(min_history)
        self.recent_window = int(recent_window)
        self.threshold_sigma = float(threshold_sigma)

    def detect_breaks(self):
        """
        Retrieves flight_history from context, performs structural break detection
        on variance columns, and saves detection results to the context.
        """
        self.log("Starting StructuralBreakAnalyzer analysis...")
        
        # 1. Contextから flight_history を取得する
        flight_history = self.context.get_data('flight_history')
        
        required_history = max(self.min_history, self.recent_window + 1)
        if flight_history is None or len(flight_history) < required_history:
            self.log(
                "Data insufficient: "
                f"Less than {required_history} flights in flight history. "
                "Skipping break detection."
            )
            result_dict = {
                'detected': False,
                'status': 'skipped',
                'reason': f'Data insufficient (less than {required_history} flights)',
                'config': self._config_dict(),
            }
            self.context.set_data('structural_break', result_dict)
            return result_dict

        # 2. 対象カラムに対して、簡易的な変化点検知を行う
        # Target columns: any columns that end with '_variance' (e.g. PC1_variance)
        target_cols = [c for c in flight_history.columns if c.endswith('_variance')]
        if not target_cols:
            result_dict = {
                'detected': False,
                'status': 'skipped',
                'reason': 'No variance columns found in flight history',
                'config': self._config_dict(),
            }
            self.context.set_data('structural_break', result_dict)
            return result_dict
        
        detected = False
        break_details = {}
        detected_columns = []
        break_timestamps = []
        
        for col in target_cols:
            series = pd.to_numeric(flight_history[col], errors='coerce').dropna()
            if len(series) < required_history:
                break_details[col] = {
                    'detected': False,
                    'reason': f'Insufficient numeric values (less than {required_history})',
                }
                continue
            
            # Baseline is everything except the recent flights being evaluated.
            baseline = series.iloc[:-self.recent_window]
            recent_values = series.iloc[-self.recent_window:]
            mean_val = float(baseline.mean())
            std_val = float(baseline.std())
            if pd.isna(std_val):
                std_val = 0.0
                
            threshold = mean_val + self.threshold_sigma * std_val
            
            is_break = bool((recent_values > threshold).all())
            
            break_details[col] = {
                'mean': mean_val,
                'std': std_val,
                'threshold': threshold,
                'last_value': float(recent_values.iloc[-1]),
                'prev_value': float(recent_values.iloc[-2]) if len(recent_values) >= 2 else None,
                'recent_values': [float(value) for value in recent_values],
                'baseline_count': int(len(baseline)),
                'recent_window': int(self.recent_window),
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
            'break_timestamp': break_timestamps[-1] if break_timestamps else None,
            'config': self._config_dict(),
        }
        
        self.context.set_data('structural_break', result_dict)
        self.log(f"StructuralBreakAnalyzer completed. Detected: {detected}")
        return result_dict

    def analyze(self):
        return self.detect_breaks()

    def _config_dict(self):
        return {
            'min_history': int(self.min_history),
            'recent_window': int(self.recent_window),
            'threshold_sigma': float(self.threshold_sigma),
        }
