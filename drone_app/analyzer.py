import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from profilecore.core.module import AnalysisModule

class TelemetryAnalyzer(AnalysisModule):
    """
    Telemetry analysis module using PCA.
    Inherits from ProfileCoreContext to manage logs and data.
    """
    def __init__(self, context):
        super().__init__(context)

    def analyze(self, data_key='raw_data', n_components=3, anomaly_z_threshold=3.0):
        """
        Retrieves data from context, performs PCA, and saves results back to context.
        """
        self.log(f"Starting analysis on data key: {data_key}")
        
        # 1. Retrieve data from context
        df = self.context.get_data(data_key)
        if df is None:
            self.log(f"Error: No data found for key '{data_key}'")
            return
        
        # 2. Pre-process data (dropna and scaling)
        numeric_df = df.select_dtypes(include=['number'])
        all_nan_columns = [
            column for column in numeric_df.columns
            if numeric_df[column].isna().all()
        ]
        columns_with_missing = {
            column: int(count)
            for column, count in numeric_df.isna().sum().items()
            if int(count) > 0
        }
        df_clean = numeric_df.dropna(axis=1, how='all')
        
        # Fill remaining NaNs if any (though dropna in parser already handled it)
        df_clean = df_clean.ffill().bfill()
        remaining_missing = int(df_clean.isna().sum().sum())
        
        # Remove constant columns (StandardScaler fails on these)
        if df_clean.empty:
            constant_columns = []
        else:
            constant_columns = [
                column for column in df_clean.columns
                if df_clean[column].dropna().nunique() <= 1
            ]
            df_clean = df_clean.drop(columns=constant_columns)

        effective_n_components = min(n_components, df_clean.shape[1], df_clean.shape[0])
        preprocessing_report = {
            "status": "prepared",
            "data_key": data_key,
            "input_rows": int(len(df)),
            "input_columns": int(len(df.columns)),
            "numeric_column_count": int(len(numeric_df.columns)),
            "numeric_columns": list(numeric_df.columns),
            "all_nan_columns": all_nan_columns,
            "columns_with_missing": columns_with_missing,
            "missing_values_before_fill": int(numeric_df.isna().sum().sum()),
            "missing_values_after_fill": remaining_missing,
            "constant_columns": constant_columns,
            "selected_columns": list(df_clean.columns),
            "selected_column_count": int(df_clean.shape[1]),
            "requested_n_components": int(n_components),
            "effective_n_components": int(effective_n_components),
            "anomaly_z_threshold": float(anomaly_z_threshold),
        }
        
        if effective_n_components < 1:
            message = f"PCA skipped: insufficient numeric variation or components. shape={df_clean.shape}"
            self.log(message)
            preprocessing_report["status"] = "skipped"
            preprocessing_report["reason"] = message
            self.context.set_artifact("pca_preprocessing_report", preprocessing_report)
            if hasattr(self.context, "add_warning"):
                self.context.add_warning(message)
            return

        if effective_n_components < n_components:
            preprocessing_report["component_adjustment_reason"] = (
                "Requested components exceeded available rows or selected numeric features."
            )

        self.log(f"Data pre-processed. Shape: {df_clean.shape}")
        
        # 3. Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_clean)
        
        # 4. Perform PCA
        self.log(f"Performing PCA with {effective_n_components} components...")
        pca = PCA(n_components=effective_n_components)
        pca_scores = pca.fit_transform(scaled_data)
        
        # 5. Create results DataFrames
        # PCA Scores
        pca_df = pd.DataFrame(
            pca_scores,
            index=df_clean.index,
            columns=[f'PC{i+1}' for i in range(effective_n_components)]
        )
        
        # PCA Explained Variance
        variance_df = pd.DataFrame({
            'Component': [f'PC{i+1}' for i in range(pca.n_components_)],
            'Explained_Variance_Ratio': pca.explained_variance_ratio_
        })
        
        # PCA Loadings (Component weights)
        loadings_df = pd.DataFrame(
            pca.components_,
            index=[f'PC{i+1}' for i in range(effective_n_components)],
            columns=df_clean.columns
        )

        loadings_artifact = {}
        for pc_name, row in loadings_df.iterrows():
            sorted_row = row.sort_values()
            loadings_artifact[pc_name] = {
                "negative": [
                    {"feature": feature, "loading": float(value)}
                    for feature, value in sorted_row.head(5).items()
                ],
                "positive": [
                    {"feature": feature, "loading": float(value)}
                    for feature, value in sorted_row.tail(5).sort_values(ascending=False).items()
                ],
            }
        
        # 6. Anomaly Detection
        anomaly_timestamps = {}
        anomaly_details = {}
        for col in pca_df.columns:
            series = pca_df[col]
            # Calculate Z-score
            z_scores = (series - series.mean()) / series.std()
            z_scores = z_scores.fillna(0.0)
            # Identify indices where absolute Z-score exceeds the configured threshold.
            anomaly_mask = z_scores.abs() > anomaly_z_threshold
            anomalies = pca_df.index[anomaly_mask]
            
            # Format timestamps to "mm:ss.s" for readability
            # Since index is Timedelta, we format it accordingly
            formatted_times = []
            for t in anomalies:
                total_seconds = t.total_seconds()
                minutes = int(total_seconds // 60)
                seconds = total_seconds % 60
                formatted_times.append(f"{minutes:02d}:{seconds:04.1f}")
            
            anomaly_timestamps[col] = formatted_times
            anomaly_details[col] = [
                {
                    "timestamp": formatted_time,
                    "z_score": float(z_score),
                    "score": float(score),
                }
                for formatted_time, z_score, score in zip(
                    formatted_times,
                    z_scores[anomaly_mask],
                    series[anomaly_mask],
                )
            ]

        # 7. Save results back to context
        self.context.set_data('pca_scores', pca_df)
        self.context.set_data('pca_variance', variance_df)
        self.context.set_data('pca_loadings', loadings_df)
        self.context.set_data('anomaly_timestamps', anomaly_timestamps)
        preprocessing_report["status"] = "completed"
        self.context.set_artifact("pca_summary", {
            "n_components": int(pca.n_components_),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative_variance": float(pca.explained_variance_ratio_.sum()),
            "sample_count": int(df_clean.shape[0]),
            "input_feature_count": int(df_clean.shape[1]),
        })
        self.context.set_artifact("pca_preprocessing_report", preprocessing_report)
        self.context.set_artifact("pca_loadings", loadings_artifact)
        self.context.set_artifact("anomaly_timestamps", anomaly_timestamps)
        self.context.set_artifact("anomaly_detection_config", {
            "method": "pca_score_zscore",
            "z_threshold": float(anomaly_z_threshold),
            "comparison": "absolute_z_score_greater_than_threshold",
        })
        self.context.set_artifact("anomaly_details", anomaly_details)
        
        self.log(f"PCA completed. Anomaly detected in: {[k for k,v in anomaly_timestamps.items() if v]}")
        self.log("Analysis results saved to context.")
