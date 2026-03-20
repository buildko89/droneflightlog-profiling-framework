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

    def analyze(self, data_key='raw_data', n_components=3):
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
        # Drop columns with too many NaNs or constants if any
        # Keep only numeric columns
        df_clean = df.select_dtypes(include=['number']).dropna(axis=1, how='all')
        
        # Fill remaining NaNs if any (though dropna in parser already handled it)
        df_clean = df_clean.ffill().bfill()
        
        # Remove constant columns (StandardScaler fails on these)
        df_clean = df_clean.loc[:, (df_clean != df_clean.iloc[0]).any()]
        
        if df_clean.empty or df_clean.shape[1] < n_components:
            self.log(f"Error: Not enough variation or components for PCA. Shape: {df_clean.shape}")
            return

        self.log(f"Data pre-processed. Shape: {df_clean.shape}")
        
        # 3. Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_clean)
        
        # 4. Perform PCA
        self.log(f"Performing PCA with {n_components} components...")
        pca = PCA(n_components=n_components)
        pca_scores = pca.fit_transform(scaled_data)
        
        # 5. Create results DataFrames
        # PCA Scores
        pca_df = pd.DataFrame(
            pca_scores,
            index=df_clean.index,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
        
        # PCA Explained Variance
        variance_df = pd.DataFrame({
            'Component': [f'PC{i+1}' for i in range(pca.n_components_)],
            'Explained_Variance_Ratio': pca.explained_variance_ratio_
        })
        
        # PCA Loadings (Component weights)
        loadings_df = pd.DataFrame(
            pca.components_,
            index=[f'PC{i+1}' for i in range(n_components)],
            columns=df_clean.columns
        )
        
        # 6. Anomaly Detection (Z-score > 3.0)
        anomaly_timestamps = {}
        for col in pca_df.columns:
            series = pca_df[col]
            # Calculate Z-score
            z_scores = (series - series.mean()) / series.std()
            # Identify indices where absolute Z-score > 3.0
            anomalies = pca_df.index[z_scores.abs() > 3.0]
            
            # Format timestamps to "mm:ss.s" for readability
            # Since index is Timedelta, we format it accordingly
            formatted_times = []
            for t in anomalies:
                total_seconds = t.total_seconds()
                minutes = int(total_seconds // 60)
                seconds = total_seconds % 60
                formatted_times.append(f"{minutes:02d}:{seconds:04.1f}")
            
            anomaly_timestamps[col] = formatted_times

        # 7. Save results back to context
        self.context.set_data('pca_scores', pca_df)
        self.context.set_data('pca_variance', variance_df)
        self.context.set_data('pca_loadings', loadings_df)
        self.context.set_data('anomaly_timestamps', anomaly_timestamps)
        
        self.log(f"PCA completed. Anomaly detected in: {[k for k,v in anomaly_timestamps.items() if v]}")
        self.log("Analysis results saved to context.")
