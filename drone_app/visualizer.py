import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

class TelemetryVisualizer:
    """
    Visualizer class for ProfileCoreContext analysis results.
    Generates plots using matplotlib and seaborn.
    """
    def __init__(self, context, output_dir='output'):
        self.context = context
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def plot_pca_results(self, filename='pca_plot.png'):
        """
        Plots PCA component scores over time or scatter.
        """
        pca_scores = self.context.get_data('pca_scores')
        if pca_scores is None:
            print("Error: No PCA scores found in context.")
            return

        plt.figure(figsize=(12, 6))
        
        # Plot each principal component over time
        for col in pca_scores.columns:
            plt.plot(pca_scores.index.total_seconds(), pca_scores[col], label=f'{col}')

        plt.title('Principal Components Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Component Score')
        plt.legend()
        plt.grid(True)
        
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"PCA visualization saved to: {save_path}")

    def plot_raw_telemetry(self, filename='raw_telemetry.png'):
        """
        Plots representative raw telemetry data from context.
        """
        df = self.context.get_data('raw_data')
        if df is None:
            print("Error: No raw data found in context.")
            return

        # Choose a subset of columns to plot (limit to top 10 if too many)
        cols_to_plot = [col for col in df.columns if 'accelerometer' in col or 'output' in col][:8]
        
        if not cols_to_plot:
            cols_to_plot = df.columns[:8]

        plt.figure(figsize=(12, 8))
        for i, col in enumerate(cols_to_plot):
            plt.subplot(len(cols_to_plot), 1, i+1)
            plt.plot(df.index.total_seconds(), df[col])
            plt.ylabel(col.split('_')[-1])
            plt.title(col, fontsize=10)
            plt.grid(True)

        plt.xlabel('Time (s)')
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Raw telemetry visualization saved to: {save_path}")

    def plot_variance(self, filename='pca_variance.png'):
        """
        Plots the explained variance ratio of PCA components.
        """
        variance_df = self.context.get_data('pca_variance')
        if variance_df is None:
            print("Error: No PCA variance found in context.")
            return

        plt.figure(figsize=(8, 6))
        sns.barplot(x='Component', y='Explained_Variance_Ratio', hue='Component', data=variance_df, palette='viridis', legend=False)
        plt.title('Explained Variance Ratio of Principal Components')
        plt.ylabel('Variance Ratio')
        plt.grid(axis='y')
        
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Variance visualization saved to: {save_path}")
