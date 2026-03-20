import pandas as pd
from pyulog import ULog
import os

class UlgParser:
    """
    Parser for .ulg files using pyulog.
    Extracts specified topics, synchronizes them by resampling, and merges into a single DataFrame.
    """
    def __init__(self, ulg_file_path: str):
        self.ulg_file_path = ulg_file_path
        if not os.path.exists(ulg_file_path):
            raise FileNotFoundError(f"ULog file not found: {ulg_file_path}")

    def parse(self, topics=['sensor_combined', 'actuator_outputs'], resample_rate='100ms') -> pd.DataFrame:
        """
        Parses the ULog file and returns a merged DataFrame.
        """
        ulog = ULog(self.ulg_file_path)
        dfs = []

        for topic in topics:
            try:
                # Get the first instance of the topic
                data = ulog.get_dataset(topic)
                df = pd.DataFrame(data.data)
                
                # Convert timestamp from microseconds to Timedelta
                df['timestamp'] = pd.to_timedelta(df['timestamp'], unit='us')
                df.set_index('timestamp', inplace=True)
                
                # Prefix columns to avoid collisions
                df.columns = [f"{topic}_{col}" for col in df.columns]
                
                # Resample to common grid
                # Use origin='start' to align across topics starting at different times
                df_resampled = df.resample(resample_rate, origin='start').mean()
                dfs.append(df_resampled)
            except (ValueError, KeyError, IndexError):
                print(f"Warning: Topic {topic} not found in log file.")

        if not dfs:
            raise ValueError("No specified topics found in the ULog file.")

        # Merge all dataframes on timestamp index
        # Use outer join to keep all time bins, then ffill
        merged_df = pd.concat(dfs, axis=1).sort_index().ffill().dropna()
        
        return merged_df

    def to_csv(self, output_path: str, topics=['sensor_combined', 'actuator_outputs'], resample_rate='100ms') -> str:
        """
        Parses the ULog file and saves the result to a CSV file.
        Returns the path to the saved CSV.
        """
        df = self.parse(topics, resample_rate)
        df.to_csv(output_path)
        return output_path
