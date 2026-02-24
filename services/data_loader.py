import pandas as pd
from pathlib import Path

def load_data():
    """Load preprocessed ARGO data from multiple files"""
    all_data = []
    
    # Load parquet files if available
    for parquet_file in Path("data").glob("*.parquet"):
        df = pd.read_parquet(parquet_file)
        all_data.append(df)
    
    # Load raw txt files if no parquet found
    if not all_data:
        for txt_file in Path("data").glob("*.txt"):
            try:
                df = pd.read_csv(txt_file, comment="#", sep=",", low_memory=False, nrows=10000)
                cols = ["latitude", "longitude", "pres_adjusted", "temp_adjusted", "psal_adjusted"]
                df = df[[c for c in cols if c in df.columns]]
                df.rename(columns={
                    "pres_adjusted": "pressure",
                    "temp_adjusted": "temperature",
                    "psal_adjusted": "salinity"
                }, inplace=True)
                all_data.append(df.dropna())
            except Exception as e:
                print(f"Error loading {txt_file}: {e}")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"✅ Loaded {len(combined_df)} records from {len(all_data)} files")
        return combined_df
    
    return pd.DataFrame(columns=["latitude", "longitude", "pressure", "temperature", "salinity"])
