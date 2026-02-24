import pandas as pd
from pathlib import Path

# ===== CONFIG =====
INPUT_FILE = "data/1.txt"     # <-- your big txt file
OUTPUT_FILE = "data/argo_clean.parquet"
CHUNK_SIZE = 500_000                 # safe for low RAM

IMPORTANT_COLS = [
    "juld",
    "latitude",
    "longitude",
    "pres_adjusted",
    "pres_adjusted_qc",
    "temp_adjusted",
    "temp_adjusted_qc",
    "psal_adjusted",
    "psal_adjusted_qc"
]

def process_chunk(chunk):
    # Keep required columns only
    chunk = chunk[IMPORTANT_COLS]

    # Apply IMOS QC (1 = good, 2 = probably good)
    chunk = chunk[
        (chunk["pres_adjusted_qc"].isin([1, 2])) &
        (chunk["temp_adjusted_qc"].isin([1, 2])) &
        (chunk["psal_adjusted_qc"].isin([1, 2]))
    ]

    # Convert time
    chunk["juld"] = pd.to_datetime(chunk["juld"], errors="coerce")

    # Rename columns (AI + UI friendly)
    chunk.rename(columns={
        "juld": "time",
        "pres_adjusted": "pressure",
        "temp_adjusted": "temperature",
        "psal_adjusted": "salinity"
    }, inplace=True)

    return chunk.dropna()

def main():
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(exist_ok=True)

    first = True
    rows_written = 0

    print("🚀 Starting ARGO preprocessing...")

    for chunk in pd.read_csv(
        INPUT_FILE,
        comment="#",
        sep=",",
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):
        clean_chunk = process_chunk(chunk)

        if clean_chunk.empty:
            continue

        clean_chunk.to_parquet(
            OUTPUT_FILE,
            engine="pyarrow",
            compression="snappy",
            append=not first
        )

        rows_written += len(clean_chunk)
        first = False

        print(f"✅ Processed {rows_written:,} rows")

    print("🎉 DONE!")
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()