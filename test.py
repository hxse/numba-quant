import pandas_ta as ta

from utils.data_loading import load_ohlcv_from_csv, convert_ohlcv_numpy

data_size = 1000 * 20  # Your specified data size

csv_file_path = "csv/BTC_USDT/future live/4h/2022-01-01 00_00_00.csv"

# Load your OHLCV data
df = load_ohlcv_from_csv(csv_file_path, data_size)

# Ensure you have 'high', 'low', and 'close' columns
# If your DataFrame columns are named differently (e.g., 'High', 'Low', 'Close'), adjust accordingly.

# Calculate PSAR using pandas_ta
# ta.psar returns a DataFrame with 'PSARl', 'PSARs', 'PSARaf', 'PSARr' columns.
# We'll select a specific column (e.g., 'PSARl' for Long PSAR) to print its values.
# The default af0, af, and max_af values (0.02, 0.02, 0.2) are used if not specified.
# You can specify them if needed: e.g., ta.psar(high=df["high"], low=df["low"], close=df["close"], af0=0.02, af=0.02, max_af=0.2)
pandas_psar_output = ta.psar(
    high=df["high"], low=df["low"], close=df["close"], talib=False
)

# PSARl and PSARs column names include the af0 and max_af values by default
# e.g., 'PSARl_0.02_0.2' and 'PSARs_0.02_0.2'
af0_default = 0.02
max_af_default = 0.2
psarl_col_name = f"PSARl_{af0_default}_{max_af_default}"
psars_col_name = f"PSARs_{af0_default}_{max_af_default}"
psaraf_col_name = f"PSARaf_{af0_default}_{max_af_default}"
psarr_col_name = f"PSARr_{af0_default}_{max_af_default}"


# You can print the entire DataFrame or specific columns.
# For consistency with your previous `pandas_rsi.values` print, let's print the values of PSARl.
print(f"Pandas TA PSAR Long Values ({psarl_col_name}):")
print(pandas_psar_output[psarl_col_name].values)

print(f"\nPandas TA PSAR Short Values ({psars_col_name}):")
print(pandas_psar_output[psars_col_name].values)

print(f"\nPandas TA PSAR AF Values ({psaraf_col_name}):")
print(pandas_psar_output[psaraf_col_name].values)

print(f"\nPandas TA PSAR Reversal Values ({psarr_col_name}):")
print(pandas_psar_output[psarr_col_name].values)

# You can also print the first few rows of the entire DataFrame for a quick overview
print("\nFull Pandas TA PSAR Output (first 5 rows):")
print(pandas_psar_output.head())
