from pathlib import Path
from src.load_data import load_raw_data

df = load_raw_data(Path("Holidays (1).xlsx"))
print(df.head())
print(df.shape)
print(df.columns)