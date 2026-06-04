# src/check_paths.py
from pathlib import Path
import os
p = Path.cwd()
raw = Path("data/raw/INDIA_VIX_minute.csv")
print("PWD:", p)
print("Python sees file exists?:", raw.exists())
print("Absolute path to file:", raw.resolve())
print("List files in data/raw:")
for f in raw.parent.iterdir():
    print(" -", f.name, f.stat().st_size)
