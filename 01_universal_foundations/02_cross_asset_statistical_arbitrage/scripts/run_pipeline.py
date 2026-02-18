import sys
from pathlib import Path
import subprocess

scripts = ['run_features.py', 'run_models.py', 'run_backtest.py']

print("="*60)
print("PIPELINE")
print("="*60)

for script in scripts:
    print(f"\n{script.replace('.py', '').replace('run_', '')}:")
    result = subprocess.run([sys.executable, f'scripts/{script}'])
    if result.returncode != 0:
        sys.exit(1)

print("\n" + "="*60)
print("COMPLETE")
print("="*60)
