"""依序執行全部分析：python src/run_all.py"""
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for script in ["01_cleaning.py", "02_eda.py", "03_simpson.py", "04_clustering.py", "05_modeling.py"]:
    print(f"\n{'=' * 20} {script} {'=' * 20}")
    runpy.run_path(str(HERE / script), run_name="__main__")
