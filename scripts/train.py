import sys
from pathlib import Path

# Allow running the script without installing the package.
project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from mlops_pipeline.training.train import run_training

if __name__ == "__main__":
    run_training()
