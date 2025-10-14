import sys
from pathlib import Path
import argparse
# Allow running the script without installing the package.
project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from mlops_pipeline.inference.predict import predict_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executa inferencia em arquivo Excel")
    parser.add_argument("input_file", type=str)
    parser.add_argument("--output_file", type=str, default="predicoes.xlsx")
    args = parser.parse_args()
    out = predict_file(args.input_file, args.output_file)
    print(f"Resultados salvos em {out}")