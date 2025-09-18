import argparse
import json
from pathlib import Path

from src.predict.fasterrcnn import predict_kfold


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to predict_fasterrcnn.json")
    ap.add_argument("--strategy", help="strict|smoothed|loose (overrides config)")
    ap.add_argument("--folds", nargs="*", type=int, help="Which folds to run (e.g., 1 3 5). If omitted, auto-detect.")
    args = ap.parse_args()

    cfg = _load_config(Path(args.config))
    if args.strategy:
        cfg["strategy"] = args.strategy
    if args.folds:
        cfg["folds"] = args.folds

    results = predict_kfold(cfg)
    print("[frcnn|predict] done.")
    for k, n in results.items():
        print(f"  {k}: wrote {n} files")


if __name__ == "__main__":
    main()
