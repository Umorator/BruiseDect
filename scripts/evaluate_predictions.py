import argparse
import json
from pathlib import Path
from src.eval.metrics import evaluate_all


def _load_config(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to evaluate.json")
    ap.add_argument("--strategy", nargs="*", help="Override strategies list e.g. strict smoothed")
    ap.add_argument("--model", nargs="*", help="Override models list e.g. YOLO FRCNN RetinaNet")
    ap.add_argument("--folds", nargs="*", type=int, help="Only these folds (e.g., 1 3 5)")
    args = ap.parse_args()

    cfg = _load_config(Path(args.config))
    if args.strategy:
        cfg["strategies"] = [s.capitalize() for s in args.strategy]
    if args.model:
        cfg["models"] = args.model
    if args.folds:
        cfg["folds"] = args.folds

    detailed_csv, summary_csv = evaluate_all(cfg)
    print("[eval] done.")
    print(f"[eval] detailed -> {detailed_csv}")
    print(f"[eval] summary  -> {summary_csv}")


if __name__ == "__main__":
    main()