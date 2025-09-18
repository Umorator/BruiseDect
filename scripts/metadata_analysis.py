# scripts/metadata_analysis.py
import os
import argparse
import json
import pandas as pd
from datetime import datetime

from src.analysis.plot_helper import (
    load_and_prepare_data,
    plot_age_distribution_by_sex,
    plot_hematoma_age_distribution,
    plot_hematoma_location_by_gender,
    plot_photos_per_case_histogram,
    plot_top5_locations_stacked_by_age_with_gender_patterns,
)


DEFAULT_AGE_RANGES = [(0, 2), (2, 6), (6, 18), (18, 60)]


def parse_age_ranges(arg: str):
    """
    Parse CLI like: "0-2,2-6,6-18,18-60" -> [(0,2),(2,6),(6,18),(18,60)]
    If arg is empty/None, returns DEFAULT_AGE_RANGES.
    """
    if not arg:
        return DEFAULT_AGE_RANGES
    out = []
    for token in arg.split(","):
        token = token.strip()
        if not token:
            continue
        a, b = token.split("-")
        out.append((float(a), float(b)))
    return [(int(a), int(b)) for a, b in out]


def summarize_counts(filtered_metadata: pd.DataFrame, age_ranges):
    """
    Make a compact CSV summary:
      - total rows per range
      - counts per gender per range
      - counts per 'Age of bruise' per range (fresh/old etc.)
    """
    rows = []
    for a, b in age_ranges:
        sub = filtered_metadata[(filtered_metadata["Age"] >= a) & (filtered_metadata["Age"] <= b)]
        total = len(sub)
        # gender counts
        gcounts = sub["Gender"].str.capitalize().value_counts().to_dict()
        # bruise-age counts
        bcounts = sub["Age of bruise"].value_counts().to_dict()
        rows.append({
            "age_min": a,
            "age_max": b,
            "n_total": total,
            "n_female": int(gcounts.get("Female", 0)),
            "n_male": int(gcounts.get("Male", 0)),
            "hematoma_fresh": int(bcounts.get("frisch", 0)),
            "hematoma_old": int(bcounts.get("älter", 0)),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser("metadata_analysis")
    ap.add_argument("--metadata_csv", required=True, help="Path to metadata CSV (e.g., metadata_AIH_feb2025.csv)")
    ap.add_argument("--image_folder", required=True, help="Folder containing images (used for filtering/ID matching)")
    ap.add_argument("--json_file", required=True, help="JSON file with case/folder -> list of IDs to keep")
    ap.add_argument("--out_dir", required=True, help="Output directory where figures and CSVs will be written")
    ap.add_argument("--age_ranges", default="", help='Optional: "0-2,2-6,6-18,18-60" (default uses the four bands)')
    ap.add_argument("--tag", default="", help="Optional short tag to suffix the run folder (e.g., Strict)")
    args = ap.parse_args()

    age_ranges = parse_age_ranges(args.age_ranges)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"metadata_run_{ts}" + (f"_{args.tag}" if args.tag else "")
    out_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    # Save a tiny run spec for reproducibility
    spec = {
        "metadata_csv": args.metadata_csv,
        "image_folder": args.image_folder,
        "json_file": args.json_file,
        "out_dir": out_dir,
        "age_ranges": age_ranges,
        "tag": args.tag,
    }
    with open(os.path.join(out_dir, "specs.json"), "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)

    print("[metadata] loading & filtering…")
    filtered_metadata = load_and_prepare_data(
        metadata_file=args.metadata_csv,
        image_folder=args.image_folder,
        json_file=args.json_file,
    )

    # 1) Global histograms / helpers
    print("[metadata] photos per case histogram…")
    plot_photos_per_case_histogram(filtered_metadata, out_dir)

    print("[metadata] optional case photo lists…")
    # Uncomment if/when you want the text/json written
    # from plot_helper import save_case_photo_info
    # save_case_photo_info(filtered_metadata, out_dir)

    # 2) Per-age-range figures
    print("[metadata] per-range plots…")
    for (amin, amax) in age_ranges:
        plot_age_distribution_by_sex(filtered_metadata, out_dir, amin, amax)
        plot_hematoma_age_distribution(filtered_metadata, amin, amax, out_dir)
        plot_hematoma_location_by_gender(filtered_metadata, out_dir, amin, amax)

    # 3) Top-5 (actually top-8 in your helper) locations stacked across age groups
    print("[metadata] stacked top locations by age w/ sex patterns…")
    plot_top5_locations_stacked_by_age_with_gender_patterns(
        filtered_metadata,
        out_dir,
        age_ranges
    )

    # 4) Summary CSV
    print("[metadata] summary CSV…")
    df_summary = summarize_counts(filtered_metadata, age_ranges)
    df_summary.to_csv(os.path.join(out_dir, "summary_counts_by_age_range.csv"), index=False)

    print(f"[done] outputs in: {out_dir}")


if __name__ == "__main__":
    main()
