import argparse, json
from pathlib import Path

# imports from src/
from src.augment.pipeline import (
    split_and_move_data,
    augment_entire_dataset,
    split_and_move_data_kfold,
)

def load_config(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))

def _as_list(x):
    if x is None: return None
    if isinstance(x, (list, tuple)): return list(x)
    return [x]

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)

    # Optional overrides
    ap.add_argument("--mode", choices=["split", "entire", "kfold"], default=None)
    ap.add_argument("--input_dir", default=None)
    ap.add_argument("--output_dir_name", default=None)
    ap.add_argument("--times", type=int, default=None)
    ap.add_argument("--resize", type=str, default=None)  # "true"/"false"
    ap.add_argument("--size_number", type=int, default=None)
    ap.add_argument("--ratios", type=float, nargs=2, default=None)
    ap.add_argument("--kfold", type=int, default=None)
    ap.add_argument("--mosaic_percentage", type=int, default=None)
    ap.add_argument("--class_id", type=int, default=None)
    ap.add_argument("--yaml_names", nargs="+", default=None)
    ap.add_argument("--extra_images_dir", nargs="+", default=None)

    # strategy templating
    ap.add_argument("--strategy", choices=["strict", "smoothed", "loose"], default=None)

    # NEW: master split spec + mosaic seed base
    ap.add_argument("--split_spec", default=None)
    ap.add_argument("--mosaic_seed_base", type=int, default=None)

    args = ap.parse_args(argv)
    cfg = load_config(Path(args.config))

    # Apply overrides if provided
    def put(key, val):
        if val is not None:
            cfg[key] = val

    put("mode", args.mode)
    put("input_dir", args.input_dir)
    put("output_dir_name", args.output_dir_name)
    put("times", args.times)
    put("size_number", args.size_number)
    put("kfold", args.kfold)
    put("mosaic_percentage", args.mosaic_percentage)
    put("class_id", args.class_id)
    put("split_spec", args.split_spec)
    put("mosaic_seed_base", args.mosaic_seed_base)

    if args.yaml_names is not None:
        cfg["yaml_class_names"] = args.yaml_names
    if args.resize is not None:
        cfg["resize"] = args.resize.lower() == "true"
    if args.extra_images_dir is not None:
        cfg["extra_images_dir"] = args.extra_images_dir

    # Resolve strategy-based templates
    strategy = args.strategy or cfg.get("strategy")
    in_tpl = cfg.get("input_dir_template")
    if in_tpl:
        if not strategy:
            raise ValueError("input_dir_template provided but no strategy was given (--strategy or 'strategy' in config).")
        cfg["input_dir"] = in_tpl.replace("{strategy}", strategy.capitalize())
    out_name = cfg.get("output_dir_name")
    if isinstance(out_name, str) and "{strategy}" in out_name:
        if not strategy:
            raise ValueError("output_dir_name uses {strategy} but no strategy was provided.")
        cfg["output_dir_name"] = out_name.replace("{strategy}", strategy.capitalize())
    if "extra_images_dir" in cfg:
        cfg["extra_images_dir"] = _as_list(cfg["extra_images_dir"])

    mode = cfg["mode"]

    if mode == "split":
        out = split_and_move_data(
            cfg["input_dir"],
            cfg["output_dir_name"],
            tuple(cfg["ratios"]),
            cfg["times"],
            cfg.get("size_number"),
            cfg.get("resize", False),
            cfg.get("class_id", 0),
            cfg.get("mosaic_percentage", 10),
            cfg.get("extra_images_dir"),
        )

    elif mode == "entire":
        out = augment_entire_dataset(
            cfg["input_dir"],
            cfg["output_dir_name"],
            cfg["times"],
            cfg.get("size_number"),
            cfg.get("resize", False),
            cfg.get("class_id", 0),
            cfg.get("mosaic_percentage", 10),
            cfg.get("extra_images_dir"),
        )

    elif mode == "kfold":
        out = split_and_move_data_kfold(
            cfg["input_dir"],
            cfg["output_dir_name"],
            cfg["kfold"],
            cfg["times"],
            cfg.get("size_number"),
            cfg.get("resize", False),
            cfg.get("class_id", 0),
            cfg.get("mosaic_percentage", 10),
            tuple(cfg.get("yaml_class_names", ["hematoma"])),
            cfg.get("extra_images_dir"),
            False,  # keep_staging
            cfg.get("split_spec"),                 # NEW
            int(cfg.get("mosaic_seed_base", 12345))# NEW
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    print(f"[augment] wrote outputs to: {out}")

if __name__ == "__main__":
    main()
