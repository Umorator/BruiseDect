# scripts/augment_finetune.py
import argparse, json
from pathlib import Path
from src.augment.pipeline_ft import add_extra_to_kfold

def _load_config(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to configs/augment_finetune.json")
    ap.add_argument("--strategy", choices=["strict","smoothed","loose"], default=None)
    args = ap.parse_args()

    cfg = _load_config(Path(args.config))
    strategy = (args.strategy or cfg["strategy"]).capitalize()

    base_aug_root   = cfg["base_aug_root_template"].format(strategy=strategy)
    out_aug_ft_root = cfg["out_aug_ft_template"].format(strategy=strategy)
    extra_dir       = cfg["extra_train_dir"]

    add_extra_to_kfold(
        base_aug_root=base_aug_root,
        extra_train_dir=extra_dir,
        out_aug_ft_root=out_aug_ft_root,
        times_extra=int(cfg.get("times_extra", 1)),
        size_number=int(cfg.get("size_number", 640)),
        resize=bool(cfg.get("resize", False)),
        class_id=int(cfg.get("class_id", 0)),
        mosaic_pct=int(cfg.get("mosaic_percentage", 10)),
        yaml_names=tuple(cfg.get("yaml_class_names", ["hematoma"])),
    )

if __name__ == "__main__":
    main()
