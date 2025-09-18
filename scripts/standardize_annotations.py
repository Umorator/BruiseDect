import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from src.annotations.merge import (
    find_common_images,
    load_annotations_for_images,
    list_all_images,
    draw_boxes_overlay,
)
from src.annotations.io import (
    load_image_cv2,
    save_image_cv2,
    write_yolo_txt,
    ensure_dir,
)
from src.annotations.strategies import get_strategy

def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def override_if_set(cfg: dict, key: str, value):
    if value is not None:
        cfg[key] = value

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to standardize.json")
    parser.add_argument("--strategy", choices=["strict", "smoothed", "loose"], required=True)

    # Optional CLI overrides (so you can change outputs without editing the file)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--class_id", type=int, default=None)
    parser.add_argument("--write_drawn", type=str, default=None, help="true/false")
    parser.add_argument("--copy_uncommon_images", type=str, default=None, help="true/false")

    args = parser.parse_args()

    cfg = load_config(Path(args.config))

    # Apply optional overrides
    override_if_set(cfg, "output_root", args.output_root)
    if "options" not in cfg:
        cfg["options"] = {}
    if args.class_id is not None:
        cfg["options"]["class_id"] = args.class_id
    if args.write_drawn is not None:
        cfg["options"]["write_drawn"] = args.write_drawn.lower() == "true"
    if args.copy_uncommon_images is not None:
        cfg["options"]["copy_uncommon_images"] = args.copy_uncommon_images.lower() == "true"

    sources = [Path(p) for p in cfg["sources"]]
    images_roots = [Path(p) for p in cfg["images_roots"]]
    output_root = Path(cfg["output_root"])
    options = cfg.get("options", {})
    valid_exts = set(options.get("valid_exts", [".jpg", ".jpeg", ".png"]))

    # Per-strategy params -> SimpleNamespace, so strategies that expect dot access still work
    params_dict = cfg["params"][args.strategy]
    params = SimpleNamespace(**params_dict)

    # Prepare output dir per strategy
    out_dir = output_root / args.strategy.capitalize()
    ensure_dir(out_dir)

    # 1) common + uncommon images
    common = find_common_images(sources)
    all_images = list_all_images(images_roots, valid_exts=valid_exts)
    uncommon = sorted(all_images - common)

    # 2) load annotations for common images
    annos = load_annotations_for_images(sources, common)

    # 3) choose strategy
    strategy_fn = get_strategy(args.strategy)

    # 4) process common images
    manifest = []
    for img_name in sorted(common):
        # find image file in the first images_root that has it
        img_path = None
        for root in images_roots:
            p = root / img_name
            if p.exists():
                img_path = p
                break
        if img_path is None:
            continue

        boxes_list = annos.get(img_name, [[], [], []])
        std_boxes = strategy_fn(boxes_list, params)

        image = load_image_cv2(img_path)
        if image is None:
            continue

        # write image copy
        save_image_cv2(out_dir / img_name, image)
        # write YOLO txt
        txt_out = out_dir / (Path(img_name).stem + ".txt")
        write_yolo_txt(txt_out, std_boxes, class_id=int(options.get("class_id", 0)))
        # overlay
        if bool(options.get("write_drawn", True)):
            overlay = draw_boxes_overlay(image, boxes_list, std_boxes)
            #save_image_cv2(out_dir / (Path(img_name).stem + "_drawn.jpg"), overlay) #save the drawn image

        manifest.append({"image": img_name, "boxes": std_boxes})

    # 5) copy uncommon images unchanged (optional)
    if bool(options.get("copy_uncommon_images", True)):
        for img_name in uncommon:
            for root in images_roots:
                p = root / img_name
                if p.exists():
                    image = load_image_cv2(p)
                    if image is not None:
                        save_image_cv2(out_dir / img_name, image)
                    break

    # 6) save a small manifest (optional)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"[standardize] strategy={args.strategy} wrote {len(manifest)} items to: {out_dir}")

if __name__ == "__main__":
    main()

