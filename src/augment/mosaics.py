import os, cv2, numpy as np, random
import albumentations as A
from .ops import _clamp_yolo_list

def _read_yolo_with_class(txt_path, default_class=0):
    bbs = []
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    c, x, y, w, h = parts
                    bbs.append([float(x), float(y), float(w), float(h), int(float(c))])
    return bbs

def _aug800(image, bboxes):
    aug = A.Compose(
        [A.RandomSizedBBoxSafeCrop(width=800, height=800, p=1.0)],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )
    bboxes = _clamp_yolo_list(bboxes)  # clamp before aug
    bbs  = [b[:4] for b in bboxes]
    labs = [b[4]  for b in bboxes]
    out  = aug(image=image, bboxes=bbs, class_labels=labs)
    return out["image"], [list(bb) + [lab] for bb, lab in zip(out["bboxes"], out["class_labels"])]

def _pixel_bboxes_from_yolo(yolos, W, H):
    px = []
    for x, y, w, h, c in yolos:
        x1 = int((x - w/2) * W); y1 = int((y - h/2) * H)
        x2 = int((x + w/2) * W); y2 = int((y + h/2) * H)
        px.append([x1, y1, x2, y2, c])
    return px

def _to_yolo_from_pixels(px_bbs, W, H):
    out = []
    for x1, y1, x2, y2, c in px_bbs:
        xc = (x1 + x2) / (2 * W)
        yc = (y1 + y2) / (2 * H)
        w  = (x2 - x1) / W
        h  = (y2 - y1) / H
        out.append([xc, yc, w, h, c])
    return out

def create_mosaic_from_directory(
    image_folder, bbox_folder, output_image_folder, output_bbox_folder,
    percentage=25, default_class_id=0, seed=None
):
    if seed is not None:
        random.seed(int(seed))
        np.random.seed(int(seed))

    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_bbox_folder,  exist_ok=True)

    image_paths = sorted([
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    n = max(0, int(len(image_paths) * (percentage / 100.0)))
    if n == 0:
        return

    for i in range(n):
        sel = random.sample(image_paths, 4)
        imgs, all_px = [], []
        for idx, ip in enumerate(sel):
            img = cv2.imread(ip)
            if img is None:
                continue
            lbl = os.path.join(bbox_folder, os.path.splitext(os.path.basename(ip))[0] + ".txt")
            bboxes = _read_yolo_with_class(lbl, default_class=default_class_id)
            bboxes = _clamp_yolo_list(bboxes)
            aug_img, aug_b = _aug800(img, bboxes)
            H, W = aug_img.shape[:2]
            px = _pixel_bboxes_from_yolo(aug_b, W, H)

            # shift into a 2x2 mosaic quadrant
            if idx == 1:
                px = [[x1+W, y1,   x2+W, y2,   c] for x1,y1,x2,y2,c in px]
            elif idx == 2:
                px = [[x1,   y1+H, x2,   y2+H, c] for x1,y1,x2,y2,c in px]
            elif idx == 3:
                px = [[x1+W, y1+H, x2+W, y2+H, c] for x1,y1,x2,y2,c in px]

            imgs.append(aug_img)
            all_px.extend(px)

        if len(imgs) != 4:
            continue

        top = np.hstack((imgs[0], imgs[1]))
        bot = np.hstack((imgs[2], imgs[3]))
        mosaic = np.vstack((top, bot))
        Hm, Wm = mosaic.shape[:2]

        out_img = os.path.join(output_image_folder, f"mosaic_{i}.jpg")
        cv2.imwrite(out_img, mosaic)

        yolos = _to_yolo_from_pixels(all_px, Wm, Hm)
        out_lbl = os.path.join(output_bbox_folder, f"mosaic_{i}.txt")
        with open(out_lbl, "w", encoding="utf-8") as f:
            for xc, yc, w, h, c in yolos:
                f.write(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
