from pathlib import Path
import cv2

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_image_cv2(path: Path):
    img = cv2.imread(str(path))
    return img

def save_image_cv2(path: Path, image):
    cv2.imwrite(str(path), image)

def read_yolo_txt(path: Path):
    """Return list of [xc, yc, w, h] (normalized floats)."""
    boxes = []
    if not path.exists():
        return boxes
    with path.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                # class xc yc w h
                try:
                    _, x, y, w, h = map(float, parts)
                    boxes.append([x, y, w, h])
                except ValueError:
                    continue
    return boxes

def write_yolo_txt(path: Path, boxes, class_id: int = 0):
    with path.open("w") as f:
        for (xc, yc, w, h) in boxes:
            f.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
