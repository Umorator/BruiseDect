import numpy as np

EPS = 1e-12

def xywh_to_xyxy(box):
    xc, yc, w, h = box
    return [xc - w/2, yc - h/2, xc + w/2, yc + h/2]

def iou_xywh(b1, b2):
    x1, y1, x2, y2 = xywh_to_xyxy(b1)
    X1, Y1, X2, Y2 = xywh_to_xyxy(b2)
    xi1, yi1 = max(x1, X1), max(y1, Y1)
    xi2, yi2 = min(x2, X2), min(y2, Y2)
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    a1 = max(EPS, (x2 - x1) * (y2 - y1))
    a2 = max(EPS, (X2 - X1) * (Y2 - Y1))
    denom = a1 + a2 - inter
    iou = inter / max(EPS, denom)
    return iou, inter, a1, a2, (xi1, yi1, xi2, yi2)

def containment(b1, b2, thr=0.8):
    x1, y1, x2, y2 = xywh_to_xyxy(b1)
    w, h = x2 - x1, y2 - y1
    shrink = (1 - thr) / 2
    sx1, sy1 = x1 + shrink * w, y1 + shrink * h
    sx2, sy2 = x2 - shrink * w, y2 - shrink * h
    X1, Y1, X2, Y2 = xywh_to_xyxy(b2)
    return (sx1 >= X1) and (sy1 >= Y1) and (sx2 <= X2) and (sy2 <= Y2)

def significant_overlap(b1, b2, min_ratio=0.2):
    iou, inter, a1, a2, _ = iou_xywh(b1, b2)
    smaller = min(a1, a2)
    return inter >= min_ratio * smaller

def nms_xywh(boxes, iou_thr=0.2, small_ratio_thr=0.20):
    if not boxes:
        return []
    arr = np.array([xywh_to_xyxy(b) for b in boxes], dtype=float)
    x1, y1, x2, y2 = arr[:,0], arr[:,1], arr[:,2], arr[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(areas)[::-1]
    keep_idx = []
    while order.size > 0:
        i = order[0]
        keep_idx.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + EPS)
        smaller = np.minimum(areas[i], areas[order[1:]])
        ratio = inter / (smaller + EPS)
        mask = (iou <= iou_thr) & (ratio <= small_ratio_thr)
        order = order[1:][mask]
    out = []
    for idx in keep_idx:
        out.append([(x1[idx]+x2[idx])/2, (y1[idx]+y2[idx])/2, x2[idx]-x1[idx], y2[idx]-y1[idx]])
    return out
