import os
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import pandas as pd

from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# tvrefs (vendor the torchvision references under src/train/tvrefs)
from src.train.tvrefs import utils
from src.train.tvrefs.engine import train_one_epoch as tv_train_one_epoch, evaluate as tv_evaluate
from PIL import Image

# ----------------------------- Dataset utils -----------------------------

def read_yolo_annotations(txt_file_path: str, img_width: int, img_height: int):
    """
    Read YOLO txt (class x y w h normalized) -> pixel xyxy + labels + areas
    """
    import torch
    boxes, areas, labels = [], [], []
    if not os.path.exists(txt_file_path):
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros(0, dtype=torch.int64), torch.zeros(0, dtype=torch.float32)

    with open(txt_file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:])
            x_pix = x * img_width
            y_pix = y * img_height
            w_pix = w * img_width
            h_pix = h * img_height
            xmin = x_pix - w_pix / 2
            ymin = y_pix - h_pix / 2
            xmax = x_pix + w_pix / 2
            ymax = y_pix + h_pix / 2
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                areas.append((xmax - xmin) * (ymax - ymin))
                labels.append(1)  # single class foreground

    import torch
    boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64)
    areas = torch.tensor(areas, dtype=torch.float32) if areas else torch.zeros(0, dtype=torch.float32)
    return boxes, labels, areas


class IMPACT_AI(torch.utils.data.Dataset):
    """
    Expects:
      <root_dir>/
        images/
        labels/
    with YOLO txt labels (class x y w h, normalized).
    """
    def __init__(self, root_dir: str, resize_number: int = 640, cocoval: bool = False):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "images")
        self.lbl_dir = os.path.join(root_dir, "labels")
        # only file names; no module objects on self
        self.imgs = sorted(
            [f for f in os.listdir(self.img_dir)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        )
        self.resize_number = int(resize_number)
        self.cocoval = bool(cocoval)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # paths
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)
        lbl_path = os.path.join(self.lbl_dir, os.path.splitext(img_name)[0] + ".txt")

        # load image (use top-level PIL.Image import directly)
        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        # load annotations
        boxes, labels, areas = read_yolo_annotations(lbl_path, w, h)

        # resize keeping aspect ratio (longest side -> resize_number)
        max_side = max(w, h)
        ratio = self.resize_number / max(max_side, 1)
        new_w, new_h = int(w * ratio), int(h * ratio)
        image = image.resize((new_w, new_h), Image.LANCZOS)

        if boxes.numel() > 0:
            boxes[:, [0, 2]] *= ratio
            boxes[:, [1, 3]] *= ratio

        # to tensors
        image = to_tensor(image)
        num_objs = boxes.shape[0]
        iscrowd = torch.zeros((num_objs,), dtype=torch.uint8)

        # torchvision refs expect int id for COCO path, tensor elsewhere
        image_id = idx if self.cocoval else torch.tensor([idx])

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": areas,
            "iscrowd": iscrowd,
        }
        return image, target



# ----------------------------- Model utils -----------------------------

class FastRCNNPredictorWithOptionalDropout(nn.Module):
    def __init__(self, in_channels, num_classes, use_dropout=False, dropout_prob=0.3):
        super().__init__()
        self.use_dropout = bool(use_dropout)
        if self.use_dropout:
            self.dropout = nn.Dropout(p=float(dropout_prob))
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if self.use_dropout:
            x = self.dropout(x)
        return self.cls_score(x), self.bbox_pred(x)


def initialize_faster_rcnn(num_classes: int, pretrained=True, trainable_backbone_layers=0,
                           use_dropout=True, dropout_prob=0.3, weights_path=None):
    """
    Initialize Faster R-CNN model with optional pre-trained weights.
    
    Args:
        weights_path: Path to pre-trained model weights. If provided, pretrained is ignored.
    """
    # First create the model structure
    if weights_path and os.path.exists(weights_path):
        print(f"Loading pre-trained weights from: {weights_path}")
        # When loading custom weights, we need to create the model without pretrained weights
        # but with the correct number of classes
        model = fasterrcnn_resnet50_fpn(weights=None, trainable_backbone_layers=trainable_backbone_layers)
        
        # Replace the box predictor to match our num_classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        if use_dropout:
            model.roi_heads.box_predictor = FastRCNNPredictorWithOptionalDropout(
                in_channels=in_features,
                num_classes=num_classes,
                use_dropout=use_dropout,
                dropout_prob=dropout_prob,
            )
        else:
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Load the state dict, but handle mismatched keys
        state_dict = torch.load(weights_path, map_location='cpu')
        
        # Filter out the classifier weights that don't match in size
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if 'roi_heads.box_predictor' in key:
                # Skip the classifier weights since they have different sizes
                if 'cls_score' in key or 'bbox_pred' in key:
                    continue
            filtered_state_dict[key] = value
        
        # Load the filtered state dict (this will load everything except the classifier)
        model.load_state_dict(filtered_state_dict, strict=False)
        
        print("Loaded pre-trained weights (excluding classifier layers due to size mismatch)")
        
    else:
        # Default initialization path
        weights = "DEFAULT" if pretrained else None
        model = fasterrcnn_resnet50_fpn(weights=weights, trainable_backbone_layers=trainable_backbone_layers)
        
        # Replace the box predictor for our specific number of classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        if use_dropout:
            model.roi_heads.box_predictor = FastRCNNPredictorWithOptionalDropout(
                in_channels=in_features,
                num_classes=num_classes,
                use_dropout=use_dropout,
                dropout_prob=dropout_prob,
            )
        else:
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model


def save_experiment_specifications(run_dir: str, model_type: str, batch_size: int, lr: float, lf: float,
                                   dropout_prob: float, learning_schedule: str, dataset: str, epochs: int):
    specs = f"""Experiment Specifications:
Model Type: {model_type}
Batch Size: {batch_size}
Learning Rate: {lr}
Dropout rate: {dropout_prob}
Lf: {lf}
Learning Schedule: {learning_schedule}
Dataset: {dataset}
Epochs: {epochs}
"""
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(run_dir) / "experiment_specifications.txt", "w") as f:
        f.write(specs)


# ----------------------------- Validation loss (optional) -----------------------------

def _eval_forward(model, images, targets):
    """
    Forward that computes losses similar to torchvision's internal training path,
    but callable under torch.no_grad().
    """
    from collections import OrderedDict
    from torchvision.models.detection.roi_heads import fastrcnn_loss
    from torchvision.models.detection.rpn import concat_box_prediction_layers

    model.eval()

    original_image_sizes: List[Tuple[int, int]] = []
    for img in images:
        val = img.shape[-2:]
        original_image_sizes.append((val[0], val[1]))

    images, targets = model.transform(images, targets)
    features = model.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])

    # ---- RPN
    features_list = list(features.values())
    objectness, pred_bbox_deltas = model.rpn.head(features_list)
    anchors = model.rpn.anchor_generator(images, features_list)
    num_anchors_per_level = [o[0].numel() for o in objectness]
    objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)

    proposals = model.rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
    proposals = proposals.view(len(anchors), -1, 4)
    proposals, _scores = model.rpn.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

    labels, matched_gt_boxes = model.rpn.assign_targets_to_anchors(anchors, targets)
    regression_targets = model.rpn.box_coder.encode(matched_gt_boxes, anchors)
    loss_objectness, loss_rpn_box_reg = model.rpn.compute_loss(objectness, pred_bbox_deltas, labels, regression_targets)

    # ---- ROI heads
    image_shapes = images.image_sizes
    proposals, matched_idxs, labels_frcnn, regression_targets = model.roi_heads.select_training_samples(proposals, targets)
    box_features = model.roi_heads.box_roi_pool(features, proposals, image_shapes)
    box_features = model.roi_heads.box_head(box_features)
    class_logits, box_regression = model.roi_heads.box_predictor(box_features)
    loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels_frcnn, regression_targets)

    losses = {
        "loss_objectness": loss_objectness,
        "loss_rpn_box_reg": loss_rpn_box_reg,
        "loss_classifier": loss_classifier,
        "loss_box_reg": loss_box_reg,
    }

    # postprocess detections (not used here, but kept for completeness)
    boxes, scores, labels_det = model.roi_heads.postprocess_detections(
        class_logits, box_regression, proposals, image_shapes
    )
    results = []
    for i in range(len(boxes)):
        results.append({"boxes": boxes[i], "labels": labels_det[i], "scores": scores[i]})
    results = model.transform.postprocess(results, images.image_sizes, original_image_sizes)

    return losses, results


@torch.no_grad()
def evaluate_loss(model, data_loader, device):
    model.eval()
    total = 0.0
    n = 0
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict, _ = _eval_forward(model, images, targets)
        loss = sum(loss_dict.values())
        total += float(loss)
        n += 1
    return total / max(n, 1)


def _eval_map50(model, data_loader, device):
    """
    Debug version to see what's happening
    """
    model.eval()
    
    # Test one batch to see predictions
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            predictions = model(images)
            
            print(f"Number of predictions: {len(predictions)}")
            for i, pred in enumerate(predictions):
                print(f"Prediction {i}: {len(pred['boxes'])} boxes")
                if len(pred['boxes']) > 0:
                    print(f"  Scores: {pred['scores'][:3]}")  # First 3 scores
                    print(f"  Boxes shape: {pred['boxes'].shape}")
            break
    
    # Then run normal evaluation
    try:
        out = tv_evaluate(model, data_loader, device=device)
        try:
            stats = out.coco_eval["bbox"].stats
            return float(stats[1])  # AP50
        except Exception:
            try:
                return float(out)
            except Exception:
                return 0.0
    except Exception as e:
        print(f"Evaluation error: {e}")
        return 0.0


# ----------------------------- Trainer -----------------------------

def train_kfold(cfg: Dict):
    """
    Train Faster R-CNN on augmented K-fold splits.

    cfg keys (with defaults):
      - strategy: "strict" | "smoothed" | "loose"
      - aug_root: "P:/BruiseDet_Repo/data/augmented"
      - project_dir: "P:/BruiseDet_Repo/outputs/training/{strategy}/FRCNN"
      - out_dir: same as project_dir
      - resize_number: 640
      - batch: 8
      - workers: 16
      - threads: (optional) set torch.set_num_threads
      - epochs: 30
      - lr: 0.001
      - lrf: 0.01
      - use_dropout: true
      - dropout_prob: 0.3
      - pretrained: true
      - trainable_backbone_layers: 0
      - device: None -> auto
      - init_from_prev: false, whether to initialize from previous weights
      - prev_weights_template: "", template path for previous weights (e.g., "path/k{fold}/best_model.pth")
    """
    # device / threads
    device_override = cfg.get("device")
    device = torch.device(device_override) if device_override else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    threads = int(cfg.get("threads", 0) or 0)
    if threads > 0:
        try:
            torch.set_num_threads(threads)
        except Exception:
            pass

    # paths
    strategy = str(cfg.get("strategy", "strict")).lower()
    strat_cap = strategy.capitalize()

    aug_root = Path(cfg.get("aug_root", r"data/augmented"))
    folds_root = aug_root / strat_cap

    proj_tpl = cfg.get("project_dir", r"outputs/training/{strategy}/FRCNN")
    out_tpl = cfg.get("out_dir", proj_tpl)
    project_dir = Path(proj_tpl.format(strategy=strat_cap))
    out_dir = Path(out_tpl.format(strategy=strat_cap))
    project_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # hparams
    resize_number = int(cfg.get("resize_number", 640))
    batch = int(cfg.get("batch", 8))
    workers = int(cfg.get("workers", 16))
    epochs = int(cfg.get("epochs", 30))
    lr = float(cfg.get("lr", 0.001))
    lrf = float(cfg.get("lrf", 0.01))
    use_dropout = bool(cfg.get("use_dropout", True))
    dropout_prob = float(cfg.get("dropout_prob", 0.3))
    pretrained = bool(cfg.get("pretrained", True))
    trainable_backbone_layers = int(cfg.get("trainable_backbone_layers", 0))
    
    # fine-tuning parameters
    init_from_prev = bool(cfg.get("init_from_prev", False))
    prev_weights_template = cfg.get("prev_weights_template", "")

    # discover folds
    if not folds_root.exists():
        raise FileNotFoundError(f"No augmented data for strategy '{strat_cap}' at: {folds_root}")
    kfold_dirs = sorted([p for p in folds_root.glob("Kfold_*") if p.is_dir()], key=lambda p: p.name)
    if not kfold_dirs:
        raise FileNotFoundError(f"No Kfold_* splits found under: {folds_root}")

    print(f"[FRCNN] strategy={strat_cap}")
    print(f"[FRCNN] data root: {folds_root}")
    print(f"[FRCNN] outputs:   {project_dir}")
    if init_from_prev:
        print(f"[FRCNN] initializing from previous weights: {prev_weights_template}")

    results = {}

    for k_idx, kdir in enumerate(kfold_dirs, start=1):
        train_dir = kdir / "train"
        val_dir = kdir / "val"
        if not (train_dir.exists() and val_dir.exists()):
            print(f"[warn] missing train/val under {kdir}, skipping")
            continue

        fold_dir = project_dir / f"k{k_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Determine weights path for this fold if init_from_prev is True
        weights_path = None
        if init_from_prev and prev_weights_template:
            try:
                weights_path = prev_weights_template.format(strategy=strat_cap, fold=k_idx)
                if not os.path.exists(weights_path):
                    print(f"[warn] Pre-trained weights not found at: {weights_path}, using default initialization")
                    weights_path = None
                else:
                    print(f"[FRCNN|k{k_idx}] Loading pre-trained weights from: {weights_path}")
            except KeyError as e:
                print(f"[warn] Invalid placeholder in weights template: {e}, using default initialization")
                weights_path = None

        # datasets & loaders
        ds_train = IMPACT_AI(str(train_dir), resize_number=resize_number, cocoval=False)
        ds_val = IMPACT_AI(str(val_dir), resize_number=resize_number, cocoval=False)
        ds_val_coco = IMPACT_AI(str(val_dir), resize_number=resize_number, cocoval=True)

        dl_train = torch.utils.data.DataLoader(
            ds_train, batch_size=batch, shuffle=True, num_workers=workers,
            pin_memory=True, collate_fn=utils.collate_fn
        )
        dl_val = torch.utils.data.DataLoader(
            ds_val, batch_size=1, shuffle=False, num_workers=workers,
            pin_memory=True, collate_fn=utils.collate_fn
        )
        dl_val_coco = torch.utils.data.DataLoader(
            ds_val_coco, batch_size=1, shuffle=False, num_workers=workers,
            pin_memory=True, collate_fn=utils.collate_fn
        )

        # model
        model = initialize_faster_rcnn(
            num_classes=2,
            pretrained=pretrained,
            trainable_backbone_layers=trainable_backbone_layers,
            use_dropout=use_dropout,
            dropout_prob=dropout_prob,
            weights_path=weights_path,
        ).to(device)

        # optim & sched
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
        lf = lambda x: max(1 - x / max(epochs, 1), 0) * (1.0 - lrf) + lrf
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

        # files
        best_model_path = fold_dir / "best_model.pth"
        checkpoint_path = fold_dir / "checkpoint.pth"
        epoch_losses_path = fold_dir / "epoch_losses.csv"
        map_per_epoch_path = fold_dir / "mAP_per_epoch.csv"

        # resume
        start_epoch = 0
        best_map = 0.0
        if checkpoint_path.exists():
            print(f"[FRCNN|k{k_idx}] resuming from: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_epoch = int(ckpt.get("epoch", -1)) + 1
            best_map = float(ckpt.get("best_map", 0.0))

        # history
        if epoch_losses_path.exists():
            epoch_losses = pd.read_csv(epoch_losses_path).values.tolist()
        else:
            epoch_losses = []
        if map_per_epoch_path.exists():
            map_hist = pd.read_csv(map_per_epoch_path).values.tolist()
            if map_hist:
                best_map = max(float(v) for _, v in map_hist)
        else:
            map_hist = []

        # save run specs
        save_experiment_specifications(
            str(fold_dir),
            model_type="fasterrcnn_resnet50_fpn",
            batch_size=batch,
            lr=lr,
            lf=lrf,
            dropout_prob=dropout_prob,
            learning_schedule="LambdaLR",
            dataset=str(train_dir),
            epochs=epochs,
        )

        # train
        for epoch in range(start_epoch, epochs):
            torch.cuda.empty_cache()

            # 🔸 pass run_dir to your engine's train_one_epoch
            tv_train_one_epoch(
                model, optimizer, dl_train, device, epoch,
                run_dir=str(fold_dir),
                print_freq=10
            )

            val_loss = evaluate_loss(model, dl_val, device=device)
            epoch_losses.append((epoch, float(val_loss)))

            map50 = _eval_map50(model, dl_val_coco, device=device)
            map_hist.append((epoch, float(map50)))

            if map50 > best_map:
                best_map = map50
                torch.save(model.state_dict(), best_model_path)
                print(f"[k{k_idx}] epoch {epoch}: new best mAP@0.5 {best_map:.4f} -> {best_model_path}")

            # checkpoint
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_map": best_map,
            }, checkpoint_path)

            lr_scheduler.step()
            print(f"[k{k_idx}] epoch {epoch+1}/{epochs} | lr={lr_scheduler.get_last_lr()[0]:.6f} | "
                  f"val_loss={float(val_loss):.4f} | mAP@0.5={map50:.4f}")

            # persist logs
            pd.DataFrame(epoch_losses, columns=["epoch", "Validation Loss"]).to_csv(epoch_losses_path, index=False)
            pd.DataFrame(map_hist, columns=["epoch", "mAP0.5"]).to_csv(map_per_epoch_path, index=False)

        results[f"k{k_idx}"] = {"best_map_50": best_map}

    # write summary
    (project_dir / "summary.json").write_text(pd.io.json.dumps(results, indent=2))
    print(f"[FRCNN] done. summary -> {project_dir / 'summary.json'}")
    return results
