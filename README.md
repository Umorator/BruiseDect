# Bruise Detection Pipeline

A comprehensive pipeline for training, evaluating, and fine-tuning object detection models for bruise detection. This repository supports multiple annotation strategies, model architectures, and includes extensive evaluation capabilities.

## Overview

This pipeline processes bruise detection data through several stages:

- **Annotation Standardization**: Normalizes raw annotations into three strategies
- **Data Augmentation**: Creates consistent K-fold splits with augmentations
- **Model Training**: Trains multiple detection architectures
- **Prediction & Evaluation**: Generates predictions and comprehensive metrics
- **Fine-tuning**: Extends training with additional curated data
- **Metadata Analysis**: Examines demographic and case characteristics

## Quick Start

### Set up environment:
```bash
pip install -r requirements.txt
```

### Run the complete pipeline:
```bash
# Standardize annotations
python -m scripts.standardize_annotations --config configs/standardize.json

# Augment data for all strategies
python -m scripts.augment_dataset --config configs/augment.json --strategy strict
python -m scripts.augment_dataset --config configs/augment.json --strategy smoothed
python -m scripts.augment_dataset --config configs/augment.json --strategy loose

# Train baseline models
python -m scripts.train_yolov9 --config configs/train_yolov9.json --strategy strict
python -m scripts.train_fasterrcnn --config configs/train_fasterrcnn.json --strategy strict
python -m scripts.train_retinanet --config configs/train_retinanet.json --strategy strict

# Generate predictions and evaluate
python -m scripts.predict_yolov9 --config configs/predict_yolov9.json --strategy strict
python -m scripts.evaluate_predictions --config configs/evaluate.json
python -m scripts.plot_pr_curves --eval_root "outputs/eval" --out "outputs/eval/plots"
```

## Pipeline Stages

### 1. Annotation Standardization
Normalizes raw annotations into three consistent strategies:
- **Strict**: Conservative annotation boundaries
- **Smoothed**: Moderately adjusted boundaries
- **Loose**: Expanded annotation boundaries

**Usage:**
```bash
python -m scripts.standardize_annotations --config configs/standardize.json
```

### 2. Data Augmentation & K-fold Splitting
Creates consistent dataset splits with augmentations:
- 5-fold cross-validation (consistent across strategies)
- On-the-fly augmentations and mosaic generation
- YOLO-style directory structure

**Usage:**
```bash
python -m scripts.augment_dataset --config configs/augment.json --strategy [strict|smoothed|loose]
```

### 3. Model Training
Train multiple object detection architectures:
- **YOLOv9**: Latest YOLO architecture with high performance
- **Faster R-CNN**: Two-stage detector with region proposals
- **RetinaNet**: Single-shot detector with focal loss

**Usage:**
```bash
python -m scripts.train_[model] --config configs/train_[model].json --strategy [strategy]
```

### 4. Prediction & Evaluation
Generate predictions and comprehensive evaluation metrics:
- Precision, Recall, F1, IoU
- Average Precision @0.5 IoU
- Per-fold metrics and aggregated summaries
- Precision-Recall curves

**Usage:**
```bash
python -m scripts.predict_[model] --config configs/predict_[model].json --strategy [strategy]
python -m scripts.evaluate_predictions --config configs/evaluate.json
python -m scripts.plot_PR_and_losses --config configs/evaluate.json --strategy [strategy]

```

### 5. Fine-tuning
Extend training with curated external data:
- Add Roboflow dataset to training splits only
- Unfreeze backbone layers and adjust hyperparameters
- Maintain original validation sets for fair comparison

For augmenting a folder with new data (Roboflow)
**Usage:**
```bash
python -m scripts.augment_finetune --config configs/augment_finetune.json --strategy [strategy]
```
For training and fine-tuning
**Usage:**
```bash
python -m scripts.train_fasterrcnn --config configs/finetune_frcnn.json --strategy [strategy]
```

### 6. Metadata Analysis
Analyze demographic and case characteristics:
- Age distribution by sex
- Bruise location analysis
- Case photo statistics
- Age band analysis (0-2, 2-6, 6-18, 18-60 years)

**Usage:**
```bash
python -m scripts.metadata_analysis --meta [metadata_csv] --out [output_dir]
```

## Directory Structure

```
BruiseDet_Repo/
├── configs/                 # Configuration files
├── scripts/                 # Main pipeline scripts
├── src/                     # Source code utilities
├── data/
│   ├── raw_data/           # Original datasets
│   ├── augmented/          # Augmented datasets
│   └── augmented_ft/       # Fine-tuning datasets
└── outputs/
    ├── training/           # Baseline model weights
    ├── training_ft/        # Fine-tuned weights
    ├── predictions/        # Baseline predictions
    ├── predictions_ft/     # Fine-tuning predictions
    └── eval/               # Evaluation results and plots
```

> **Note**: The `outputs/` directory contains generated files and is excluded from version control via `.gitignore`.
