import json
import re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style for professional appearance
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12

# ----------------------------- IO helpers -----------------------------

def load_config(p: Path) -> dict:
    """Load JSON configuration file."""
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dir(p: Path):
    """Ensure directory exists."""
    p.mkdir(parents=True, exist_ok=True)

# ----------------------------- Data loading -----------------------------

def extract_tensor_value(tensor_string):
    """Extract numeric value from strings like 'tensor(0.2813, device='cuda:0')'"""
    if isinstance(tensor_string, str):
        match = re.search(r'tensor\(([\d\.]+)', str(tensor_string))
        if match:
            return float(match.group(1))
    return np.nan

def load_model_data(base_path, model, fold, strategy):
    """Load training and validation loss data for a specific model, fold, and strategy."""
    try:
        fold_dir = f"Kfold_{fold}" if model in ['FRCNN', 'RetinaNet'] else f"k{fold}"
        strategy_path = Path(base_path) / strategy / model / fold_dir
        
        if model in ['FRCNN', 'RetinaNet']:
            # Load training data
            train_path = strategy_path / 'average_training_losses.csv'
            if not train_path.exists():
                return np.array([]), np.array([])
                
            train_df = pd.read_csv(train_path)
            
            # Load validation data
            val_path = strategy_path / 'epoch_losses.csv'
            if not val_path.exists():
                return np.array([]), np.array([])
                
            val_df = pd.read_csv(val_path)
            
            # Convert training data to numeric
            train_loss = pd.to_numeric(train_df['total_loss'], errors='coerce').dropna().values
            
            # Handle validation data
            val_loss = []
            
            if model == 'FRCNN':
                # For FRCNN: Extract numeric values from tensor strings
                for col in val_df.columns:
                    if col.lower() in ['validation loss', 'val_loss', 'validation_loss', 'val loss']:
                        val_loss = val_df[col].apply(extract_tensor_value).dropna().values
                        break
                
                if len(val_loss) == 0:
                    # If specific validation column not found, try any column that might contain tensor values
                    for col in val_df.columns:
                        if col.lower() != 'epoch':
                            val_loss = val_df[col].apply(extract_tensor_value).dropna().values
                            if len(val_loss) > 0:
                                break
            
            else:  # RetinaNet
                # For RetinaNet: Use regular numeric conversion
                for col in val_df.columns:
                    if col.lower() in ['validation loss', 'val_loss', 'validation_loss', 'val loss']:
                        val_loss = pd.to_numeric(val_df[col], errors='coerce').dropna().values
                        break
                
                if len(val_loss) == 0:
                    # If specific validation column not found, try any numeric column
                    for col in val_df.columns:
                        if col.lower() != 'epoch' and pd.api.types.is_numeric_dtype(val_df[col]):
                            val_loss = val_df[col].dropna().values
                            break
            
            # For RetinaNet, skip epoch 0 if it's an outlier
            if model == 'RetinaNet':
                if len(train_loss) > 0 and train_loss[0] > 100:
                    train_loss = train_loss[1:]
                
                if len(val_loss) > 0 and val_loss[0] > 100:
                    val_loss = val_loss[1:]
            
            return train_loss, val_loss
            
        else:  # YOLO
            results_path = strategy_path / 'results.csv'
            if not results_path.exists():
                return np.array([]), np.array([])
                
            results_df = pd.read_csv(results_path)
            
            # Convert all relevant columns to numeric
            box_loss_train = pd.to_numeric(results_df['train/box_loss'], errors='coerce').dropna().values
            cls_loss_train = pd.to_numeric(results_df['train/cls_loss'], errors='coerce').dropna().values
            dfl_loss_train = pd.to_numeric(results_df['train/dfl_loss'], errors='coerce').dropna().values
            
            box_loss_val = pd.to_numeric(results_df['val/box_loss'], errors='coerce').dropna().values
            cls_loss_val = pd.to_numeric(results_df['val/cls_loss'], errors='coerce').dropna().values
            dfl_loss_val = pd.to_numeric(results_df['val/dfl_loss'], errors='coerce').dropna().values
            
            # Calculate total losses
            min_length_train = min(len(box_loss_train), len(cls_loss_train), len(dfl_loss_train))
            train_loss = (box_loss_train[:min_length_train] + 
                         cls_loss_train[:min_length_train] + 
                         dfl_loss_train[:min_length_train])
            
            min_length_val = min(len(box_loss_val), len(cls_loss_val), len(dfl_loss_val))
            val_loss = (box_loss_val[:min_length_val] + 
                       cls_loss_val[:min_length_val] + 
                       dfl_loss_val[:min_length_val])
            
            return train_loss, val_loss
            
    except Exception as e:
        print(f"[error] Failed to load {model}/{strategy}/fold{fold}: {e}")
        return np.array([]), np.array([])

def process_training_data(base_path, strategies, models, folds):
    """Process training data for all strategies, models, and folds."""
    all_curves_data = {}
    
    for strategy in strategies:
        print(f"\n[info] Processing {strategy}...")
        strategy_curves_data = {}
        
        for model in models:
            print(f"  [info] Processing {model}...")
            
            all_train_losses = []
            all_val_losses = []
            successful_folds = 0
            
            for fold in folds:
                train_loss, val_loss = load_model_data(base_path, model, fold, strategy)
                
                if len(train_loss) > 0:
                    all_train_losses.append(train_loss)
                if len(val_loss) > 0:
                    all_val_losses.append(val_loss)
                
                if len(train_loss) > 0 or len(val_loss) > 0:
                    successful_folds += 1
                    print(f"    [info] Fold {fold}: Train epochs: {len(train_loss)}, Val epochs: {len(val_loss)}")
                else:
                    print(f"    [warn] Fold {fold}: No data found")
            
            if all_train_losses or all_val_losses:
                strategy_curves_data[model] = (all_train_losses, all_val_losses)
                print(f"  [info] {model}: {successful_folds}/{len(folds)} folds successful")
        
        all_curves_data[strategy] = strategy_curves_data
    
    return all_curves_data

# ----------------------------- Plotting -----------------------------

def plot_loss_curves_with_std(curves_data, out_path: Path, title: str = "Training and Validation Losses"):
    """Plot loss curves with standard deviation."""
    ensure_dir(out_path.parent)
    
    num_strategies = len(curves_data)
    fig, axes = plt.subplots(1, num_strategies, figsize=(6 * num_strategies, 6))
    if num_strategies == 1:
        axes = [axes]
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # Colors for training and validation
    train_color = '#1f77b4'  # Blue
    val_color = '#d62728'    # Red
    
    for idx, (strategy, model_data) in enumerate(curves_data.items()):
        ax = axes[idx]
        
        for model_label, (train_losses, val_losses) in model_data.items():
            # Plot training loss
            if train_losses:
                min_length = min(len(loss) for loss in train_losses)
                train_array = np.array([loss[:min_length] for loss in train_losses])
                
                train_mean = np.mean(train_array, axis=0)
                train_std = np.std(train_array, axis=0)
                
                epochs = np.arange(1, min_length + 1)
                
                ax.plot(epochs, train_mean, label=f'{model_label} Train', color=train_color, linewidth=2.5)
                ax.fill_between(epochs, train_mean - train_std, train_mean + train_std, 
                              alpha=0.3, color=train_color, label=f'{model_label} Train ±1 SD')
            
            # Plot validation loss
            if val_losses:
                min_length_val = min(len(loss) for loss in val_losses)
                val_array = np.array([loss[:min_length_val] for loss in val_losses])
                
                val_mean = np.mean(val_array, axis=0)
                val_std = np.std(val_array, axis=0)
                
                epochs_val = np.arange(1, min_length_val + 1)
                
                ax.plot(epochs_val, val_mean, label=f'{model_label} Val', color=val_color, 
                       linewidth=2.5, linestyle='--')
                ax.fill_between(epochs_val, val_mean - val_std, val_mean + val_std, 
                              alpha=0.3, color=val_color, label=f'{model_label} Val ±1 SD')
        
        ax.set_title(f'{strategy}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Loss', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 31)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()