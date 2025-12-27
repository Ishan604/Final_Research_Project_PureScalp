"""
Utility functions for scalp classification system
"""
import os
import cv2
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from config import Config


# Scalp TYPE Dataset 
class ScalpDataset(Dataset):
    def __init__(self, csv_file: str, images_dir: str, transform=None):
        self.data = pd.read_csv(csv_file)
        self.images_dir = images_dir
        self.transform = transform

        self.data = self._process_labels(self.data)

        self.classes = ['dandruff', 'dandruff_sensitive', 'oily', 'sensitive']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def _process_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = df.columns.str.strip()

        def get_scalp_type(row):
            if row.get('d', 0) == 1 and row.get('ds', 0) == 0 and row.get('o', 0) == 0 and row.get('s', 0) == 0:
                return 'dandruff'
            elif row.get('ds', 0) == 1:
                return 'dandruff_sensitive'
            elif row.get('o', 0) == 1:
                return 'oily'
            elif row.get('s', 0) == 1:
                return 'sensitive'
            return None

        df['scalp_type'] = df.apply(get_scalp_type, axis=1)
        df = df.dropna(subset=['scalp_type']).reset_index(drop=True)
        return df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['filename']
        img_path = os.path.join(self.images_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)['image']

        label = self.class_to_idx[self.data.iloc[idx]['scalp_type']]
        return image, torch.tensor(label, dtype=torch.long)


# FIXED: Scalp Validator Dataset (AUTO CSV SUPPORT)
class ScalpValidatorDataset(Dataset):
    """
    Binary dataset:
      non_scalp -> 0
      scalp     -> 1

    Supports CSVs WITHOUT 'filename' column
    (uses first column automatically)
    """

    def __init__(
        self,
        scalp_csv: str,
        scalp_images_dir: str,
        nonscalp_csv: str,
        nonscalp_images_dir: str,
        transform=None
    ):
        self.transform = transform

        scalp_df = pd.read_csv(scalp_csv)
        nonscalp_df = pd.read_csv(nonscalp_csv)

        # AUTO-DETECT FIRST COLUMN
        scalp_col = scalp_df.columns[0]
        nonscalp_col = nonscalp_df.columns[0]

        scalp_df = scalp_df[[scalp_col]].rename(columns={scalp_col: 'filename'})
        nonscalp_df = nonscalp_df[[nonscalp_col]].rename(columns={nonscalp_col: 'filename'})

        scalp_df['label'] = 1
        scalp_df['images_dir'] = scalp_images_dir

        nonscalp_df['label'] = 0
        nonscalp_df['images_dir'] = nonscalp_images_dir

        self.data = pd.concat([scalp_df, nonscalp_df], ignore_index=True)
        self.data = self.data.sample(frac=1.0, random_state=42).reset_index(drop=True)

        self.class_names = ['non_scalp', 'scalp']

        print("🧪 Validator Dataset Loaded")
        print(self.data['label'].value_counts().rename({0: 'non_scalp', 1: 'scalp'}))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(row['images_dir'], row['filename'])

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)['image']

        return image, torch.tensor(row['label'], dtype=torch.long)


# Transforms
def get_transforms(is_training=True):
    if is_training:
        return A.Compose([
            A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20, p=0.5),
            A.ColorJitter(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


# Data Loaders
def create_data_loaders():
    train_ds = ScalpDataset(Config.TRAIN_CSV_PATH, Config.TRAIN_IMAGES_PATH, get_transforms(True))
    val_ds = ScalpDataset(Config.VALID_CSV_PATH, Config.VALID_IMAGES_PATH, get_transforms(False))

    return (
        DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True),
        DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False),
        train_ds.classes
    )


def create_validator_data_loaders():
    train_ds = ScalpValidatorDataset(
        Config.TRAIN_CSV_PATH,
        Config.TRAIN_IMAGES_PATH,
        Config.NONSCALP_TRAIN_CSV_PATH,
        Config.NONSCALP_TRAIN_IMAGES_PATH,
        get_transforms(True)
    )

    val_ds = ScalpValidatorDataset(
        Config.VALID_CSV_PATH,
        Config.VALID_IMAGES_PATH,
        Config.NONSCALP_VALID_CSV_PATH,
        Config.NONSCALP_VALID_IMAGES_PATH,
        get_transforms(False)
    )

    return (
        DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True),
        DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False),
        train_ds.class_names
    )


# Metrics 
def calculate_metrics(y_true, y_pred, class_names):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'classification_report': classification_report(y_true, y_pred, target_names=class_names),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }

# Plotting & Saving Utilities (REQUIRED by train.py)

def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    import matplotlib.pyplot as plt
    import os

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(14, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label='Train Accuracy')
    plt.plot(epochs, val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training vs Validation Accuracy')
    plt.legend()
    plt.grid(True)

    os.makedirs(Config.ML_MODELS_PATH, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.ML_MODELS_PATH, 'training_history.png'))
    plt.show()


def plot_confusion_matrix(cm, class_names):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    os.makedirs(Config.ML_MODELS_PATH, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.ML_MODELS_PATH, 'confusion_matrix.png'))
    plt.show()


def save_class_names(class_names):
    import json
    import os

    os.makedirs(Config.ML_MODELS_PATH, exist_ok=True)
    path = Config.CLASS_NAMES_PATH

    with open(path, 'w') as f:
        json.dump(class_names, f, indent=4)

    print(f"Class names saved to: {path}")

# Runtime utilities required by Flask routes (app.py)

def create_upload_folder():
    """
    Ensure upload folder exists (Config.UPLOAD_FOLDER).
    Used by routes before saving uploaded images.
    """
    import os
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)


def load_class_names():
    """
    Load class names from Config.CLASS_NAMES_PATH.
    This is used by the classifier to map predicted index -> label.
    """
    import json
    if not os.path.exists(Config.CLASS_NAMES_PATH):
        raise FileNotFoundError(
            f"class_names.json not found at {Config.CLASS_NAMES_PATH}. "
            f"Train the classifier or ensure the file exists."
        )

    with open(Config.CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)

    return class_names


def preprocess_single_image(image_path: str) -> torch.Tensor:
    """
    Load one image from disk and preprocess it for PyTorch inference.
    Returns tensor shape: (1, 3, H, W)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = get_transforms(is_training=False)
    transformed = transform(image=image)
    image_tensor = transformed["image"].unsqueeze(0)  # add batch dimension
    return image_tensor


def is_scalp_image(image_path: str, validator_model, threshold: float = None):
    """
    Use the ScalpValidator model to check whether an image is a scalp image.

    Returns:
      (is_scalp: bool, scalp_confidence: float)
    """
    if threshold is None:
        threshold = Config.SCALP_VALIDATION_THRESHOLD

    image_tensor = preprocess_single_image(image_path).to(Config.DEVICE)

    validator_model.eval()
    with torch.no_grad():
        outputs = validator_model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        scalp_conf = float(probs[0][1].item())  # index 1 = "scalp"

    return scalp_conf >= threshold, scalp_conf
