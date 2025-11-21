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
from PIL import Image
from typing import Tuple, List, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from config import Config

class ScalpDataset(Dataset):
    """Custom dataset for scalp images"""
    
    def __init__(self, csv_file: str, images_dir: str, transform=None):
        self.data = pd.read_csv(csv_file)
        self.images_dir = images_dir
        self.transform = transform
        
        # Convert one-hot encoding to single labels
        self.data = self._process_labels(self.data)
        
        # Create label mapping
        self.classes = ['dandruff', 'dandruff_sensitive', 'oily', 'sensitive']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        print(f"Dataset loaded: {len(self.data)} images")
        print(f"Classes distribution:")
        print(self.data['scalp_type'].value_counts())
        
    def _process_labels(self, df):
        """Convert one-hot encoding to single scalp_type column"""
        
        # Clean column names (remove spaces)
        df.columns = df.columns.str.strip()
        
        def get_scalp_type(row):
            if row['d'] == 1 and row['ds'] == 0 and row['o'] == 0 and row['s'] == 0:
                return 'dandruff'
            elif row['ds'] == 1:
                return 'dandruff_sensitive'
            elif row['o'] == 1:
                return 'oily'
            elif row['s'] == 1:
                return 'sensitive'
            else:
                # Handle null images or multiple labels - skip these for now
                return None
        
        df['scalp_type'] = df.apply(get_scalp_type, axis=1)
        # Remove rows with None (null images or invalid labels)
        df = df.dropna(subset=['scalp_type'])
        df = df.reset_index(drop=True)
        return df
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get image path and label
        img_name = self.data.iloc[idx]['filename']
        img_path = os.path.join(self.images_dir, img_name)
        label = self.data.iloc[idx]['scalp_type']
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # Convert label to index
        label_idx = self.class_to_idx[label]
        
        return image, torch.tensor(label_idx, dtype=torch.long)

def get_transforms(is_training: bool = True) -> A.Compose:
    """Get image transforms for training or validation"""
    
    if is_training:
        transform = A.Compose([
            A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
            A.Rotate(limit=Config.ROTATION_RANGE, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(
                brightness=Config.BRIGHTNESS_RANGE,
                contrast=Config.CONTRAST_RANGE,
                saturation=Config.SATURATION_RANGE,
                hue=Config.HUE_RANGE,
                p=0.5
            ),
            A.GaussianBlur(blur_limit=3, p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    return transform

def create_data_loaders() -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders"""
    
    # Create datasets
    train_dataset = ScalpDataset(
        csv_file=Config.TRAIN_CSV_PATH,
        images_dir=Config.TRAIN_IMAGES_PATH,
        transform=get_transforms(is_training=True)
    )
    
    valid_dataset = ScalpDataset(
        csv_file=Config.VALID_CSV_PATH,
        images_dir=Config.VALID_IMAGES_PATH,
        transform=get_transforms(is_training=False)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, valid_loader, train_dataset.classes

def preprocess_single_image(image_path: str) -> torch.Tensor:
    """Preprocess a single image for prediction"""
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply transforms
    transform = get_transforms(is_training=False)
    transformed = transform(image=image)
    image_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension
    
    return image_tensor

def is_scalp_image(image_path: str, validator_model, threshold: float = Config.SCALP_VALIDATION_THRESHOLD) -> Tuple[bool, float]:
    """
    Validate if an image is a scalp image using binary classification
    Returns: (is_scalp, confidence)
    """
    try:
        # Preprocess image
        image_tensor = preprocess_single_image(image_path)
        
        # Move to device
        image_tensor = image_tensor.to(Config.DEVICE)
        
        # Make prediction
        validator_model.eval()
        with torch.no_grad():
            outputs = validator_model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            scalp_confidence = probabilities[0][1].item()  # Assuming class 1 is 'scalp'
        
        is_scalp = scalp_confidence >= threshold
        return is_scalp, scalp_confidence
        
    except Exception as e:
        print(f"Error in scalp validation: {e}")
        return False, 0.0

def calculate_metrics(y_true: List[int], y_pred: List[int], class_names: List[str]) -> Dict[str, Any]:
    """Calculate evaluation metrics"""
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted'),
        'classification_report': classification_report(y_true, y_pred, target_names=class_names),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    return metrics

def plot_training_history(train_losses: List[float], val_losses: List[float], train_accs: List[float], val_accs: List[float]):
    """Plot training history"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_accs, label='Training Accuracy', color='blue')
    ax2.plot(val_accs, label='Validation Accuracy', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.ML_MODELS_PATH, 'training_history.png'))
    plt.show()

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str]):
    """Plot confusion matrix"""
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.ML_MODELS_PATH, 'confusion_matrix.png'))
    plt.show()

def save_class_names(class_names: List[str]):
    """Save class names to JSON file"""
    os.makedirs(Config.ML_MODELS_PATH, exist_ok=True)
    with open(Config.CLASS_NAMES_PATH, 'w') as f:
        json.dump(class_names, f)
    print(f"✅ Class names saved to: {Config.CLASS_NAMES_PATH}")

def load_class_names() -> List[str]:
    """Load class names from JSON file"""
    try:
        with open(Config.CLASS_NAMES_PATH, 'r') as f:
            class_names = json.load(f)
        return class_names
    except FileNotFoundError:
        print(f"❌ Class names file not found: {Config.CLASS_NAMES_PATH}")
        return []

def validate_file_upload(file) -> Tuple[bool, str]:
    """Validate uploaded file"""
    
    if not file or file.filename == '':
        return False, "No file selected"
    
    # Check file extension
    if not ('.' in file.filename and 
            file.filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS):
        return False, f"Invalid file type. Allowed: {', '.join(Config.ALLOWED_EXTENSIONS)}"
    
    # Check file size (this should be done before reading the file in production)
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)  # Reset file pointer
    
    if file_size > Config.MAX_FILE_SIZE:
        return False, f"File too large. Maximum size: {Config.MAX_FILE_SIZE // (1024*1024)}MB"
    
    return True, "File is valid"

def create_upload_folder():
    """Create upload folder if it doesn't exist"""
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(Config.ML_MODELS_PATH, exist_ok=True)