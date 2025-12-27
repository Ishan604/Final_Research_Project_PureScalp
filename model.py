"""
PyTorch models for scalp classification system
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from config import Config

class ScalpValidator(nn.Module):
    """
    Binary classification model to validate if an image is a scalp image
    Uses MobileNetV2 as backbone for efficiency
    """
    
    def __init__(self):
        super(ScalpValidator, self).__init__()
        
        # Load pre-trained MobileNetV2
        self.backbone = models.mobilenet_v2(pretrained=True)
        
        # Freeze early layers
        for param in self.backbone.features[:10].parameters():
            param.requires_grad = False
        
        # Modify classifier for binary classification (scalp vs non-scalp)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # Binary classification: [non-scalp, scalp]
        )
    
    def forward(self, x):
        return self.backbone(x)

class ScalpClassifier(nn.Module):
    """
    Multi-class classification model for scalp type classification
    Uses EfficientNet-B0 as backbone for better accuracy
    """
    
    def __init__(self, num_classes: int = Config.NUM_CLASSES):
        super(ScalpClassifier, self).__init__()
        
        # Load pre-trained EfficientNet-B0
        self.backbone = models.efficientnet_b0(pretrained=True)
        
        # Freeze early layers
        for param in self.backbone.features[:5].parameters():
            param.requires_grad = False
        
        # Modify classifier for scalp type classification
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class EarlyStopping:
    """Early stopping utility to prevent overfitting"""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.001, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        """Save model weights"""
        self.best_weights = model.state_dict().copy()

def create_scalp_validator() -> ScalpValidator:
    """Create and return scalp validator model"""
    model = ScalpValidator()
    return model.to(Config.DEVICE)

def create_scalp_classifier(num_classes: int = Config.NUM_CLASSES) -> ScalpClassifier:
    """Create and return scalp classifier model"""
    model = ScalpClassifier(num_classes)
    return model.to(Config.DEVICE)

def save_model(model: nn.Module, filepath: str, class_names: list = None):
    """Save PyTorch model"""
    
    # Create directory if it doesn't exist
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Prepare save dictionary
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
    }
    
    # Add class names if provided
    if class_names is not None:
        save_dict['class_names'] = class_names
    
    # Save model
    torch.save(save_dict, filepath)
    print(f"Model saved to: {filepath}")

def load_model(filepath: str, model_class, num_classes: int = None) -> nn.Module:
    """Load PyTorch model"""
    
    try:
        # Load checkpoint
        checkpoint = torch.load(filepath, map_location=Config.DEVICE)
        
        # Create model instance
        if model_class == ScalpValidator:
            model = ScalpValidator()
        elif model_class == ScalpClassifier:
            if num_classes is None:
                num_classes = Config.NUM_CLASSES
            model = ScalpClassifier(num_classes)
        else:
            raise ValueError(f"Unknown model class: {model_class}")
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(Config.DEVICE)
        model.eval()
        
        print(f"Model loaded from: {filepath}")
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_info(model: nn.Module) -> dict:
    """Get model information"""
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = count_parameters(model)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'frozen_parameters': total_params - trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Approximate size in MB
        'device': next(model.parameters()).device
    }