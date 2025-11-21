"""
Configuration file for scalp classification system
"""
import torch
import os

class Config:
    # Data paths
    DATASET_PATH = 'datasets'
    TRAIN_DATA_PATH = os.path.join(DATASET_PATH, 'train')
    VALID_DATA_PATH = os.path.join(DATASET_PATH, 'valid')
    TRAIN_IMAGES_PATH = os.path.join(TRAIN_DATA_PATH, 'images')
    VALID_IMAGES_PATH = os.path.join(VALID_DATA_PATH, 'images')
    TRAIN_CSV_PATH = os.path.join(TRAIN_DATA_PATH, 'train_labels.csv')
    VALID_CSV_PATH = os.path.join(VALID_DATA_PATH, 'valid_labels.csv')
    
    # Model save paths
    ML_MODELS_PATH = 'ml_models'
    SCALP_CLASSIFIER_PATH = os.path.join(ML_MODELS_PATH, 'scalp_classifier.pth')
    SCALP_VALIDATOR_PATH = os.path.join(ML_MODELS_PATH, 'scalp_validator.pth')
    CLASS_NAMES_PATH = os.path.join(ML_MODELS_PATH, 'class_names.json')
    
    # Image parameters
    IMG_SIZE = 224
    IMG_CHANNELS = 3
    
    # Training parameters
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    PATIENCE = 15  # Early stopping patience
    
    # Model parameters
    NUM_CLASSES = 4  # dandruff (d), dandruff+sensitive (ds), oily (o), sensitive (s)
    DROPOUT_RATE = 0.5
    CLASS_MAPPING = {
        'd': 'dandruff',
        'ds': 'dandruff_sensitive', 
        'o': 'oily',
        's': 'sensitive'
    }
    
    # Device configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data augmentation parameters
    ROTATION_RANGE = 20
    BRIGHTNESS_RANGE = 0.2
    CONTRAST_RANGE = 0.2
    SATURATION_RANGE = 0.2
    HUE_RANGE = 0.1
    
    # Upload configuration
    UPLOAD_FOLDER = 'static/uploads'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
    
    # Validation thresholds
    SCALP_VALIDATION_THRESHOLD = 0.7  # Confidence threshold for scalp validation
    CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for classification
    REQUIRE_SCALP_VALIDATION = False  # If True and validator missing, do NOT classify