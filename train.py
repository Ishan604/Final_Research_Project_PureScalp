"""
Main training script for scalp classification system
1) Train scalp validator (scalp vs non-scalp) and save scalp_validator.pth
2) Train scalp classifier (4 classes) and save scalp_classifier.pth
"""
import os
import time
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from config import Config
from model import (
    create_scalp_classifier,
    create_scalp_validator,
    EarlyStopping,
    save_model,
    get_model_info
)
from utils import (
    create_data_loaders,
    create_validator_data_loaders,
    calculate_metrics,
    plot_training_history,
    plot_confusion_matrix,
    save_class_names
)


def train_epoch(model: nn.Module, train_loader, criterion, optimizer, device: torch.device) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    train_bar = tqdm(train_loader, desc='Training')

    for images, labels in train_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        train_bar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100. * correct / total:.2f}%'})

    return running_loss / max(1, len(train_loader)), 100. * correct / max(1, total)


def validate_epoch(model: nn.Module, val_loader, criterion, device: torch.device) -> Tuple[float, float, List[int], List[int]]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predicted = []
    all_labels = []

    val_bar = tqdm(val_loader, desc='Validation')

    with torch.no_grad():
        for images, labels in val_bar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predicted.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

            val_bar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100. * correct / total:.2f}%'})

    return running_loss / max(1, len(val_loader)), 100. * correct / max(1, total), all_predicted, all_labels


# 1) Train + Save Scalp Validator (binary)
def train_scalp_validator():
    print("Training Scalp Validator (Scalp vs Non-Scalp)")
    print("=" * 60)

    # checks
    needed = [
        Config.TRAIN_CSV_PATH, Config.VALID_CSV_PATH,
        Config.NONSCALP_TRAIN_CSV_PATH, Config.NONSCALP_VALID_CSV_PATH,
        Config.TRAIN_IMAGES_PATH, Config.VALID_IMAGES_PATH,
        Config.NONSCALP_TRAIN_IMAGES_PATH, Config.NONSCALP_VALID_IMAGES_PATH
    ]
    for p in needed:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required path missing: {p}")

    os.makedirs(Config.ML_MODELS_PATH, exist_ok=True)

    print(f"Using device: {Config.DEVICE}\n")

    print("Loading validator data...")
    train_loader, val_loader, class_names = create_validator_data_loaders()
    print(f"Validator classes: {class_names}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}\n")

    print("Creating validator model...")
    model = create_scalp_validator()
    info = get_model_info(model)
    print(f"Model: {model.__class__.__name__}")
    print(f"Trainable params: {info['trainable_parameters']:,}")
    print(f"Model size: {info['model_size_mb']:.1f} MB\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    early_stopping = EarlyStopping(patience=Config.PATIENCE, restore_best_weights=True)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    print("Starting validator training...")
    start_time = time.time()

    for epoch in range(Config.NUM_EPOCHS):
        print(f"\n Epoch {epoch + 1}/{Config.NUM_EPOCHS}")
        print("-" * 30)

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
        val_loss, val_acc, val_pred, val_lab = validate_epoch(model, val_loader, criterion, Config.DEVICE)

        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Training   - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        print(f"LR: {scheduler.get_last_lr()[0]:.6f}")

        if early_stopping(val_loss, model):
            print(f"\n Early stopping triggered at epoch {epoch + 1}")
            break

    print(f"\n Validator training completed in {(time.time() - start_time) / 60:.2f} minutes")

    metrics = calculate_metrics(val_lab, val_pred, class_names)
    print("\n Validator Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1:       {metrics['f1_score']:.4f}")
    print(metrics['classification_report'])

    print("\n Saving plots...")
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    plot_confusion_matrix(metrics['confusion_matrix'], class_names)

    print("\n Saving validator model...")
    save_model(model, Config.SCALP_VALIDATOR_PATH, class_names)


# 2) Train + Save Scalp Classifier (4-class)
def train_scalp_classifier():
    print("\n Training Scalp Classifier (4 classes)")
    print("=" * 60)

    if not os.path.exists(Config.DATASET_PATH):
        raise FileNotFoundError(f"Dataset folder not found: {Config.DATASET_PATH}")
    if not os.path.exists(Config.TRAIN_CSV_PATH):
        raise FileNotFoundError(f"Training CSV not found: {Config.TRAIN_CSV_PATH}")
    if not os.path.exists(Config.VALID_CSV_PATH):
        raise FileNotFoundError(f"Validation CSV not found: {Config.VALID_CSV_PATH}")

    os.makedirs(Config.ML_MODELS_PATH, exist_ok=True)
    print(f"Using device: {Config.DEVICE}\n")

    print("Loading classifier data...")
    train_loader, val_loader, class_names = create_data_loaders()
    Config.NUM_CLASSES = len(class_names)

    print(f"Classes: {class_names}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}\n")

    print("Creating classifier model...")
    model = create_scalp_classifier(len(class_names))
    info = get_model_info(model)
    print(f"Model: {model.__class__.__name__}")
    print(f"Trainable params: {info['trainable_parameters']:,}")
    print(f"Model size: {info['model_size_mb']:.1f} MB\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    early_stopping = EarlyStopping(patience=Config.PATIENCE, restore_best_weights=True)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    print("Starting classifier training...")
    start_time = time.time()

    for epoch in range(Config.NUM_EPOCHS):
        print(f"\n Epoch {epoch + 1}/{Config.NUM_EPOCHS}")
        print("-" * 30)

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
        val_loss, val_acc, val_pred, val_lab = validate_epoch(model, val_loader, criterion, Config.DEVICE)

        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Training   - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        print(f"LR: {scheduler.get_last_lr()[0]:.6f}")

        if early_stopping(val_loss, model):
            print(f"\n Early stopping triggered at epoch {epoch + 1}")
            break

    print(f"\n Classifier training completed in {(time.time() - start_time) / 60:.2f} minutes")

    metrics = calculate_metrics(val_lab, val_pred, class_names)
    print("\n Classifier Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1:       {metrics['f1_score']:.4f}")
    print(metrics['classification_report'])

    print("\n Saving plots...")
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    plot_confusion_matrix(metrics['confusion_matrix'], class_names)

    print("\n Saving classifier model...")
    save_model(model, Config.SCALP_CLASSIFIER_PATH, class_names)
    save_class_names(class_names)


if __name__ == "__main__":
    train_scalp_validator()
    train_scalp_classifier()
