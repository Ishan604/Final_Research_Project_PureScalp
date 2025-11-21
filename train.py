"""
Main training script for scalp classification system
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from typing import List, Tuple

from config import Config
from model import (
    create_scalp_classifier, 
    EarlyStopping, 
    save_model, 
    get_model_info
)
from utils import (
    create_data_loaders,
    calculate_metrics,
    plot_training_history,
    plot_confusion_matrix,
    save_class_names
)

def train_epoch(model: nn.Module, 
                train_loader, 
                criterion, 
                optimizer, 
                device: torch.device) -> Tuple[float, float]:
    """Train model for one epoch"""
    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    train_bar = tqdm(train_loader, desc='Training')
    
    for batch_idx, (images, labels) in enumerate(train_bar):
        # Move data to device
        images, labels = images.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        train_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100. * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate_epoch(model: nn.Module, 
                  val_loader, 
                  criterion, 
                  device: torch.device) -> Tuple[float, float, List[int], List[int]]:
    """Validate model for one epoch"""
    
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predicted = []
    all_labels = []
    
    val_bar = tqdm(val_loader, desc='Validation')
    
    with torch.no_grad():
        for images, labels in val_bar:
            # Move data to device
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store predictions for metrics calculation
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            val_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc, all_predicted, all_labels

def train_scalp_classifier():
    """Main function to train the scalp classifier"""
    
    print("🚀 Starting Scalp Classification Training")
    print("=" * 50)
    
    # Check if required folders exist
    if not os.path.exists(Config.DATASET_PATH):
        print(f"❌ Dataset folder not found: {Config.DATASET_PATH}")
        return
    
    if not os.path.exists(Config.TRAIN_CSV_PATH):
        print(f"❌ Training CSV not found: {Config.TRAIN_CSV_PATH}")
        return
        
    if not os.path.exists(Config.VALID_CSV_PATH):
        print(f"❌ Validation CSV not found: {Config.VALID_CSV_PATH}")
        return
    
    # Create ml_models folder if it doesn't exist
    os.makedirs(Config.ML_MODELS_PATH, exist_ok=True)
    
    # Check device
    print(f"Using device: {Config.DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Create data loaders
    print("📊 Loading data...")
    try:
        train_loader, val_loader, class_names = create_data_loaders()
        print(f"✅ Data loaded successfully!")
        print(f"Classes: {class_names}")
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        print()
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Update config with actual number of classes
    Config.NUM_CLASSES = len(class_names)
    
    # Create model
    print("🏗️  Creating model...")
    model = create_scalp_classifier(len(class_names))
    
    # Print model info
    model_info = get_model_info(model)
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {model_info['total_parameters']:,}")
    print(f"Trainable parameters: {model_info['trainable_parameters']:,}")
    print(f"Model size: {model_info['model_size_mb']:.1f} MB")
    print()
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=Config.PATIENCE, restore_best_weights=True)
    
    # Training history
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Training loop
    print("🎯 Starting training...")
    start_time = time.time()
    
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")
        print("-" * 30)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
        
        # Validate
        val_loss, val_acc, val_predicted, val_labels = validate_epoch(model, val_loader, criterion, Config.DEVICE)
        
        # Update learning rate
        scheduler.step()
        
        # Store history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Print epoch results
        print(f"Training   - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Early stopping check
        if early_stopping(val_loss, model):
            print(f"\\n⏹️  Early stopping triggered at epoch {epoch+1}")
            break
    
    # Training completed
    training_time = time.time() - start_time
    print(f"\\n✅ Training completed in {training_time/60:.2f} minutes")
    
    # Final evaluation
    print("\\n📈 Final Evaluation")
    print("=" * 30)
    
    # Get final predictions for detailed metrics
    model.eval()
    final_predicted = []
    final_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            final_predicted.extend(predicted.cpu().numpy())
            final_labels.extend(labels.cpu().numpy())
    
    # Calculate detailed metrics
    metrics = calculate_metrics(final_labels, final_predicted, class_names)
    
    # Print final results
    print(f"Average Training Loss: {np.mean(train_losses):.4f}")
    print(f"Average Validation Loss: {np.mean(val_losses):.4f}")
    print(f"Final Training Accuracy: {train_accs[-1]:.2f}%")
    print(f"Final Validation Accuracy: {val_accs[-1]:.2f}%")
    print()
    
    print("📊 Final Evaluation Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print()
    
    print("📋 Detailed Classification Report:")
    print(metrics['classification_report'])
    
    # Plot results
    print("📈 Generating plots...")
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    plot_confusion_matrix(metrics['confusion_matrix'], class_names)
    
    # Save model
    print("💾 Saving model...")
    save_model(model, Config.SCALP_CLASSIFIER_PATH, class_names)
    save_class_names(class_names)

if __name__ == "__main__":
    train_scalp_classifier()