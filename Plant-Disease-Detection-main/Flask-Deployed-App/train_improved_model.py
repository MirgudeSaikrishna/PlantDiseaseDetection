"""
Enhanced Model Training Script for Plant Disease Detection
Author: [Your Name]
Date: [Current Date]

This script implements advanced training techniques including:
- Data augmentation
- Learning rate scheduling
- Early stopping
- Model checkpointing
- Advanced metrics tracking
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from datetime import datetime
import os
import json
from CNN import ImprovedCNN, CNN

class PlantDiseaseTrainer:
    def __init__(self, model_name="improved_cnn", num_classes=39, device=None):
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def create_data_transforms(self):
        """Create data augmentation transforms for training and validation"""
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return train_transform, val_transform
    
    def load_dataset(self, data_path, train_transform, val_transform):
        """Load and split the dataset"""
        # Load full dataset
        full_dataset = datasets.ImageFolder(data_path, transform=train_transform)
        
        # Split dataset
        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.15 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )
        
        # Apply validation transform to validation and test sets
        val_dataset.dataset.transform = val_transform
        test_dataset.dataset.transform = val_transform
        
        return train_dataset, val_dataset, test_dataset
    
    def create_data_loaders(self, train_dataset, val_dataset, test_dataset, batch_size=32):
        """Create data loaders with appropriate batch sizes"""
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        return train_loader, val_loader, test_loader
    
    def initialize_model(self, use_improved=True):
        """Initialize the model"""
        if use_improved:
            self.model = ImprovedCNN(self.num_classes)
        else:
            self.model = ImprovedCNN(self.num_classes)  # Use original CNN
        
        self.model = self.model.to(self.device)
        return self.model
    
    def train_epoch(self, model, train_loader, criterion, optimizer):
        """Train for one epoch"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, model, val_loader, criterion):
        """Validate for one epoch"""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, epochs=50, lr=0.001, patience=10):
        """Main training loop with early stopping"""
        # Initialize model
        self.initialize_model(use_improved=True)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        print(f"Training on {self.device}")
        print(f"Model: {self.model_name}")
        print(f"Epochs: {epochs}, Learning Rate: {lr}")
        print("-" * 50)
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(self.model, train_loader, criterion, optimizer)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(self.model, val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            print("-" * 30)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        return self.model
    
    def evaluate(self, test_loader):
        """Evaluate model on test set"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = output.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Calculate metrics
        accuracy = 100. * sum(p == t for p, t in zip(all_predictions, all_targets)) / len(all_targets)
        
        print(f"Test Accuracy: {accuracy:.2f}%")
        
        return all_predictions, all_targets, accuracy
    
    def plot_training_history(self, save_path="training_history.png"):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath):
        """Save the trained model"""
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")
    
    def save_training_info(self, filepath, test_accuracy):
        """Save training information"""
        info = {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "device": str(self.device),
            "final_train_loss": self.train_losses[-1] if self.train_losses else None,
            "final_val_loss": self.val_losses[-1] if self.val_losses else None,
            "final_train_acc": self.train_accuracies[-1] if self.train_accuracies else None,
            "final_val_acc": self.val_accuracies[-1] if self.val_accuracies else None,
            "test_accuracy": test_accuracy,
            "training_date": datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"Training info saved to {filepath}")

def main():
    """Main training function"""
    # Configuration
    DATA_PATH = "Dataset"  # Update with your dataset path
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    PATIENCE = 10
    
    # Create trainer
    trainer = PlantDiseaseTrainer()
    
    # Create transforms
    train_transform, val_transform = trainer.create_data_transforms()
    
    # Load dataset
    print("Loading dataset...")
    train_dataset, val_dataset, test_dataset = trainer.load_dataset(
        DATA_PATH, train_transform, val_transform
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = trainer.create_data_loaders(
        train_dataset, val_dataset, test_dataset, BATCH_SIZE
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Train model
    print("Starting training...")
    model = trainer.train(train_loader, val_loader, EPOCHS, LEARNING_RATE, PATIENCE)
    
    # Evaluate model
    print("Evaluating model...")
    predictions, targets, test_accuracy = trainer.evaluate(test_loader)
    
    # Plot training history
    trainer.plot_training_history("training_history.png")
    
    # Save model and info
    trainer.save_model("improved_plant_disease_model.pt")
    trainer.save_training_info("training_info.json", test_accuracy)
    
    print("Training completed!")

if __name__ == "__main__":
    main()
