import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torchvision.models as models

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def get_data_loaders():
    print("\nChecking CSV file structure...")
    try:
        df = pd.read_csv(config.csv_file)
        print("CSV columns:", df.columns.tolist())
        print("\nFirst few rows:")
        print(df.head())
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        raise

# Configuration
class Config:
    data_dir = 'data'
    csv_file = os.path.join(data_dir, 'labels.csv')
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    num_classes = None  # Will be set based on data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_save_path = 'doseguard_model.pth'

config = Config()

# Dataset class
class MedicineDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.medicine_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
        # Get column names from CSV
        columns = self.medicine_frame.columns.tolist()
        print("Available columns:", columns)
        
        # Assuming first column is image path and second column is class label
        self.image_column = columns[0]
        self.label_column = columns[1]
        
        print(f"Using columns - Image: {self.image_column}, Label: {self.label_column}")
        
        # Get unique classes and convert to strings
        self.classes = [str(c) for c in self.medicine_frame[self.label_column].unique()]
        self.label_to_idx = {label: idx for idx, label in enumerate(self.classes)}
    
    def __len__(self):
        # Add this method to return the length of the dataset
        return len(self.medicine_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir, str(self.medicine_frame.iloc[idx][self.image_column]))
        image = Image.open(img_name).convert('RGB')
        label = str(self.medicine_frame.iloc[idx][self.label_column])  # Convert to string
        label_idx = self.label_to_idx[label]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label_idx, label

# CNN Model
class DoseGuardCNN(nn.Module):
    def __init__(self, num_classes):
        super(DoseGuardCNN, self).__init__()
        # Use a pre-trained ResNet as the backbone
        self.model = models.resnet18(pretrained=True)
        
        # Replace the final fully connected layer with one matching our number of classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

# Data preprocessing and loading
def get_data_loaders():
    # Define transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create the dataset
    full_dataset = MedicineDataset(csv_file=config.csv_file, root_dir=config.data_dir, transform=None)
    
    # Set the number of classes
    config.num_classes = len(full_dataset.classes)
    print(f"Number of classes: {config.num_classes}")
    print(f"Classes: {full_dataset.classes}")
    
    # Split the dataset
    train_indices, test_indices = train_test_split(range(len(full_dataset)), test_size=0.2, random_state=42, stratify=[full_dataset[i][1] for i in range(len(full_dataset))])
    
    # Create train and test datasets with appropriate transforms
    train_dataset = MedicineDataset(csv_file=config.csv_file, root_dir=config.data_dir, transform=train_transform)
    test_dataset = MedicineDataset(csv_file=config.csv_file, root_dir=config.data_dir, transform=test_transform)
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader, full_dataset.classes, full_dataset.label_to_idx

# Training function
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels, _ in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    
    return train_loss, train_acc

# Evaluation function
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    all_original_labels = []
    
    with torch.no_grad():
        for inputs, labels, original_labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())
            all_original_labels.extend(original_labels)
    
    test_loss = running_loss / len(test_loader)
    test_acc = 100. * correct / total
    
    return test_loss, test_acc, all_preds, all_labels, all_probs, all_original_labels

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

# Function to plot model performance
def plot_performance(train_losses, train_accs, test_losses, test_accs):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Testing Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Testing Accuracy')
    
    plt.tight_layout()
    plt.savefig('model_performance.png')
    plt.show()

# Function to visualize prediction confidence
def plot_prediction_confidence(probs, labels, class_names):
    plt.figure(figsize=(12, 6))
    
    # Get the prediction probabilities for the correct class
    correct_class_probs = [probs[i][labels[i]] for i in range(len(probs))]
    
    plt.hist(correct_class_probs, bins=20, alpha=0.7)
    plt.axvline(x=0.5, color='red', linestyle='--', label='50% Confidence')
    plt.xlabel('Prediction Confidence for Correct Class')
    plt.ylabel('Count')
    plt.title('Model Confidence in Predictions')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('prediction_confidence.png')
    plt.show()

# Main training loop
def main():
    print(f"Using device: {config.device}")
    
    # Get data loaders
    train_loader, test_loader, classes, label_to_idx = get_data_loaders()
    
    # Initialize the model
    model = DoseGuardCNN(config.num_classes).to(config.device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Lists to store metrics for plotting
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    # Training loop
    print("Starting training...")
    for epoch in range(config.num_epochs):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, config.device)
        test_loss, test_acc, _, _, _, _ = evaluate_model(model, test_loader, criterion, config.device)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        print(f"Epoch {epoch+1}/{config.num_epochs}, "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    # Save the model
    torch.save(model.state_dict(), config.model_save_path)
    print(f"Model saved to {config.model_save_path}")
    
    # Final evaluation
    _, _, all_preds, all_labels, all_probs, all_original_labels = evaluate_model(model, test_loader, criterion, config.device)
    
    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_preds, classes)
    
    # Plot model performance
    plot_performance(train_losses, train_accs, test_losses, test_accs)
    
    # Plot prediction confidence
    plot_prediction_confidence(all_probs, all_labels, classes)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))

if __name__ == "__main__":
    main()