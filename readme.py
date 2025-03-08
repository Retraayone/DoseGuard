# Import required libraries for data handling and visualization
import os                  # For file and path operations
import pandas as pd        # For handling CSV files and data structures
import numpy as np        # For numerical computations
import matplotlib.pyplot as plt   # For creating plots
import seaborn as sns     # For enhanced visualizations
from sklearn.metrics import classification_report, confusion_matrix  # For model evaluation
from sklearn.model_selection import train_test_split  # For splitting dataset

# Import PyTorch-related libraries
import torch              # Main PyTorch library
import torch.nn as nn     # Neural network modules
import torch.optim as optim  # Optimization algorithms
from torch.utils.data import Dataset, DataLoader  # For dataset handling
from torchvision import transforms  # For image transformations
from PIL import Image     # For image loading and processing
import torchvision.models as models  # For pre-trained models

# Set random seeds for reproducibility
torch.manual_seed(42)    # PyTorch random seed
np.random.seed(42)       # NumPy random seed

# Configuration class to store all parameters
class Config:
    # Data-related settings
    data_dir = 'data'    # Directory containing image data
    csv_file = os.path.join(data_dir, 'labels.csv')  # Path to labels file
    
    # Training hyperparameters
    batch_size = 32      # Number of images per training batch
    num_epochs = 10      # Number of training cycles
    learning_rate = 0.001  # Learning rate for optimizer
    
    # Model parameters
    num_classes = None   # Will be set dynamically based on dataset
    
    # Hardware settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available
    
    # Model saving
    model_save_path = 'doseguard_model.pth'  # Path to save trained model

# Create global configuration instance
config = Config()

class MedicineDataset(Dataset):
    """Custom Dataset class for handling medicine images"""
    def __init__(self, csv_file, root_dir, transform=None):
        # Load CSV data and initialize settings
        self.medicine_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
        # Get column names from CSV
        columns = self.medicine_frame.columns.tolist()
        print("Available columns:", columns)
        
        # Set column names for data
        self.image_column = columns[0]  # First column contains image paths
        self.label_column = columns[1]  # Second column contains labels
        
        print(f"Using columns - Image: {self.image_column}, Label: {self.label_column}")
        
        # Create class mapping
        self.classes = [str(c) for c in self.medicine_frame[self.label_column].unique()]
        self.label_to_idx = {label: idx for idx, label in enumerate(self.classes)}
    
    def __len__(self):
        """Return total number of samples"""
        return len(self.medicine_frame)
    
    def __getitem__(self, idx):
        """Get a single sample from the dataset"""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Load and process image
        img_name = os.path.join(self.root_dir, str(self.medicine_frame.iloc[idx][self.image_column]))
        image = Image.open(img_name).convert('RGB')
        
        # Get and process label
        label = str(self.medicine_frame.iloc[idx][self.label_column])
        label_idx = self.label_to_idx[label]
        
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        
        return image, label_idx, label

class DoseGuardCNN(nn.Module):
    """CNN Model architecture using ResNet18 backbone"""
    def __init__(self, num_classes):
        super(DoseGuardCNN, self).__init__()
        # Load pre-trained ResNet18
        self.model = models.resnet18(pretrained=True)
        
        # Modify final layer for our specific number of classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        """Forward pass through the network"""
        return self.model(x)

def get_data_loaders():
    """Prepare and return data loaders for training and testing"""
    # Define training data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),      # Resize to ResNet input size
        transforms.RandomHorizontalFlip(),   # Data augmentation
        transforms.RandomRotation(10),       # Random rotation up to 10 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Color augmentation
        transforms.ToTensor(),              # Convert to tensor
        transforms.Normalize(                # Normalize using ImageNet stats
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Define test data transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Create initial dataset
    full_dataset = MedicineDataset(
        csv_file=config.csv_file,
        root_dir=config.data_dir,
        transform=None
    )
    
    # Set number of classes
    config.num_classes = len(full_dataset.classes)
    print(f"Number of classes: {config.num_classes}")
    print(f"Classes: {full_dataset.classes}")
    
    # Split dataset into train and test sets
    train_indices, test_indices = train_test_split(
        range(len(full_dataset)),
        test_size=0.2,
        random_state=42,
        stratify=[full_dataset[i][1] for i in range(len(full_dataset))]
    )
    
    # Create datasets with transforms
    train_dataset = MedicineDataset(
        csv_file=config.csv_file,
        root_dir=config.data_dir,
        transform=train_transform
    )
    test_dataset = MedicineDataset(
        csv_file=config.csv_file,
        root_dir=config.data_dir,
        transform=test_transform
    )
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, test_loader, full_dataset.classes, full_dataset.label_to_idx

def train_model(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch"""
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels, _ in train_loader:
        # Move data to device
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Track statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    # Calculate epoch statistics
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    
    return train_loss, train_acc

def evaluate_model(model, test_loader, criterion, device):
    """Evaluate the model on test data"""
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    all_original_labels = []
    
    with torch.no_grad():  # No gradient computation needed
        for inputs, labels, original_labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Calculate probabilities and predictions
            _, predicted = outputs.max(1)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Track statistics
            running_loss += loss.item()
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Store predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())
            all_original_labels.extend(original_labels)
    
    # Calculate test statistics
    test_loss = running_loss / len(test_loader)
    test_acc = 100. * correct / total
    
    return test_loss, test_acc, all_preds, all_labels, all_probs, all_original_labels

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

def plot_performance(train_losses, train_accs, test_losses, test_accs):
    """Plot and save training/testing performance metrics"""
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Testing Loss')
    
    # Plot accuracies
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

def plot_prediction_confidence(probs, labels, class_names):
    """Plot and save model prediction confidence distribution"""
    plt.figure(figsize=(12, 6))
    
    # Calculate confidence for correct predictions
    correct_class_probs = [probs[i][labels[i]] for i in range(len(probs))]
    
    # Create histogram
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

def main():
    """Main training and evaluation routine"""
    print(f"Using device: {config.device}")
    
    # Prepare data
    train_loader, test_loader, classes, label_to_idx = get_data_loaders()
    
    # Initialize model, loss function, and optimizer
    model = DoseGuardCNN(config.num_classes).to(config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Initialize tracking lists
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    
    # Training loop
    print("Starting training...")
    for epoch in range(config.num_epochs):
        # Train and evaluate for one epoch
        train_loss, train_acc = train_model(
            model, train_loader, criterion, optimizer, config.device
        )
        test_loss, test_acc, _, _, _, _ = evaluate_model(
            model, test_loader, criterion, config.device
        )
        
        # Store metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        # Print progress
        print(f"Epoch {epoch+1}/{config.num_epochs}, "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    # Save trained model
    torch.save(model.state_dict(), config.model_save_path)
    print(f"Model saved to {config.model_save_path}")
    
    # Final evaluation
    _, _, all_preds, all_labels, all_probs, all_original_labels = evaluate_model(
        model, test_loader, criterion, config.device
    )
    
    # Generate visualizations
    plot_confusion_matrix(all_labels, all_preds, classes)
    plot_performance(train_losses, train_accs, test_losses, test_accs)
    plot_prediction_confidence(all_probs, all_labels, classes)
    
    # Print final classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))

if __name__ == "__main__":
    main()