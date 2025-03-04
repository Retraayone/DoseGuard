## Overview
DoseGuard is a deep learning-based medicine classification system that uses computer vision to identify different types of medicines from images. The system utilizes a pre-trained ResNet18 model with custom modifications for medicine classification.



## Requirements

### Dependencies
```bash
pip install torch torchvision pandas numpy matplotlib seaborn scikit-learn pillow
```

### Data Format
The `labels.csv` file should have the following structure:
```csv
image_path,label
medicine1.jpg,Medicine_A
medicine2.jpg,Medicine_B
...
```

## Code Components

### 1. Configuration (`Config` class)
```python
class Config:
    data_dir = 'data'
    csv_file = os.path.join(data_dir, 'labels.csv')
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    num_classes = None  # Set automatically based on data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_save_path = 'doseguard_model.pth'
```

### 2. Dataset Handling (`MedicineDataset` class)
- Custom dataset class extending `torch.utils.data.Dataset`
- Handles image loading and preprocessing
- Converts labels to numerical indices

### 3. Model Architecture (`DoseGuardCNN` class)
- Based on pre-trained ResNet18
- Modified final layer for medicine classification
- Supports transfer learning

### 4. Training Pipeline
- Data augmentation for training:
  - Random horizontal flips
  - Random rotations
  - Color jittering
- Training/validation split (80/20)
- Adam optimizer
- CrossEntropy loss function

### 5. Evaluation Metrics
- Confusion matrix
- Classification report
- Training/testing loss curves
- Prediction confidence visualization

## Usage

### Training
```bash
python main.py
```

### Model Outputs
The system generates:
1. `confusion_matrix.png`: Visualization of model predictions
2. `model_performance.png`: Training and testing metrics
3. `prediction_confidence.png`: Model confidence analysis
4. `doseguard_model.pth`: Saved model weights

## Key Features

### Data Preprocessing
- Image resizing to 224x224
- Normalization using ImageNet statistics
- Data augmentation for training

### Model Training
```python
# Training loop parameters
num_epochs = 10
learning_rate = 0.001
batch_size = 32
```

### Visualization Tools
1. Confusion Matrix:
```python
def plot_confusion_matrix(y_true, y_pred, classes)
```

2. Performance Metrics:
```python
def plot_performance(train_losses, train_accs, test_losses, test_accs)
```

3. Prediction Confidence:
```python
def plot_prediction_confidence(probs, labels, class_names)
```

## Performance Monitoring
- Training/validation loss tracking
- Accuracy metrics
- Per-class performance analysis
- Model confidence assessment

## Implementation Details

### PyTorch Components Used
- `torch.nn.Module`: Base class for neural network
- `torch.utils.data`: Data loading utilities
- `torchvision.models`: Pre-trained models
- `torchvision.transforms`: Image transformations

### Additional Libraries
- `pandas`: Data management
- `numpy`: Numerical operations
- `matplotlib/seaborn`: Visualization
- `scikit-learn`: Metrics and data splitting
- `PIL`: Image processing

## Error Handling
The code includes error handling for:
- CSV file loading
- Image file loading
- Model training
- Data preprocessing