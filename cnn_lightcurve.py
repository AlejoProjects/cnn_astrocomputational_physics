import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Import PyTorch libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Import scikit-learn tools for data loading and evaluation
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_curve, auc, ConfusionMatrixDisplay, confusion_matrix
)

# ─── 1) DISCOVER IMAGES & MULTI-CLASS LABELS ──────────────────────────────────
data = load_files(
    container_path='.', 
    categories=['binary_star','normal_star','pulsating_star','exoplanet'],
    load_content=False,
    shuffle=True,
    random_state=413
)
file_paths = np.array(data['filenames'])
orig_targets = data['target']
target_names = data['target_names']

# ─── 2) BUILD BINARY LABELS ───────────────────────────────────────────────────
y = np.array([1 if target_names[t]=='exoplanet' else 0
              for t in orig_targets], dtype=np.float32) # Use float32 for PyTorch loss function

# ─── 3) LOAD & PREPROCESS IMAGES FOR PYTORCH ──────────────────────────────────
images = []
for fp in file_paths:
    img = (Image.open(fp)
           .convert('L')  # grayscale
           .resize((28, 28), Image.Resampling.LANCZOS))
    images.append(np.array(img, dtype=np.float32))

X = np.array(images)
X = X / 255.0  # Scale pixels to [0, 1]

# Reshape for PyTorch: (N, C, H, W) -> (num_samples, channels, height, width)
X = np.expand_dims(X, axis=1) 
print(f"Data shape for PyTorch: {X.shape}")


# ─── 4) TRAIN/TEST SPLIT & CREATE DATALOADERS ─────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=413,
    stratify=y
)

# Convert numpy arrays to PyTorch Tensors
X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train).unsqueeze(1)
X_test_tensor = torch.tensor(X_test)
y_test_tensor = torch.tensor(y_test).unsqueeze(1)

# Create PyTorch Datasets and DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ─── 5) DEFINE THE CNN MODEL IN PYTORCH ───────────────────────────────────────
class LightCurveCNN(nn.Module):
    def __init__(self):
        super(LightCurveCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        # After two pooling layers, 28x28 image becomes 7x7
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 1)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply layers
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

model = LightCurveCNN()
print(model)

# ─── 6) DEFINE LOSS FUNCTION AND OPTIMIZER ────────────────────────────────────
criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ─── 7) TRAIN THE MODEL ───────────────────────────────────────────────────────
epochs = 25
train_losses = []

print("\n--- Starting Model Training ---")
for epoch in range(epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    for inputs, labels in train_loader:
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

print("--- Model Training Finished ---\n")
# Save the trained model's state
torch.save(model.state_dict(), 'exoplanet_cnn_model.pth')
print("Model saved to exoplanet_cnn_model.pth")
# ─── 8) PLOT TRAINING LOSS CURVE ─────────────────────────────────────────────
plt.figure(figsize=(6, 4))
plt.plot(train_losses, label='Training loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('CNN Training Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_loss_pytorch.png")
plt.show()


# ─── 9) EVALUATE ON TEST SET ─────────────────────────────────────────────────
model.eval()  # Set model to evaluation mode
all_preds = []
all_scores = []
with torch.no_grad():  # No need to calculate gradients during evaluation
    for inputs, _ in test_loader:
        outputs = model(inputs)
        all_scores.extend(outputs.numpy().flatten())
        preds = torch.round(outputs)
        all_preds.extend(preds.numpy().flatten())

y_pred = np.array(all_preds)
y_scores = np.array(all_scores)
y_test_numpy = y_test_tensor.numpy().flatten()

acc = accuracy_score(y_test_numpy, y_pred)
print(f"Test Accuracy: {acc:.3%}")


# ─── 10) ROC CURVE ───────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test_numpy, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (PyTorch): Exoplanet vs. Not")
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve_pytorch.png")
plt.show()

# ─── 11) CONFUSION MATRIX ───────────────────────────────────────────────────
cm = confusion_matrix(y_test_numpy, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["not exoplanet", "exoplanet"]
)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (PyTorch)")
plt.tight_layout()
plt.savefig("confusion_matrix_pytorch.png")
plt.show()
print(f"Test Accuracy: {acc:.3%}")
