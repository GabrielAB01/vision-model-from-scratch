# Simple MLP with torch

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# Plot data distribution
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', edgecolor='k', s=20)
plt.title('Training Data Distribution')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.subplot(1, 2, 2)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', edgecolor='k', s=20)
plt.title('Test Data Distribution')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.tight_layout()
plt.show()

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the model
class SimpleMLP(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(SimpleMLP, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.fc2 = nn.Linear(hidden_size, output_size)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x
	
# Hyperparameters
input_size = X_train.shape[1]
hidden_size = 64
output_size = 2
num_epochs = 20
learning_rate = 0.001

# Initialize the model, loss function and optimizer
model = SimpleMLP(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Print model summary
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("CUDA current device:", torch.cuda.current_device())
print("CUDA device name:", torch.cuda.get_device_name(0), "\n\n")
summary(model, input_size=(32, input_size), device="cpu")
print("\n\n")

# Training loop
train_losses = []
train_accuracies = []
for epoch in range(num_epochs):
	model.train()
	epoch_loss = 0
	epoch_accuracy = 0
	for batch_X, batch_y in train_loader:
		optimizer.zero_grad()
		output = model(batch_X)
		loss = criterion(output, batch_y)
		loss.backward()
		optimizer.step()
		
		epoch_loss += loss.item()
		predicted = torch.argmax(output, dim=1)
		epoch_accuracy += (predicted == batch_y).float().mean().item()

	train_losses.append(epoch_loss / len(train_loader))
	train_accuracies.append(epoch_accuracy / len(train_loader))
	print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_losses[-1]:.4f}, Accuracy: {train_accuracies[-1]:.4f}')
	
# Plot training loss and accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss vs Epoch')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.xlabel('Epoch')	
plt.ylabel('Accuracy')
plt.title('Training Accuracy vs Epoch')
plt.legend()
plt.show()

# Evaluation
model.eval()
y_pred = []
y_true = []
with torch.no_grad():
	for batch_X, batch_y in test_loader:
		output = model(batch_X)
		predicted = torch.argmax(output, dim=1)
		y_pred.extend(predicted.numpy())
		y_true.extend(batch_y.numpy())
y_pred = np.array(y_pred)
y_true = np.array(y_true)
print("Accuracy:", accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred))
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(np.unique(y_true)))
plt.xticks(tick_marks, np.unique(y_true))
plt.yticks(tick_marks, np.unique(y_true))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
