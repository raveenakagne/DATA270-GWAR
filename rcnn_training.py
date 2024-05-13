import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import pandas as pd

# Configuration and Constants
data_file = "data.csv"
image_dir = "processed_images"
image_size = 224
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
output_size = 6
batch_size = 512
device = "cpu" 

# Data setup
df = pd.read_csv(data_file)
df = df[['filename', 'quantity']]
train_val, test = train_test_split(df, test_size=0.1, random_state=42)
train, val = train_test_split(train_val, test_size=0.11, random_state=42)

image_transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Model setup
def get_faster_rcnn_backbone():
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    model.to(device)
    return model

# Load and prepare images
def prepare_data(df, image_dir, transforms):
    x, y = [], []
    for _, row in df.iterrows():
        image_path = os.path.join(image_dir, row['filename'])
        image = Image.open(image_path).convert('RGB')
        image = transforms(image)
        x.append(image)
        y.append(row['quantity'])
    x = torch.stack(x).to(device)
    y = torch.tensor(y, dtype=torch.long).to(device)
    return x, y

# Neural network for classification
class CountingNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(CountingNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.fc(x)

# Extract embeddings using the Faster R-CNN backbone
def extract_embeddings(model, x):
    embeddings = []
    for image in x:
        with torch.no_grad():
            features = model.backbone(image.unsqueeze(0))  # Unsqueeze to add batch dimension
            embedding = torch.flatten(features['0'], start_dim=1)
            embeddings.append(embedding)
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings

# Training and evaluation
def train_and_evaluate(train_loader, val_loader, output_size, device):
    model = CountingNetwork(train_loader.dataset.tensors[0].size(1), output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_val_acc = 25 

    for epoch in range(10):  # Adjust epochs as necessary
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            print(loss)
            loss.backward()
            optimizer.step()

        model.eval()
        val_accuracies = []
        for val_inputs, val_labels in val_loader:
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
            val_outputs = model(val_inputs)
            val_accuracy = calculate_accuracy(val_outputs, val_labels)
            val_accuracies.append(val_accuracy)
            
        avg_val_accuracy = sum(val_accuracies) / len(val_accuracies)
        if avg_val_accuracy > best_val_acc:
            best_val_acc = avg_val_accuracy
            torch.save(model.state_dict(), 'best_rcnn_snapshot_model.pth')
        print(f'Epoch {epoch+1}, Validation Accuracy: {avg_val_accuracy}%')
    return model, best_val_acc 

# Function to calculate accuracy
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return 100 * correct / total

if __name__ == '__main__':
    train_x, train_y = prepare_data(train, image_dir, image_transforms)
    val_x, val_y = prepare_data(val, image_dir, image_transforms)
    test_x, test_y = prepare_data(test, image_dir, image_transforms)
    
    faster_rcnn_model = get_faster_rcnn_backbone()
    train_embeddings = extract_embeddings(faster_rcnn_model, train_x)
    val_embeddings = extract_embeddings(faster_rcnn_model, val_x)
    test_embeddings = extract_embeddings(faster_rcnn_model, test_x)
    
    train_dataset = TensorDataset(train_embeddings, train_y)
    val_dataset = TensorDataset(val_embeddings, val_y)
    test_dataset = TensorDataset(test_embeddings, test_y)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model, best_val_acc = train_and_evaluate(train_loader, val_loader, output_size, device)

    print(f"Top Validation Accuracy: {best_val_acc}%")
