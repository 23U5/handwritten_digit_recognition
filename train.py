import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

class CNN(nn.Module):
    def __init__(self, num_classes=36):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Tăng lên 32 bộ lọc
        self.bn1 = nn.BatchNorm2d(32)  # Thêm batch normalization
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Tăng lên 64 bộ lọc
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Thêm tầng tích chập
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)  # Tăng dropout lên 0.5
        self.fc1 = nn.Linear(128 * 3 * 3, 256)  # Điều chỉnh đầu vào (sau 3 lần pooling: 28 -> 14 -> 7 -> 3)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        x = x.view(-1, 128 * 3 * 3)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
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
    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total
    return train_loss, train_accuracy

def evaluate(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss = running_loss / len(data_loader)
    val_accuracy = correct / total
    return val_loss, val_accuracy

def plot_metrics(train_accuracies, val_accuracies, train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_accuracies, label='Train Accuracy', color='blue')
    plt.plot(val_accuracies, label='Validation Accuracy', color='orange')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0.4, 1.0)
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0.0, 1.8)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('pic21.png')
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.RandomRotation(15),  # Tăng góc xoay
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15)),  # Tăng dịch chuyển
        transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),  # Thêm thay đổi tỷ lệ
        transforms.RandomAffine(degrees=0, shear=10),  # Thêm biến dạng
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Tải dữ liệu từ thư mục tùy chỉnh
    full_train_dataset = datasets.ImageFolder(root='./data/dataset', transform=transform)
    test_dataset = datasets.ImageFolder(root='./data/dataset', transform=transform)

    # Tách tập validation (20% tập huấn luyện)
    num_train = len(full_train_dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(0.2 * num_train)
    train_idx, val_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = DataLoader(full_train_dataset, batch_size=64, sampler=train_sampler)
    val_loader = DataLoader(full_train_dataset, batch_size=64, sampler=val_sampler)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = CNN(num_classes=36).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-3)  # Giảm lr, tăng weight decay

    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []

    num_epochs = 100  # Tăng số epoch
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    torch.save(model.state_dict(), './models/cnn_custom.pth')
    plot_metrics(train_accuracies, val_accuracies, train_losses, val_losses)

if __name__ == "__main__":
    main()