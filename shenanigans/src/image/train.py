import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from .config import DEVICE, EPOCHS, LR, MODEL_DIR, SEED
from .dataset import get_dataloaders
from .model import get_model
from .utils import set_seed

def train():
    set_seed(SEED)

    train_dataset, test_dataset, train_loader, test_loader = get_dataloaders()

    print("Using device:", DEVICE)
    print("Classes:", train_dataset.classes)
    print("Train size:", len(train_dataset))
    print("Test size:", len(test_dataset))

    model = get_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"Loss: {running_loss:.4f}")
        print(f"Training Accuracy: {accuracy:.4f}")

    save_path = MODEL_DIR / "resnet18_cifake.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")

if __name__ == "__main__":
    train()