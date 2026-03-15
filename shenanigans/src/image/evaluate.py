import json
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

from .config import DEVICE, MODEL_DIR, RESULTS_DIR
from .dataset import get_dataloaders
from .model import get_model

def evaluate():
    train_dataset, test_dataset, train_loader, test_loader = get_dataloaders()

    model = get_model()
    model.load_state_dict(torch.load(MODEL_DIR / "resnet18_cifake.pth", map_location=DEVICE))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)

    print("Test Accuracy:", acc)
    print("Test F1:", f1)
    print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

    cm = confusion_matrix(all_labels, all_preds)

    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    plt.savefig(RESULTS_DIR / "figures" / "image_confusion_matrix.png", bbox_inches="tight")
    plt.show()

    metrics = {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "classes": train_dataset.classes
    }

    with open(RESULTS_DIR / "metrics" / "image_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    evaluate()