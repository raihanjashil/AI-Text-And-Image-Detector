from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from .config import TRAIN_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE, NUM_WORKERS

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    return train_transform, test_transform

def get_dataloaders():
    train_transform, test_transform = get_transforms()

    train_dataset = datasets.ImageFolder(
        root=str(TRAIN_DIR),
        transform=train_transform
    )

    test_dataset = datasets.ImageFolder(
        root=str(TEST_DIR),
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return train_dataset, test_dataset, train_loader, test_loader