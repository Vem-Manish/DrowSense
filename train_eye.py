import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

# --- CONFIG ---
DATASET_PATH = "dataset_new"
TRAIN_DIR = os.path.join(DATASET_PATH, "train")
TEST_DIR = os.path.join(DATASET_PATH, "test")

IMG_SIZE = 224
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- FILTER FUNCTION (IMPORTANT) ---
def filter_eye_classes(dataset):
    eye_classes = ['Closed', 'Open']

    class_to_idx = dataset.class_to_idx
    valid_class_indices = [class_to_idx[c] for c in eye_classes]

    indices = [i for i, (_, label) in enumerate(dataset.samples) if label in valid_class_indices]

    subset = Subset(dataset, indices)

    # Remap labels: Closed->0, Open->1
    subset.targets = [0 if dataset.samples[i][1] == class_to_idx['Closed'] else 1 for i in indices]

    return subset


# --- TRAIN FUNCTION ---
def train_mobilenet_v3():
    print(f"[INFO] Using device: {DEVICE}")

    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    # --- LOAD FULL DATASET ---
    train_dataset_full = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
    test_dataset_full = datasets.ImageFolder(TEST_DIR, transform=val_transforms)

    # --- FILTER ONLY EYES ---
    train_dataset = filter_eye_classes(train_dataset_full)
    test_dataset = filter_eye_classes(test_dataset_full)

    class_names = ['Closed', 'Open']
    num_classes = 2

    print("[INFO] Using ONLY eye classes:", class_names)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- MODEL ---
    weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
    model = models.mobilenet_v3_large(weights=weights)

    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_precision': [], 'val_recall': []
    }

    best_acc = 0.0

    # --- TRAIN LOOP ---
    for epoch in range(EPOCHS):
        model.train()
        train_correct = 0
        train_total = 0
        train_loss_sum = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for inputs, labels in loop:
            inputs = inputs.to(DEVICE)
            labels = torch.tensor(labels).to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)

            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            train_loss_sum += loss.item() * inputs.size(0)

            loop.set_postfix(loss=loss.item(), acc=train_correct / train_total)

        history['train_loss'].append(train_loss_sum / train_total)
        history['train_acc'].append(train_correct / train_total)

        # --- VALIDATION ---
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_sum = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(DEVICE)
                labels = torch.tensor(labels).to(DEVICE)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                val_loss_sum += loss.item() * inputs.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = val_correct / val_total
        val_loss = val_loss_sum / val_total

        val_prec = precision_score(all_labels, all_preds)
        val_rec = recall_score(all_labels, all_preds)

        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_prec)
        history['val_recall'].append(val_rec)

        print(f"Val Acc: {val_acc:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "eye_model.pth")
            print("[Saved Best Eye Model]")

    # --- PLOTS ---
    epochs_range = range(1, EPOCHS + 1)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, history['train_loss'])
    plt.plot(epochs_range, history['val_loss'])
    plt.title("Loss")

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, history['train_acc'])
    plt.plot(epochs_range, history['val_acc'])
    plt.title("Accuracy")

    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, history['val_precision'])
    plt.plot(epochs_range, history['val_recall'])
    plt.title("Precision & Recall")

    plt.tight_layout()
    plt.show()

    # --- CONFUSION MATRIX ---
    cm = confusion_matrix(all_labels, all_preds)

    plt.imshow(cm)
    plt.title("Eye Confusion Matrix")
    plt.colorbar()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center')

    plt.xticks([0, 1], class_names)
    plt.yticks([0, 1], class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.show()


if __name__ == "__main__":
    train_mobilenet_v3()
