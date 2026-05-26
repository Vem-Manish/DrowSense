import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import os
from tqdm import tqdm

# --- NEW IMPORTS FOR PLOTTING & METRICS ---
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

# --- CONFIGURATION ---
DATASET_PATH = "yawning"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_yawn_mouth():
    print(f"[INFO] Training Advanced Yawn Detector on {DEVICE}...")

    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] Dataset folder '{DATASET_PATH}' not found!")
        return

    # --- 1. AUGMENTATION ---
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # --- 2. LOAD & SPLIT DATA ---
    full_dataset = datasets.ImageFolder(DATASET_PATH, transform=train_transforms)
    class_names = full_dataset.classes  # Get class names for Confusion Matrix
    print(f"[INFO] Classes Found: {full_dataset.class_to_idx}")

    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- 3. CLASS WEIGHTING ---
    try:
        yawn_path = os.path.join(DATASET_PATH, "yawn")
        noyawn_path = os.path.join(DATASET_PATH, "no yawn")
        if not os.path.exists(noyawn_path): noyawn_path = os.path.join(DATASET_PATH, "no_yawn")

        n_yawn = len(os.listdir(yawn_path))
        n_noyawn = len(os.listdir(noyawn_path))

        pos_weight_val = n_noyawn / max(n_yawn, 1)
        pos_weight = torch.tensor([pos_weight_val]).to(DEVICE)
        print(f"[INFO] Weighting: {n_noyawn} Non-Yawns vs {n_yawn} Yawns. Pos Weight: {pos_weight_val:.2f}")
    except Exception as e:
        print(f"[WARN] Could not calculate auto-weights: {e}")
        pos_weight = torch.tensor([1.0]).to(DEVICE)

    # --- 4. MODEL SETUP ---
    weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
    model = models.mobilenet_v3_large(weights=weights)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.features[-3:].parameters():
        param.requires_grad = True

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(128, 1)
    )

    model = model.to(DEVICE)

    # --- 5. OPTIMIZER & LOSS ---
    optimizer = optim.Adam([
        {'params': model.features.parameters(), 'lr': 1e-5},
        {'params': model.classifier.parameters(), 'lr': 1e-4}
    ])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- HISTORY LISTS ---
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_precision': [], 'val_recall': []
    }

    # --- 6. TRAINING LOOP ---
    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        train_correct = 0
        train_total = 0
        epoch_train_loss = 0.0

        for inputs, labels in loop:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = torch.sigmoid(outputs) > 0.5
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            epoch_train_loss += loss.item() * inputs.size(0)

            loop.set_postfix(loss=loss.item(), acc=train_correct / train_total)

        # Store Train Metrics
        history['train_loss'].append(epoch_train_loss / train_total)
        history['train_acc'].append(train_correct / train_total)

        # --- VALIDATION ---
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_sum = 0

        # We collect all labels/preds for Precision/Recall/Confusion Matrix
        all_val_labels = []
        all_val_preds = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                preds = torch.sigmoid(outputs) > 0.5

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                val_loss_sum += loss.item() * inputs.size(0)

                # Move to CPU for metrics
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(preds.cpu().numpy())

        if val_total > 0:
            val_acc = val_correct / val_total
            val_loss = val_loss_sum / val_total

            # Calculate Precision & Recall
            val_prec = precision_score(all_val_labels, all_val_preds, zero_division=0)
            val_rec = recall_score(all_val_labels, all_val_preds, zero_division=0)

            # Store Valid Metrics
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_precision'].append(val_prec)
            history['val_recall'].append(val_rec)

            print(f"--> Val Acc: {val_acc:.4f} | Prec: {val_prec:.4f} | Recall: {val_rec:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                # --- UPDATED SAVE PATH ---
                torch.save(model.state_dict(), "yawn_model_2.pth")
                print("    [Saved New Best Model]")

    print("[INFO] Done. 'yawn_model_2.pth' is ready.")

    # --- 7. VISUALIZATION (GRAPHS & CONFUSION MATRIX) ---
    print("[INFO] Generating plots...")

    epochs_range = range(1, EPOCHS + 1)

    plt.figure(figsize=(15, 5))

    # Subplot 1: Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss')
    plt.plot(epochs_range, history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Subplot 2: Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, history['train_acc'], label='Train Acc')
    plt.plot(epochs_range, history['val_acc'], label='Val Acc')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Subplot 3: Precision & Recall (Val Only)
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, history['val_precision'], label='Precision', linestyle='--')
    plt.plot(epochs_range, history['val_recall'], label='Recall', linestyle=':')
    plt.title('Validation Precision & Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_graphs_yawn.png')
    plt.show()

    # --- CONFUSION MATRIX (Final Epoch) ---
    all_val_preds_int = [int(p) for p in all_val_preds]
    all_val_labels_int = [int(l) for l in all_val_labels]

    cm = confusion_matrix(all_val_labels_int, all_val_preds_int)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    plt.figure(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('Confusion Matrix (Final Epoch)')
    plt.savefig('confusion_matrix_yawn.png')
    plt.show()


if __name__ == "__main__":
    train_yawn_mouth()