import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# =====================================================
# SETTINGS
# =====================================================
DATA_DIR = Path("dataset")
MODEL_PATH = Path("models/face_cnn.pth")

IMAGE_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 60
PATIENCE = 10
SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# =====================================================
# LOAD DATA
# Folder structure: dataset/category/person_id/images
# =====================================================
class_names = []
records_by_class = {}

print("Scanning dataset...")
for category in ("staff", "students"):
    category_dir = DATA_DIR / category

    if not category_dir.exists():
        print(f"Warning: Missing folder {category_dir}. Skipping.")
        continue

    for person_dir in sorted(category_dir.iterdir()):
        if not person_dir.is_dir():
            continue

        images = sorted(
            p for p in person_dir.iterdir()
            if p.suffix.lower() in EXTENSIONS
        )

        if not images:
            continue

        label = f"{category}_{person_dir.name}"
        class_names.append(label)
        records_by_class[label] = images

if len(class_names) < 2:
    raise RuntimeError(
        "Need at least two identity folders containing images."
    )

train_records, val_records, test_records = [], [], []

print("\nDataset Statistics:")
for label in class_names:
    images = records_by_class[label].copy()
    random.shuffle(images)

    n = len(images)
    
    # Recommendation for small datasets
    if n < 20:
        print(f"WARNING: {label} has only {n} images. "
              "For robust recognition, aim for 50+ varied images per person.")

    n_train = max(1, int(n * 0.70))
    n_val = max(1, int(n * 0.15))

    # Ensure we don't overflow
    if n_train + n_val >= n:
        n_train = n - 2
        n_val = 1
        
    if n_train < 1: n_train = 1
    if n_val < 1: n_val = 1

    label_id = class_names.index(label)

    train_records += [(p, label_id) for p in images[:n_train]]
    val_records += [(p, label_id) for p in images[n_train:n_train + n_val]]
    test_records += [(p, label_id) for p in images[n_train + n_val:]]

    print(f"{label}: Total={n}, Train={n_train}, Val={n_val}, Test={n - n_train - n_val}")

print("\nClasses:", class_names)
print("Device:", DEVICE)


# =====================================================
# PREPROCESSING
# Must match recognize_faces.py exactly
# =====================================================
def preprocess_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(
        gray, (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )
    return gray


# =====================================================
# DATASET
# =====================================================
class FaceDataset(Dataset):
    def __init__(self, records, augment=False):
        self.records = records
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, label = self.records[index]

        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Cannot read: {path}")

        gray = preprocess_face(image)

        if self.augment:
            # Random Horizontal Flip
            if random.random() < 0.5:
                gray = cv2.flip(gray, 1)

            # Random Rotation & Scale
            angle = random.uniform(-10, 10)
            scale = random.uniform(0.9, 1.1)
            matrix = cv2.getRotationMatrix2D(
                (IMAGE_SIZE / 2, IMAGE_SIZE / 2),
                angle,
                scale
            )
            gray = cv2.warpAffine(
                gray, matrix, (IMAGE_SIZE, IMAGE_SIZE),
                borderMode=cv2.BORDER_REFLECT_101
            )

            # Random Brightness/Contrast
            if random.random() < 0.5:
                alpha = random.uniform(0.8, 1.2)
                beta = random.uniform(-20, 20)
                gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

        gray = gray.astype(np.float32) / 255.0
        tensor = torch.from_numpy(gray).unsqueeze(0)

        return tensor, torch.tensor(label, dtype=torch.long)


train_loader = DataLoader(
    FaceDataset(train_records, augment=True),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)

val_loader = DataLoader(
    FaceDataset(val_records),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

test_loader = DataLoader(
    FaceDataset(test_records),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)


# =====================================================
# CNN MODEL
# =====================================================
class FaceCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 192, 3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model = FaceCNN(len(class_names)).to(DEVICE)

# Class Weights for Imbalance
counts = np.bincount(
    [label for _, label in train_records],
    minlength=len(class_names)
)
# Avoid division by zero
weights = len(train_records) / (len(class_names) * np.maximum(counts, 1))

criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(weights, dtype=torch.float32, device=DEVICE)
)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=0.001, weight_decay=0.0001
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)


# =====================================================
# EVALUATION
# =====================================================
def evaluate(loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        return 0.0, 0.0

    return total_loss / total, correct / total


# =====================================================
# TRAIN LOOP
# =====================================================
best_val_loss = float("inf")
bad_epochs = 0

print("\nTraining started...")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    train_loss = total_loss / max(total, 1)
    train_acc = correct / max(total, 1)
    val_loss, val_acc = evaluate(val_loader)

    scheduler.step(val_loss)

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1%} | "
        f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1%}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        bad_epochs = 0

        torch.save({
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "image_size": IMAGE_SIZE,
            "architecture": "GrayscaleFaceCNN_v1"
        }, MODEL_PATH)
        print("  -> Best model saved.")

    else:
        bad_epochs += 1

    if bad_epochs >= PATIENCE:
        print("Early stopping triggered.")
        break


# =====================================================
# FINAL TEST
# =====================================================
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])

test_loss, test_acc = evaluate(test_loader)

print("\n" + "="*40)
print("Training Complete.")
print(f"Test Accuracy: {test_acc:.1%}")
print(f"Model Saved at: {MODEL_PATH.resolve()}")
print("="*40)