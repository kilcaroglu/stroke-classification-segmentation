# IMPORT REQUIRED LIBRARIES
import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import random



# SET SEED FOR REPRODUCIBILITY
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



# GLOBAL CONFIGURATION
DATA_DIR = "data/train_val"            # Dataset root directory
NUM_CLASSES = 3                        # Number of output classes
EPOCHS = 50                            # Number of training epochs
BATCH_SIZE = 32                        # Batch size
LR = 1e-4                              # Learning rate
PATIENCE = 15                          # Early stopping patience
K_FOLD = 5                             # Number of CV folds
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "/outputs/mobilenetv3_large"

print(DEVICE)
os.makedirs(SAVE_DIR, exist_ok=True)



# IMAGE TRANSFORMS
# ImageNet normalization values
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]
# Training-time data augmentation
train_transform = transforms.Compose([
    transforms.RandomApply([transforms.RandomRotation(degrees=10)], p=0.5),
    transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.05, 0.05))], p=0.5),
    transforms.RandomApply([transforms.RandomAffine(degrees=0, scale=(0.95, 1.05))], p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])
# Validation transform (no augmentation)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])



# DATASET INITIALIZATION
# Base dataset used only for labels (stratified split)
base_dataset = datasets.ImageFolder(DATA_DIR)
targets = np.array(base_dataset.targets)
# Separate datasets with different transforms
train_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
val_dataset   = datasets.ImageFolder(DATA_DIR, transform=test_transform)



# STRATIFIED K-FOLD SETUP
skf = StratifiedKFold(n_splits=K_FOLD, shuffle=True, random_state=SEED)

# Containers for cross-validation results
cv_f1_macro = []
cv_f1_weighted = []
cv_acc = []



# CROSS-VALIDATION LOOP
for fold, (train_idx, val_idx) in enumerate(skf.split(base_dataset, targets)):
    print(f"\n{'='*20} FOLD {fold+1}/{K_FOLD} {'='*20}")

    os.makedirs(f"{SAVE_DIR}/fold{fold+1}", exist_ok=True)

    # DATA SPLIT
    train_ds = Subset(train_dataset, train_idx)
    val_ds   = Subset(val_dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # CLASS WEIGHT COMPUTATION
    # Used to address class imbalance
    train_targets = targets[train_idx]
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_targets),
        y=train_targets
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

    # MODEL DEFINITION
    # Pretrained MobileNetV3_Large with modified classification head
    model = models.mobilenet_v3_large(weights="IMAGENET1K_V1")
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)
    model.to(DEVICE)


    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # MODEL SELECTION TRACKERS
    best_f1_macro = 0
    best_f1_weighted = 0
    best_acc = 0
    best_f1_macro_wts = copy.deepcopy(model.state_dict())
    best_f1_weighted_wts = copy.deepcopy(model.state_dict())
    best_acc_wts = copy.deepcopy(model.state_dict())
    early_stop_counter = 0
    # Training history for visualization
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [],  "val_acc": [],
        "train_f1_macro": [],   "val_f1_macro": [],
        "train_f1_weighted": [],   "val_f1_weighted": []
    }


    # TRAINING LOOP
    for epoch in range(EPOCHS):
        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

        # TRAIN PHASE
        model.train()
        train_loss = 0
        y_true_train, y_pred_train = [], []

        for imgs, labels in tqdm(train_loader, leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, 1)
            train_loss += loss.item() * imgs.size(0)
            y_true_train.extend(labels.cpu().numpy())
            y_pred_train.extend(preds.cpu().numpy())

        train_loss /= len(train_loader.dataset)
        train_acc = accuracy_score(y_true_train, y_pred_train)
        train_f1_macro  = f1_score(y_true_train, y_pred_train, average="macro")
        train_f1_weighted  = f1_score(y_true_train, y_pred_train, average="weighted")

        # VALIDATION PHASE
        model.eval()
        val_loss = 0
        y_true_val, y_pred_val = [], []

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, leave=False):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                preds = torch.argmax(outputs, 1)
                val_loss += loss.item() * imgs.size(0)
                y_true_val.extend(labels.cpu().numpy())
                y_pred_val.extend(preds.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(y_true_val, y_pred_val)
        val_f1_macro  = f1_score(y_true_val, y_pred_val, average="macro")
        val_f1_weighted  = f1_score(y_true_val, y_pred_val, average="weighted")

        scheduler.step(val_loss)
        # Save metrics
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_f1_macro"].append(train_f1_macro)
        history["val_f1_macro"].append(val_f1_macro)
        history["train_f1_weighted"].append(train_f1_weighted)
        history["val_f1_weighted"].append(val_f1_weighted)

        print(f"TRAIN | Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1_macro:.4f}")
        print(f"VAL   | Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1_macro:.4f}")

        # MODEL SELECTION
        # Update best model based on macro F1 score
        if val_f1_macro > best_f1_macro:
            best_f1_macro = val_f1_macro
            print("Best macro f1")
            best_f1_macro_wts = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # Update best model based on weighted F1 score
        if val_f1_weighted > best_f1_weighted:
            best_f1_weighted = val_f1_weighted
            print("Best weighted f1")
            best_f1_weighted_wts = copy.deepcopy(model.state_dict())

        # Update best model based on accuracy score
        if val_acc > best_acc:
            best_acc = val_acc
            print("Best acc")
            best_acc_wts = copy.deepcopy(model.state_dict())


        # EARLY STOPPING
        if early_stop_counter >= PATIENCE:
            print("Early stopping!")
            break

    # SAVE BEST MODELS
    # Save best model weights based on macro F1 score
    torch.save(best_f1_macro_wts, f"{SAVE_DIR}/fold{fold+1}/fold{fold+1}_best_f1_macro_model_weights.pth")
    # Save best model weights based on weighted F1 score
    torch.save(best_f1_weighted_wts, f"{SAVE_DIR}/fold{fold+1}/fold{fold+1}_best_f1_weighted_model_weights.pth")
    # Save best model weights based on accuracy score
    torch.save(best_acc_wts, f"{SAVE_DIR}/fold{fold+1}/fold{fold+1}_best_acc_model_weights.pth")

    # Save best model based on macro F1 score
    best_f1_macro_model_copy = copy.deepcopy(model)
    best_f1_macro_model_copy.load_state_dict(best_f1_macro_wts)
    torch.save(best_f1_macro_model_copy, f"{SAVE_DIR}/fold{fold+1}/fold{fold+1}_best_f1_macro_full_model.pth")

    # Save best model weights based on weighted F1 score
    best_f1_weighted_model_copy = copy.deepcopy(model)
    best_f1_weighted_model_copy.load_state_dict(best_f1_weighted_wts)
    torch.save(best_f1_weighted_model_copy, f"{SAVE_DIR}/fold{fold+1}/fold{fold+1}_best_f1_weighted_full_model.pth")

    # Save best model weights based on accuracy score
    best_acc_model_copy = copy.deepcopy(model)
    best_acc_model_copy.load_state_dict(best_acc_wts)
    torch.save(best_acc_model_copy, f"{SAVE_DIR}/fold{fold+1}/fold{fold+1}_best_acc_full_model.pth")


    # FINAL EVALUATION USING THE MODEL SELECTED BASED ON MACRO F1 SCORE (PER FOLD)
    model.load_state_dict(best_f1_macro_wts)
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            preds = torch.argmax(outputs, 1)

            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())


    # Class names
    display_classes = ["Hemorrhagic", "Ischemic", "No Stroke"]

    # Compute CV metrics
    val_acc = accuracy_score(y_true, y_pred)
    val_f1_macro  = f1_score(y_true, y_pred, average="macro")
    val_f1_weighted  = f1_score(y_true, y_pred, average="weighted")

    # Store CV metrics
    cv_f1_macro.append(val_f1_macro)
    cv_f1_weighted.append(val_f1_weighted)
    cv_acc.append(val_acc)

    # Save classification report in TXT format
    report = classification_report(y_true, y_pred, target_names=display_classes, digits=4)
    with open(f"{SAVE_DIR}/fold{fold+1}/fold{fold+1}_classification_report.txt", "w") as f:
        f.write(report)

    # Save classification report in CSV format
    report_dict = classification_report(y_true, y_pred, target_names=display_classes, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    df_report.to_csv(f"{SAVE_DIR}/fold{fold+1}/fold{fold+1}_classification_report.csv", float_format='%.4f')


    # Save confusion matrix in TXT format
    cm = confusion_matrix(y_true, y_pred)
    np.savetxt(f"{SAVE_DIR}/fold{fold+1}/fold{fold+1}_confusion_matrix.txt", cm, fmt="%d")

    # Save confusion matrix in CSV format
    df_cm = pd.DataFrame(cm, index=display_classes, columns=display_classes)
    df_cm.to_csv(f"{SAVE_DIR}/fold{fold+1}/fold{fold+1}_confusion_matrix.csv")

    # Plot and save confusion matrix in PNG format
    plt.figure(figsize=(5, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=display_classes,
        yticklabels=display_classes,
    )

    plt.title(f"Fold {fold+1} Confusion Matrix", fontsize=13)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.savefig(f"{SAVE_DIR}/fold{fold+1}/fold{fold+1}_confusion_matrix.png", dpi=300)
    plt.close()


    # TRAINING CURVES
    for metric in [["loss", "Loss"], ["acc", "Accuracy"], ["f1_macro", "F1 (Macro)"], ["f1_weighted", "F1 (Weighted)"]]:
        plt.figure()
        plt.plot(history[f"train_{metric[0]}"], label="train")
        plt.plot(history[f"val_{metric[0]}"], label="val")
        plt.legend()
        plt.title(f"Fold {fold+1} {metric[1]}")
        plt.savefig(f"{SAVE_DIR}/fold{fold+1}/fold{fold+1}_{metric[1]}.png")
        plt.close()



# CROSS-VALIDATION SUMMARY
cv_f1_macro = np.array(cv_f1_macro)
cv_f1_weighted = np.array(cv_f1_weighted)
cv_acc = np.array(cv_acc)

print("\n5-FOLD CV RESULTS")
print(f"Accuracy : {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
print(f"Macro F1-score : {cv_f1_macro.mean():.4f} ± {cv_f1_macro.std():.4f}")
print(f"Weighted F1-score : {cv_f1_weighted.mean():.4f} ± {cv_f1_weighted.std():.4f}")

# Save summary in TXT format
with open(f"{SAVE_DIR}/cv_summary.txt", "w") as f:
    f.write("5-Fold Cross Validation Summary\n")
    f.write(f"Accuracy : {cv_acc.mean():.4f} ± {cv_acc.std():.4f}\n")
    f.write(f"Macro F1-score : {cv_f1_macro.mean():.4f} ± {cv_f1_macro.std():.4f}\n")
    f.write(f"Weighted F1-score : {cv_f1_weighted.mean():.4f} ± {cv_f1_weighted.std():.4f}\n")


print("\nAll fold trainings completed successfully.")