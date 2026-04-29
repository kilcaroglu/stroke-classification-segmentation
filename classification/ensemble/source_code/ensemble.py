# IMPORT REQUIRED LIBRARIES
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# GLOBAL CONFIGURATION
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 3
BATCH_SIZE = 16
SAVE_DIR = "/outputs/ensemble"
DATA_DIR = "data/test"
DISPLAY_CLASSES = ["Hemorrhagic", "Ischemic", "No Stroke"]

os.makedirs(SAVE_DIR, exist_ok=True)

# IMAGE TRANSFORMS
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

# Transforms according to different input sizes
transform_224 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

transform_299 = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

# Datasets according to different input sizes
dataset_224 = datasets.ImageFolder(DATA_DIR, transform=transform_224)
dataset_299 = datasets.ImageFolder(DATA_DIR, transform=transform_299)

# Loader according to different input sizes
val_loader_224 = DataLoader(dataset_224, batch_size=BATCH_SIZE, shuffle=False)
val_loader_299 = DataLoader(dataset_299, batch_size=BATCH_SIZE, shuffle=False)

# MODEL DEFINITION
models_list = []

# ResNet18
model1 = models.resnet18(weights="IMAGENET1K_V1")
model1.fc = nn.Linear(model1.fc.in_features, NUM_CLASSES)
model1.load_state_dict(torch.load("/base_models/resnet18/model_weights/fold5_best_f1_macro_model_weights.pth", map_location=DEVICE))
model1.to(DEVICE).eval()
models_list.append((model1, val_loader_224))

# DenseNet121
model2 = models.densenet121(weights="IMAGENET1K_V1")
model2.classifier = nn.Linear(model2.classifier.in_features, NUM_CLASSES)
model2.load_state_dict(torch.load("/base_models/densenet121/model_weights/fold3_best_f1_macro_model_weights.pth", map_location=DEVICE))
model2.to(DEVICE).eval()
models_list.append((model2, val_loader_224))

# EfficientNet-B0
model3 = models.efficientnet_b0(weights="IMAGENET1K_V1")
model3.classifier[1] = nn.Linear(model3.classifier[1].in_features, NUM_CLASSES)
model3.load_state_dict(torch.load("/base_models/efficientnet_b0/model_weights/fold3_best_f1_macro_model_weights.pth", map_location=DEVICE))
model3.to(DEVICE).eval()
models_list.append((model3, val_loader_224))

# Inception-V3
model4 = models.inception_v3(weights="IMAGENET1K_V1", aux_logits=True)
model4.aux_logits = False
model4.AuxLogits = None
model4.fc = nn.Linear(model4.fc.in_features, NUM_CLASSES)
model4.load_state_dict(torch.load("/base_models/inceptionv3/model_weights/fold1_best_f1_macro_model_weights.pth", map_location=DEVICE))
model4.to(DEVICE).eval()
models_list.append((model4, val_loader_299))

# MobileNetV3-Large
model5 = models.mobilenet_v3_large(weights="IMAGENET1K_V1")
model5.classifier[3] = nn.Linear(model5.classifier[3].in_features, NUM_CLASSES)
model5.load_state_dict(torch.load("/base_models/mobilenetv3_large/model_weights/fold4_best_f1_macro_model_weights.pth", map_location=DEVICE))
model5.to(DEVICE).eval()
models_list.append((model5, val_loader_224))


# LOGIT-LEVEL ENSEMBLE
all_preds = []
all_targets = []

# Select loader by model
loaders = [item[1] for item in models_list]
models = [item[0] for item in models_list]

with torch.no_grad():
    for batches in zip(*loaders):
        logits_sum = 0
        targets_batch = None

        for i, (model, (images, targets)) in enumerate(zip(models, batches)):
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            if i == 0:
                targets_batch = targets

            logits = model(images)
            logits_sum += logits

        ensemble_logits = logits_sum / len(models)
        preds = torch.argmax(ensemble_logits, dim=1)

        all_preds.append(preds.cpu())
        all_targets.append(targets_batch.cpu())

all_preds_lle = torch.cat(all_preds).numpy()
all_targets_lle = torch.cat(all_targets).numpy()

# SAVING AND PRINTING RESULTS
# Classification Report
print("\nLOGIT AVERAGING ENSEMBLE CLASSIFICATION REPORT\n")
print(classification_report(all_targets_lle, all_preds_lle, target_names=DISPLAY_CLASSES, digits=4))

cm = confusion_matrix(all_targets_lle, all_preds_lle)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=DISPLAY_CLASSES, yticklabels=DISPLAY_CLASSES)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Logit Averaging Confusion Matrix")
plt.savefig(f"{SAVE_DIR}/logit_level_ensemble_confusion_matrix.png", dpi=300)
plt.show()
plt.close()

# Confusion Matrix
np.savetxt(f"{SAVE_DIR}/logit_level_ensemble_confusion_matrix.txt", cm, fmt="%d")
pd.DataFrame(cm, index=DISPLAY_CLASSES, columns=DISPLAY_CLASSES).to_csv(
    f"{SAVE_DIR}/logit_level_ensemble_confusion_matrix.csv"
)

with open(f"{SAVE_DIR}/logit_level_ensemble_classification_report.txt", "w") as f:
    f.write(classification_report(all_targets_lle, all_preds_lle, target_names=DISPLAY_CLASSES, digits=4))

pd.DataFrame(classification_report(all_targets_lle, all_preds_lle, target_names=DISPLAY_CLASSES, output_dict=True)).transpose().to_csv(
    f"{SAVE_DIR}/logit_level_ensemble_classification_report.csv", float_format="%.4f"
)