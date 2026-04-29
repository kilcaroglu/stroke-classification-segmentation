# IMPORT REQUIRED LIBRARIES
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex, BinaryPrecision, BinaryRecall


# SET SEED FOR REPRODUCIBILITY
SEED = 42
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"

# SAVE DIRECTORY
SAVE_DIR = "/results/unet_densenet121/"
os.makedirs(SAVE_DIR, exist_ok=True)

# DATA DIRECTORY
IMG_DIR = "data/train_val/images"
MASK_DIR = "data/train_val/masks"

# DATA SPLIT
files = sorted(os.listdir(IMG_DIR))
random.Random(SEED).shuffle(files)

split_idx = int(len(files)*0.8)
train_files = files[:split_idx]
val_files   = files[split_idx:]


# DATASET DEFINITION
class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, file_list, is_train=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.files = file_list
        self.is_train = is_train

        # Base transforms
        self.resize = transforms.Resize((512, 512))
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.files)

    def apply_augmentations(self, image, mask):
        # Random horizontal flip
        if random.random() < 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)

        # Random rotation
        if random.random() < 0.5:
            angle = random.uniform(-10, 10)
            image = transforms.functional.rotate(image, angle)
            mask = transforms.functional.rotate(mask, angle)

        # Random translation
        if random.random() < 0.5:
            tx = random.uniform(-0.05, 0.05) * image.size[0]
            ty = random.uniform(-0.05, 0.05) * image.size[1]
            image = transforms.functional.affine(image, angle=0, translate=[tx, ty], scale=1.0, shear=0)
            mask = transforms.functional.affine(mask, angle=0, translate=[tx, ty], scale=1.0, shear=0)

        # Random scale
        if random.random() < 0.5:
            scale = random.uniform(0.95, 1.05)
            image = transforms.functional.affine(image, angle=0, translate=[0, 0], scale=scale, shear=0)
            mask = transforms.functional.affine(mask, angle=0, translate=[0, 0], scale=scale, shear=0)

        return image, mask


    def __getitem__(self, idx):
        fname = self.files[idx]

        # Load images
        image = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, fname))

        # Resize
        image = self.resize(image)
        mask = self.resize(mask)

        # Apply augmentations only for training
        if self.is_train:
            image, mask = self.apply_augmentations(image, mask)

        # Convert to tensor and normalize
        image = self.to_tensor(image)
        image = self.normalize(image)

        mask = self.to_tensor(mask)
        mask = (mask > 0).float()

        return image, mask

# Create datasets
train_dataset = SegDataset(IMG_DIR, MASK_DIR, train_files, is_train=True)
val_dataset   = SegDataset(IMG_DIR, MASK_DIR, val_files, is_train=False)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=4, shuffle=False)


# MODEL DEFINITION
# UNET with Densenet-121 backbone
model = smp.Unet(encoder_name="densenet121",
                 encoder_weights="imagenet",
                 in_channels=3,
                 classes=1,
                 activation=None).to(device)

# DEFINE LOSS AND OPTIMIZER
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# DEFINE METRICS
dice_metric = BinaryF1Score().to(device)
iou_metric  = BinaryJaccardIndex().to(device)


# MODEL SELECTION TRACKERS
EPOCHS = 50
best_dice = 0
patience = 15
early_stop_counter = 0
# Training history for visualization
history = {k:[] for k in
["train_loss","val_loss","train_dice","val_dice","train_iou","val_iou"]}

# TRAINING LOOP
for epoch in range(EPOCHS):
    # TRAIN PHASE
    model.train()
    dice_metric.reset()
    iou_metric.reset()
    train_loss = 0

    for x, y in tqdm(train_loader, leave=False):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        preds = (torch.sigmoid(logits) > 0.5).float()

        train_loss += loss.item()
        dice_metric.update(preds, y)
        iou_metric.update(preds, y)

    tl = train_loss / len(train_loader)
    td = dice_metric.compute().item()
    ti = iou_metric.compute().item()

    # VALIDATION PHASE
    model.eval()
    dice_metric.reset()
    iou_metric.reset()
    val_loss = 0

    with torch.no_grad():
        for x, y in tqdm(val_loader, leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            preds = (torch.sigmoid(logits) > 0.5).float()

            val_loss += loss.item()
            dice_metric.update(preds, y)
            iou_metric.update(preds, y)

    vl = val_loss / len(val_loader)
    vd = dice_metric.compute().item()
    vi = iou_metric.compute().item()


    # Store metrics
    history["train_loss"].append(tl)
    history["val_loss"].append(vl)
    history["train_dice"].append(td)
    history["val_dice"].append(vd)
    history["train_iou"].append(ti)
    history["val_iou"].append(vi)

    print(f"Epoch {epoch+1} | Train Loss: {tl:.4f} | Val Loss: {vl:.4f} | Train Dice: {td:.4f} | Val Dice: {vd:.4f} | Train IoU: {ti:.4f} | Val IoU: {vi:.4f}")

    # MODEL SELECTION
    # Save best model based on dice score
    if vd > best_dice:
        best_dice = vd
        torch.save(model.state_dict(),f"{SAVE_DIR}best_model.pth")
        print("Best model saved!")
        early_stop_counter = 0
    else:
        early_stop_counter += 1

    # EARLY STOPPING
    if early_stop_counter >= patience:
      print("Early stopping!")
      break


# PLOT AND SAVE TRAINING CURVES
def plot_metric(name):
    plt.figure()
    plt.plot(history[f"train_{name}"])
    plt.plot(history[f"val_{name}"])
    plt.legend(["train","val"])
    plt.title(name)
    plt.savefig(f"{SAVE_DIR}{name}.png")
    plt.close()

for m in ["loss","dice","iou"]:
    plot_metric(m)


# SAVE RESULTS USING THE MODEL SELECTED BASED ON DICE
model.load_state_dict(torch.load(f"{SAVE_DIR}best_model.pth"))
model.eval()

precision = BinaryPrecision().to(device)
recall    = BinaryRecall().to(device)

dice_metric.reset()
iou_metric.reset()

with torch.no_grad():
    for x,y in val_loader:
        x,y=x.to(device),y.to(device)
        preds=torch.sigmoid(model(x))>0.5

        dice_metric.update(preds,y)
        iou_metric.update(preds,y)
        precision.update(preds,y)
        recall.update(preds,y)

results={
    "Dice":dice_metric.compute().item(),
    "IoU":iou_metric.compute().item(),
    "Precision":precision.compute().item(),
    "Recall":recall.compute().item()}

# Save results in TXT format
with open(f"{SAVE_DIR}final_metrics.txt","w") as f:
    for k,v in results.items():
        f.write(f"{k}: {v:.4f}\n")

print(results)