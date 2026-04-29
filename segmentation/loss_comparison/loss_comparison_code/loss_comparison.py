# IMPORT REQUIRED LIBRARIES
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
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
SAVE_DIR = "/loss_comparison_results"
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



# LOSS DEFINITION
# BCE Loss
class BCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        return self.bce(logits, targets)

# Weighted BCE Loss
class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=1.0):
        super().__init__()
        self.pos_weight = torch.tensor([pos_weight])

    def forward(self, logits, targets):
        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight.to(logits.device)
        )
        return loss

# Positive weight calculation function
def calculate_pos_weight(dataset):
    print("Calculating positive weight")
    total_pos = 0
    total_neg = 0
    for _, masks in tqdm(dataset):
        total_pos += masks.sum().item()
        total_neg += (1 - masks).sum().item()
    pos_weight = total_neg / (total_pos + 1e-6)
    return pos_weight

# DICE Loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / \
               (probs.sum() + targets.sum() + self.smooth)

        return 1 - dice

# TVERSKY Loss
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        probs = probs.view(-1)
        targets = targets.view(-1)

        TP = (probs * targets).sum()
        FP = ((1 - targets) * probs).sum()
        FN = (targets * (1 - probs)).sum()

        tversky = (TP + self.smooth) / \
                   (TP + self.alpha * FN + self.beta * FP + self.smooth)

        return 1 - tversky

# FOCAL TVERSKY Loss
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=1.33, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        probs = probs.view(-1)
        targets = targets.view(-1)

        TP = (probs * targets).sum()
        FP = ((1 - targets) * probs).sum()
        FN = (targets * (1 - probs)).sum()

        tversky = (TP + self.smooth) / \
                   (TP + self.alpha * FN + self.beta * FP + self.smooth)

        focal_tversky = torch.pow((1 - tversky), self.gamma)

        return focal_tversky

# COMBO Loss (BCE + DICE)
class ComboLoss(nn.Module):
    def __init__(self, weight_bce=0.5, weight_dice=0.5):
        super().__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)

        return self.weight_bce * bce_loss + \
               self.weight_dice * dice_loss


pos_weight = calculate_pos_weight(train_dataset)
print(f"pos_weight: {pos_weight:.4f}")

loss_list = {
    "BCE_Loss": BCEWithLogitsLoss(),
    "Weighted_BCE_Loss": WeightedBCELoss(pos_weight=pos_weight),
    "Dice_Loss": DiceLoss(),
    "Tversky_Loss": TverskyLoss(alpha=0.7, beta=0.3),
    "Focal_Tversky_Loss": FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=4/3),
    "Combo_Loss": ComboLoss(weight_bce=0.5, weight_dice=0.5)
}

loss_result_list = {}

for loss_name, loss_function in loss_list.items():

    print(f"\nLoss function: {loss_name}")

    LOSS_SAVE_DIR = f"{SAVE_DIR}/{loss_name}"
    os.makedirs(LOSS_SAVE_DIR, exist_ok=True)

    # MODEL DEFINITION
    # UNET with EfficientNet_B0 backbone
    model = smp.Unet(encoder_name="efficientnet-b0",
                 encoder_weights="imagenet",
                 in_channels=3,
                 classes=1,
                 activation=None).to(device)

    criterion = loss_function
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
            torch.save(model.state_dict(),f"{LOSS_SAVE_DIR}/best_model.pth")
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
        plt.savefig(f"{LOSS_SAVE_DIR}/{name}.png")
        plt.close()

    for m in ["loss","dice","iou"]:
        plot_metric(m)


    # SAVE RESULTS USING THE MODEL SELECTED BASED ON DICE
    model.load_state_dict(torch.load(f"{LOSS_SAVE_DIR}/best_model.pth"))
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

    loss_result_list[loss_name] = results

    # Save results in TXT format
    with open(f"{LOSS_SAVE_DIR}/final_metrics.txt","w") as f:
        for k,v in results.items():
            f.write(f"{k}: {v:.4f}\n")

    print(results)

print("Comparison of loss functions is complete.")

columns = ["Loss", "Dice", "IoU", "Precision", "Recall"]
header = "{:<20} {:<8} {:<8} {:<10} {:<8}".format(*columns)

with open(f"{SAVE_DIR}/loss_result.txt", "w") as f:
    print(header)
    f.write(header + "\n")

    for name, result in loss_result_list.items():
        values = [v for v in result.values()]
        row = "{:<20} {:<8.4f} {:<8.4f} {:<10.4f} {:<8.4f}".format(name, *values)
        print(row)
        f.write(row + "\n")