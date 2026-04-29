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
SAVE_DIR = "/results/unet_attention/"
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
# Attention U-Net
# Conv Block
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

# Gated Attention Block for skip connections
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# Bottleneck Attention
class BottleneckAttention(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.query = nn.Conv2d(in_ch, in_ch//8, 1)
        self.key   = nn.Conv2d(in_ch, in_ch//8, 1)
        self.value = nn.Conv2d(in_ch, in_ch, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B,C,H,W = x.size()
        q = self.query(x).view(B, -1, H*W).permute(0,2,1)
        k = self.key(x).view(B, -1, H*W)
        v = self.value(x).view(B, -1, H*W)

        attn = torch.softmax(torch.bmm(q,k), dim=-1)
        out = torch.bmm(v, attn.permute(0,2,1))
        out = out.view(B,C,H,W)
        return self.gamma * out + x

# Attention Backbone UNet
class AttentionUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()

        # Encoder
        self.d1 = ConvBlock(in_ch, 64)
        self.d2 = ConvBlock(64, 128)
        self.d3 = ConvBlock(128, 256)
        self.d4 = ConvBlock(256, 512)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bridge = ConvBlock(512, 1024)
        self.b_attn = BottleneckAttention(1024)

        # Decoder
        self.u4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.att4 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.c4 = ConvBlock(1024, 512)

        self.u3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.att3 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.c3 = ConvBlock(512, 256)

        self.u2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.att2 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.c2 = ConvBlock(256, 128)

        self.u1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.att1 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.c1 = ConvBlock(128, 64)

        # Output
        self.out = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        # Encoder
        d1 = self.d1(x)
        d2 = self.d2(self.pool(d1))
        d3 = self.d3(self.pool(d2))
        d4 = self.d4(self.pool(d3))

        # Bottleneck with attention
        b = self.bridge(self.pool(d4))
        #b = self.b_attn(b)

        # Decoder with gated attention
        u4 = self.u4(b)
        d4_att = self.att4(u4, d4)
        u4 = self.c4(torch.cat([u4, d4_att], dim=1))

        u3 = self.u3(u4)
        d3_att = self.att3(u3, d3)
        u3 = self.c3(torch.cat([u3, d3_att], dim=1))

        u2 = self.u2(u3)
        d2_att = self.att2(u2, d2)
        u2 = self.c2(torch.cat([u2, d2_att], dim=1))

        u1 = self.u1(u2)
        d1_att = self.att1(u1, d1)
        u1 = self.c1(torch.cat([u1, d1_att], dim=1))

        return self.out(u1)

model = AttentionUNet().to(device)

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