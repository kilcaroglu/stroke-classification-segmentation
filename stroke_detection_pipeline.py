# IMPORT REQUIRED LIBRARIES
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models, transforms
import segmentation_models_pytorch as smp

# DEVICE
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 3
CLASS_NAMES = ["Hemorrhagic","Ischemic","No Stroke"]

# TRANSFORMS
imagenet_mean = [0.485,0.456,0.406]
imagenet_std  = [0.229,0.224,0.225]

transform_224 = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean,imagenet_std)
])

transform_299 = transforms.Compose([
    transforms.Resize((299,299)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean,imagenet_std)
])

seg_resize = transforms.Resize((512,512))

# LOAD CLASSIFICATION MODELS
models_list = []

# ResNet18
m1 = models.resnet18(weights="IMAGENET1K_V1")
m1.fc = nn.Linear(m1.fc.in_features,NUM_CLASSES)
m1.load_state_dict(torch.load("/classification/base_models/resnet18/model_weights/fold5_best_f1_macro_model_weights.pth",map_location=DEVICE))
m1.to(DEVICE).eval()
models_list.append((m1,transform_224))

# DenseNet121
m2 = models.densenet121(weights="IMAGENET1K_V1")
m2.classifier = nn.Linear(m2.classifier.in_features,NUM_CLASSES)
m2.load_state_dict(torch.load("/classification/base_models/densenet121/model_weights/fold3_best_f1_macro_model_weights.pth",map_location=DEVICE))
m2.to(DEVICE).eval()
models_list.append((m2,transform_224))

# EfficientNet-B0
m3 = models.efficientnet_b0(weights="IMAGENET1K_V1")
m3.classifier[1] = nn.Linear(m3.classifier[1].in_features,NUM_CLASSES)
m3.load_state_dict(torch.load("/classification/base_models/efficientnet_b0/model_weights/fold3_best_f1_macro_model_weights.pth",map_location=DEVICE))
m3.to(DEVICE).eval()
models_list.append((m3,transform_224))

# InceptionV3
m4 = models.inception_v3(weights="IMAGENET1K_V1", aux_logits=True)
m4.aux_logits = False
m4.AuxLogits = None
m4.fc = nn.Linear(m4.fc.in_features,NUM_CLASSES)
m4.load_state_dict(torch.load("/classification/base_models/inceptionv3/model_weights/fold1_best_f1_macro_model_weights.pth",map_location=DEVICE))
m4.to(DEVICE).eval()
models_list.append((m4,transform_299))

# MobileNetV3
m5 = models.mobilenet_v3_large(weights="IMAGENET1K_V1")
m5.classifier[3] = nn.Linear(m5.classifier[3].in_features,NUM_CLASSES)
m5.load_state_dict(torch.load("/classification/base_models/mobilenetv3_large/model_weights/fold4_best_f1_macro_model_weights.pth",map_location=DEVICE))
m5.to(DEVICE).eval()
models_list.append((m5,transform_224))


# LOAD SEGMENTATION MODEL
seg_model = smp.Unet(
    encoder_name="efficientnet-b0",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None
)

seg_model.load_state_dict(torch.load("/segmentation/loss_comparison/BCE_Loss/best_model.pth",map_location=DEVICE))
seg_model.to(DEVICE)
seg_model.eval()


# ENSEMBLE CLASSIFICATION
def ensemble_predict(image):

    logits_sum = None

    for model,transform in models_list:

        img = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(img)

        if logits_sum is None:
            logits_sum = logits
        else:
            logits_sum += logits

    logits_avg = logits_sum / len(models_list)

    pred = torch.argmax(logits_avg,dim=1).item()

    return CLASS_NAMES[pred]


# SEGMENTATION
def segment_image(image):

    img = seg_resize(image)

    tensor = transforms.ToTensor()(img)
    tensor = transforms.Normalize(imagenet_mean,imagenet_std)(tensor)

    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        logits = seg_model(tensor)
        pred = torch.sigmoid(logits)
        pred = (pred>0.5).float()

    mask = pred.squeeze().cpu().numpy()

    mask = cv2.resize(mask,(image.size[0],image.size[1]))

    return mask


# OVERLAY MASK
def overlay_mask(image,mask):

    image_np = np.array(image)

    mask_color = np.zeros_like(image_np)
    mask_color[:,:,0] = mask*255

    overlay = cv2.addWeighted(image_np,0.7,mask_color,0.3,0)

    return overlay


# MAIN PIPELINE
IMAGE_PATH = "ct_image.png"

image = Image.open(IMAGE_PATH).convert("RGB")

prediction = ensemble_predict(image)

print("Prediction:",prediction)

if prediction == "No Stroke":

    plt.imshow(image)
    plt.title("No Stroke Detected")
    plt.axis("off")
    plt.show()

else:

    mask = segment_image(image)

    overlay = overlay_mask(image,mask)

    plt.figure(figsize=(12,8))

    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(overlay)
    plt.title(f"{prediction} Lesion")
    plt.axis("off")

    plt.show()