import torch
import torchvision.models as models
import torchvision.transforms as T
from torchvision.datasets import EuroSAT
import numpy as np

USE_GPU = torch.cuda.is_available()
BATCH_SIZE = 64 if USE_GPU else 32
OUTPUT_FILE = 'eurosat_features.npz'

print(f"Using: {'GPU' if USE_GPU else 'CPU'}")

model = models.resnet18(weights='IMAGENET1K_V1')
model = torch.nn.Sequential(*list(model.children())[:-1])  # remove FC layer
model.eval()

if USE_GPU:
    model = model.cuda()


transform = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("Downloading EuroSAT dataset...")
dataset = EuroSAT(root='./data', download=True, transform=transform)
loader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    num_workers=2 if USE_GPU else 0
)

print(f"Dataset size: {len(dataset)} images")
print(f"Classes: {dataset.classes}")
print()
print("Extracting features...")
features, labels = [], []

with torch.no_grad():
    for i, (imgs, lbls) in enumerate(loader):
        if USE_GPU:
            imgs = imgs.cuda()
        
        feat = model(imgs).squeeze()
        
        if USE_GPU:
            feat = feat.cpu()
        
        features.append(feat.numpy())
        labels.extend(lbls.numpy())

features = np.vstack(features)
labels = np.array(labels)

print(f"Features shape: {features.shape}")
print(f"Labels shape: {labels.shape}")

np.savez(OUTPUT_FILE, features=features, labels=labels)
print(f"Saved: {OUTPUT_FILE}")