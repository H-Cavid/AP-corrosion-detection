# Fine-tuning script for ResNet-50 (binary corrosion classification)
# Executed as a standalone Python script in a tmux session for long-running training (random seed = 123)


import torch, time, random, numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os

seed = 123 # setting random seed for reproducible training
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# defining project paths relative to repository root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

data_dir = os.path.join(PROJECT_ROOT, "data")
model_name = os.path.join(
    PROJECT_ROOT, "models", f"resnet50_finetuned_seed{seed}_best.pth"
)
history_name = os.path.join(
    PROJECT_ROOT, "results", f"finetune_history_seed{seed}.npy"
)

# previous server-specific paths used during training
# data_dir = "/home/shared-data/corrosion_images"
# model_name = f"resnet50_finetuned_seed{seed}_best.pth"
# history_name = f"finetune_history_seed{seed}.npy"

transform = transforms.Compose([ # defining image preprocessing and normalization
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

print(f" Loading dataset from: {data_dir}") # loading full dataset and creating train, validation, and test splits
full_dataset = datasets.ImageFolder(data_dir, transform=transform)
print(f" Loaded {len(full_dataset)} images total.")

print(" Splitting dataset into train/val/test (80/10/10)...")
train_size = int(0.8 * len(full_dataset))
val_size   = int(0.1 * len(full_dataset))
test_size  = len(full_dataset) - train_size - val_size
train_ds, val_ds, test_ds = random_split(full_dataset,[train_size,val_size,test_size])
print(f"   → Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

train_loader = DataLoader(train_ds,batch_size=64,shuffle=True)
val_loader   = DataLoader(val_ds,batch_size=64,shuffle=False)
test_loader  = DataLoader(test_ds,batch_size=64,shuffle=False)

print("\n Loading pretrained ResNet-50 model...") # loading ImageNet-pretrained ResNet-50 backbone
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

for name, param in model.named_parameters(): # freezing early layers and fine-tuning higher layers and classifier
    if "layer3" in name or "layer4" in name or "fc" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f" Model ready on device: {device}")

criterion = nn.CrossEntropyLoss() # defining loss function, optimizer, and training settings
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0005)
best_val = 0
patience = 7
pat_left = patience
epochs = 40

history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []} # Initializing history dict

print("\n Starting fine-tuning...\n") # running training loop with validation after each epoch
start_time = time.time()

for epoch in range(epochs):
    model.train()
    correct, total, loss_sum = 0, 0, 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

    train_acc = correct / total
    train_loss = loss_sum / total

    # ---- Validation ----
    model.eval()
    v_correct, v_total, v_loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            v_loss_sum += loss.item() * x.size(0)
            v_correct += (out.argmax(1) == y).sum().item()
            v_total += y.size(0)

    val_acc = v_correct / v_total
    val_loss = v_loss_sum / v_total

    history["train_acc"].append(train_acc)     # Saving training history 
    history["val_acc"].append(val_acc)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    print(f"Epoch {epoch+1:02d}: "
          f"Train Acc={train_acc*100:.2f}% | Val Acc={val_acc*100:.2f}% | "
          f"Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f}")

    if val_acc > best_val:     #  Early stopping 

        best_val = val_acc
        pat_left = patience
        torch.save(model.state_dict(), model_name)
        print(f" Model improved and saved as {model_name}")
    else:
        pat_left -= 1
        if pat_left == 0:
            print(f" Early stopping at epoch {epoch+1}")
            break

elapsed = (time.time() - start_time) / 60
print(f"\n Fine-tuning finished in {elapsed:.1f} minutes.")

np.save(history_name, history) # saving training history for later analysis

print(f"Training history saved as {history_name}")

print("🔍 Loading best model and evaluating on test set...") # evaluating best checkpoint on the test set

model.load_state_dict(torch.load(model_name))
model.eval()
correct, total = 0, 0

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

test_acc = 100 * correct / total
print(f"\n Final Test Accuracy (Fine-tuned, seed={seed}): {test_acc:.2f}%")
print(f" All done! Model and history saved for seed={seed}\n")
