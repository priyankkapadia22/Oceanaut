import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from dataset import create_dataloaders
from model import EnhancedUNet

# Hyperparams
dataset_path = "EUVP"
batch_size = 4
num_epochs = 100
learning_rate = 1e-5  # lowered for stability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charbonnier Loss only
class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.eps = epsilon
    def forward(self, x, y):
        diff = x - y
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))

# Load data
train_loader, val_loader, _ = create_dataloaders(dataset_path, batch_size=batch_size, num_workers=0)

# Model setup
model = EnhancedUNet(use_transformer=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scaler = GradScaler('cuda')

charbonnier = CharbonnierLoss().to(device)

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        if data is None:
            continue
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        with autocast('cuda'):
            outputs = model(inputs)
            outputs = torch.clamp(outputs, 0, 1)
            loss = charbonnier(outputs, targets)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN/Inf loss skipped at epoch {epoch+1}, batch {batch_idx+1}")
                continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Average Training Loss: {avg_loss:.4f}")

    # Validation (optional)
    model.eval()
    val_loss_total = 0.0
    with torch.no_grad():
        for data in val_loader:
            if data is None:
                continue
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            outputs = torch.clamp(outputs, 0, 1)
            val_loss = charbonnier(outputs, targets)
            val_loss_total += val_loss.item()
    avg_val_loss = val_loss_total / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")

    # Save checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch+1}.pth")
        print(f"Model checkpoint saved at epoch {epoch+1}")
