import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import L1Loss
from model import UNet
from dataset import create_dataloaders
import torchvision.models as models
from pytorch_msssim import ms_ssim

# ----------------- Perceptual Loss -----------------
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.features = nn.Sequential(*list(vgg16.features.children())[:9]).eval()
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, enhanced_img, target_img):
        enhanced_resized = nn.functional.interpolate(enhanced_img, size=(224, 224), mode='bilinear')
        target_resized = nn.functional.interpolate(target_img, size=(224, 224), mode='bilinear')
        return nn.functional.l1_loss(self.features(enhanced_resized), self.features(target_resized))

# ----------------- Training Loop -----------------
def train_model(dataset_base_path, num_epochs=100, batch_size=8, use_transformer=False):
    train_loader, validation_loader, test_loader = create_dataloaders(
        dataset_base_path=dataset_base_path,
        batch_size=batch_size
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = UNet(use_transformer=use_transformer).to(device)

    l1_criterion = L1Loss()
    perceptual_criterion = PerceptualLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    print("Starting training with combined L1 + Perceptual + MS-SSIM losses...")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        # Define dynamic weights for loss terms (example: ramp up perceptual and MS-SSIM loss weights)
        max_perceptual_weight = 0.01
        max_msssim_weight = 0.1
        ramp_up_epochs = 30

        # Calculates weight factor from 0 to max over ramp_up_epochs
        def get_weight(epoch, max_weight):
            if epoch >= ramp_up_epochs:
                return max_weight
            else:
                return max_weight * (epoch / ramp_up_epochs)

        perceptual_weight = get_weight(epoch, max_perceptual_weight)
        msssim_weight = get_weight(epoch, max_msssim_weight)

        for i, data in enumerate(train_loader):
            if data is None:
                continue
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            l1_loss = l1_criterion(outputs, targets)
            perceptual_loss = perceptual_criterion(outputs, targets)
            msssim_loss = 1 - ms_ssim(outputs, targets, data_range=1.0, size_average=True)

            loss = l1_loss + perceptual_weight * perceptual_loss + msssim_weight * msssim_loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(validation_loader):
                if data is None:
                    continue
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                l1_loss = l1_criterion(outputs, targets)
                perceptual_loss = perceptual_criterion(outputs, targets)
                msssim_loss = 1 - ms_ssim(outputs, targets, data_range=1.0, size_average=True)

                loss = l1_loss + 0.01 * perceptual_loss + 0.1 * msssim_loss
                val_loss += loss.item() * inputs.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(validation_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'unet_se_transformer_best_model.pth')
            print("Model saved successfully!")
        else:
            epochs_no_improve += 1
            print(f"No improvement. Epochs without improvement: {epochs_no_improve}")
            if epochs_no_improve == 15:
                print("Early stopping triggered.")
                break

    print("Training completed.")
    return model

if __name__ == '__main__':
    dataset_path = "EUVP"
    model = train_model(dataset_base_path=dataset_path, use_transformer=True)