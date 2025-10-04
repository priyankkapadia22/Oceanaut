import os
import glob
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ---------- CONFIG ----------
DATA_ROOT = r"D:\Oceanaut\New folder\EUVP\modified\train"
CHECKPOINT_55 = r"D:\Oceanaut\New folder\checkpoints\model_epoch_55.pth"
CHECKPOINT_85 = r"D:\Oceanaut\New folder\checkpoints\model_epoch_85.pth"

IMG_H, IMG_W = 256, 256
BATCH_SIZE = 8
EPOCHS = 60
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_image_torch(path, target_size=(IMG_W, IMG_H)):
    im = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, target_size)
    im = im.astype(np.float32) / 255.0
    im_tensor = torch.from_numpy(im).permute(2, 0, 1)  # (3,H,W)
    return im_tensor

class FusionUNet(nn.Module):
    def __init__(self, in_channels=18, out_channels=3):
        super(FusionUNet, self).__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
        self.enc1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = conv_block(256, 512)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = conv_block(512 + 256, 256)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = conv_block(256 + 128, 128)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = conv_block(128 + 64, 64)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        self.out_act = nn.Sigmoid()
    def forward(self, x):
        c1 = self.enc1(x)
        p1 = self.pool1(c1)
        c2 = self.enc2(p1)
        p2 = self.pool2(c2)
        c3 = self.enc3(p2)
        p3 = self.pool3(c3)
        b = self.bottleneck(p3)
        u3 = self.up3(b)
        u3 = torch.cat([u3, c3], dim=1)
        d3 = self.dec3(u3)
        u2 = self.up2(d3)
        u2 = torch.cat([u2, c2], dim=1)
        d2 = self.dec2(u2)
        u1 = self.up1(d2)
        u1 = torch.cat([u1, c1], dim=1)
        d1 = self.dec1(u1)
        out = self.out_conv(d1)
        out = self.out_act(out)
        return out

def composite_loss(y_pred, y_true, prev_pred, prev_true):
    l1 = torch.mean(torch.abs(y_pred - y_true))
    temporal = torch.mean((y_pred - prev_pred - (y_true - prev_true)) ** 2)
    loss = l1 + 0.5 * l1 + 0.2 * temporal
    return loss

class FusionDataset(Dataset):
    def __init__(self, input_files, ref_files, model_55, model_85, device):
        self.input_files = input_files
        self.ref_files = ref_files
        self.model_55 = model_55
        self.model_85 = model_85
        self.device = device
    def __len__(self):
        return len(self.input_files) - 2
    def __getitem__(self, idx):
        i0, i1, i2 = idx, idx + 1, idx + 2
        in0 = read_image_torch(self.input_files[i0])
        in1 = read_image_torch(self.input_files[i1])
        in2 = read_image_torch(self.input_files[i2])
        with torch.no_grad():
            o55_0 = self.model_55(in0.unsqueeze(0).to(self.device)).squeeze(0).cpu()
            o85_0 = self.model_85(in0.unsqueeze(0).to(self.device)).squeeze(0).cpu()
            o55_1 = self.model_55(in1.unsqueeze(0).to(self.device)).squeeze(0).cpu()
            o85_1 = self.model_85(in1.unsqueeze(0).to(self.device)).squeeze(0).cpu()
            o55_2 = self.model_55(in2.unsqueeze(0).to(self.device)).squeeze(0).cpu()
            o85_2 = self.model_85(in2.unsqueeze(0).to(self.device)).squeeze(0).cpu()
        stacked = torch.cat([o55_0, o85_0, o55_1, o85_1, o55_2, o85_2], dim=0)
        gt1 = read_image_torch(self.ref_files[i1])
        gt0 = read_image_torch(self.ref_files[i0])
        return stacked, gt1, gt0

def find_latest_checkpoint():
    checkpoints = glob.glob("fusion_model_epoch_*.pth")
    if not checkpoints:
        return None, 0
    epochs = [int(os.path.splitext(os.path.basename(ckpt))[0].split('_')[-1]) for ckpt in checkpoints]
    max_epoch = max(epochs)
    latest_ckpt = f"fusion_model_epoch_{max_epoch}.pth"
    return latest_ckpt, max_epoch

if __name__ == "__main__":
    from model import EnhancedUNet

    print("Loading pretrained models...")
    model_55 = EnhancedUNet(use_transformer=True).to(DEVICE)
    model_55.load_state_dict(torch.load(CHECKPOINT_55, map_location=DEVICE))
    model_55.eval()
    model_85 = EnhancedUNet(use_transformer=True).to(DEVICE)
    model_85.load_state_dict(torch.load(CHECKPOINT_85, map_location=DEVICE))
    model_85.eval()
    print("Pretrained models loaded.")

    input_files = sorted(glob.glob(os.path.join(DATA_ROOT, "input", "*")))
    ref_files = sorted(glob.glob(os.path.join(DATA_ROOT, "reference", "*")))
    assert len(input_files) == len(ref_files), "Input/reference counts differ"

    dataset = FusionDataset(input_files, ref_files, model_55, model_85, DEVICE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    fusion_model = FusionUNet().to(DEVICE)
    optimizer = optim.Adam(fusion_model.parameters(), lr=LEARNING_RATE)

    # Resume logic
    resume_ckpt, resume_epoch = find_latest_checkpoint()
    if resume_ckpt:
        print(f"Resuming from checkpoint {resume_ckpt} at epoch {resume_epoch}")
        fusion_model.load_state_dict(torch.load(resume_ckpt, map_location=DEVICE))
        start_epoch = resume_epoch
    else:
        print("Training from scratch")
        start_epoch = 0

    fusion_model.train()
    prev_pred_buffer = None

    for epoch in range(start_epoch, EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        epoch_loss = 0.0
        prev_pred_buffer = None
        for batch_idx, (X_batch, Y_batch, Prev_batch) in enumerate(dataloader):
            X_batch = X_batch.to(DEVICE)
            Y_batch = Y_batch.to(DEVICE)
            Prev_batch = Prev_batch.to(DEVICE)
            if prev_pred_buffer is None:
                prev_pred_buffer = torch.zeros_like(Y_batch)
            optimizer.zero_grad()
            preds = fusion_model(X_batch)
            loss = composite_loss(preds, Y_batch, prev_pred_buffer, Prev_batch)
            loss.backward()
            optimizer.step()
            prev_pred_buffer = preds.detach()
            epoch_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"  Step {batch_idx}/{len(dataloader)} Loss: {loss.item():.6f}")
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.6f}")
        torch.save(fusion_model.state_dict(), f"fusion_model_epoch_{epoch + 1}.pth")