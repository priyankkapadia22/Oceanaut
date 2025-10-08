import os
import glob
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from torchvision.utils import save_image

IMG_H, IMG_W = 256, 256
BATCH_SIZE = 8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CHECKPOINT_FUSION = "fusion_model_epoch_48.pth" # <-- update to your checkpoint
CHECKPOINT_55 = r"D:\Oceanaut\New folder\checkpoints\model_epoch_55.pth"
CHECKPOINT_85 = r"D:\Oceanaut\New folder\checkpoints\model_epoch_85.pth"
DATA_ROOT = r"D:\Oceanaut\New folder\EUVP\test_samples" # <-- test set path
VISUAL_DIR = "fusion_test_visuals"
os.makedirs(VISUAL_DIR, exist_ok=True)

def read_image_torch(path, target_size=(IMG_W, IMG_H)):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, target_size)
    im = im.astype(np.float32) / 255.0
    im_tensor = torch.from_numpy(im).permute(2, 0, 1) # (3,H,W)
    return im_tensor

class FusionDatasetTest(Dataset):
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
        return stacked, gt1, in1 # return input (stacked), target (gt1), center input frame

class FusionUNet(nn.Module):
    # ... use the same definition as in your training code ...
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

if __name__ == "__main__":
    # Load models
    from model import EnhancedUNet
    model_55 = EnhancedUNet(use_transformer=True).to(DEVICE)
    model_55.load_state_dict(torch.load(CHECKPOINT_55, map_location=DEVICE))
    model_55.eval()
    model_85 = EnhancedUNet(use_transformer=True).to(DEVICE)
    model_85.load_state_dict(torch.load(CHECKPOINT_85, map_location=DEVICE))
    model_85.eval()

    fusion_model = FusionUNet().to(DEVICE)
    fusion_model.load_state_dict(torch.load(CHECKPOINT_FUSION, map_location=DEVICE))
    fusion_model.eval()

    input_files = sorted(glob.glob(os.path.join(DATA_ROOT, "Inp", "*")))
    ref_files = sorted(glob.glob(os.path.join(DATA_ROOT, "GTr", "*")))

    dataset = FusionDatasetTest(input_files, ref_files, model_55, model_85, DEVICE)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    print("Testing model...")
    for idx, (X, gt, in_img) in enumerate(dataloader):
        with torch.no_grad():
            X = X.to(DEVICE).unsqueeze(0) if X.dim() == 3 else X.to(DEVICE) # [1,18,H,W]
            pred = fusion_model(X)
            pred = pred.squeeze(0).cpu().clamp(0, 1)
            gt = gt.squeeze(0).clamp(0, 1)
            psnr = compare_psnr(gt.numpy().transpose(1,2,0), pred.numpy().transpose(1,2,0), data_range=1)
            ssim = compare_ssim(gt.numpy().transpose(1,2,0), pred.numpy().transpose(1,2,0), channel_axis=2, data_range=1)
            total_psnr += psnr
            total_ssim += ssim
            count += 1
            # Save visuals for the first 10 samples
            if idx < 10:
                imgs = [in_img.squeeze(0).clamp(0,1), pred, gt]
                img_out = torch.cat(imgs, dim=2) # [3, H, 3*W]
                #save_image(img_out, os.path.join(VISUAL_DIR, f"cmp_{idx+1}.png"))
    print(f"\nTest Results: Average PSNR = {total_psnr/count:.2f} dB | Average SSIM = {total_ssim/count:.4f}")
    print(f"Visual comparison images saved in: {VISUAL_DIR}")