import os
import torch
from torch.amp import autocast
from model import EnhancedUNet
from dataset import create_dataloaders
import numpy as np
from skimage.metrics import structural_similarity as ssim
from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageDraw, ImageFont
import random

# Configuration
model_path = r"D:\Oceanaut\New folder\fusion_model_epoch_26.pth"  # Update to your best checkpoint
dataset_path = "EUVP"
batch_size = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
visual_samples = 5
visual_dir = "test_visuals"

os.makedirs(visual_dir, exist_ok=True)

# PSNR
def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return 100.0
    return 10 * np.log10(1.0 / mse)

# SSIM
def calculate_ssim(img1, img2):
    img1_np = img1.permute(1, 2, 0).cpu().numpy()
    img2_np = img2.permute(1, 2, 0).cpu().numpy()
    return ssim(img1_np, img2_np, channel_axis=-1, data_range=1)

# Add titles
def add_titles(image, titles):
    width, height = image.size
    title_height = 30
    title_img = Image.new("RGB", (width, title_height), (255, 255, 255))
    draw = ImageDraw.Draw(title_img)
    num_titles = len(titles)
    seg_width = width // num_titles
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    for i, title in enumerate(titles):
        w, h = draw.textsize(title, font=font)
        x = seg_width * i + (seg_width - w) // 2
        y = (title_height - h) // 2
        draw.text((x, y), title, fill=(0, 0, 0), font=font)
    combined = Image.new("RGB", (width, height + title_height), (255, 255, 255))
    combined.paste(title_img, (0, 0))
    combined.paste(image, (0, title_height))
    return combined

def evaluate_model(model_file, dataset_path, batch_size=4, n_visuals=5):
    _, _, test_loader = create_dataloaders(dataset_path, batch_size=batch_size, num_workers=0)
    model = EnhancedUNet(use_transformer=True).to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()

    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    # Sample indices for visuals
    indices = list(range(len(test_loader.dataset)))
    random.shuffle(indices)
    sample_indices = set(indices[:n_visuals])
    saved_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch is None:
                continue
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            with autocast('cuda'):
                outputs = model(inputs)
            outputs = torch.clamp(outputs, 0.0, 1.0)

            for j in range(outputs.size(0)):
                input_img = inputs[j].cpu()
                output_img = outputs[j].cpu()
                target_img = targets[j].cpu()

                psnr_val = calculate_psnr(output_img, target_img)
                ssim_val = calculate_ssim(output_img, target_img)
                total_psnr += psnr_val
                total_ssim += ssim_val
                count += 1

                # Save visual samples
                global_idx = batch_idx * batch_size + j
                if global_idx in sample_indices and saved_count < n_visuals:
                    imgs = [
                        to_pil_image(input_img),
                        to_pil_image(output_img),
                        to_pil_image(target_img)
                    ]
                    imgs = [img.resize((256, 256)) for img in imgs]
                    canvas = Image.new('RGB', (256 * 3, 256))
                    for k, img in enumerate(imgs):
                        canvas.paste(img, (256 * k, 0))
                    canvas = add_titles(canvas, ["Input", "Enhanced", "Ground Truth"])
                    canvas.save(os.path.join(visual_dir, f"visual_{saved_count+1}.png"))
                    saved_count += 1

    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    print(f"Test set results: Average PSNR = {avg_psnr:.2f} dB | Average SSIM = {avg_ssim:.4f}")
    print(f"Visual comparison images saved in: {visual_dir}")

if __name__ == "__main__":
    evaluate_model(model_path, dataset_path, batch_size=batch_size, n_visuals=visual_samples)