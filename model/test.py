import torch
from torch.utils.data import DataLoader
from dataset import create_dataloaders
from model import UNet
from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageDraw, ImageFont
import os
from math import log10
from skimage.metrics import structural_similarity as ssim


def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return float('inf')
    return 10 * log10(1 / mse)

def calculate_ssim(img1, img2):
    img1_np = img1.permute(1, 2, 0).cpu().numpy()  # CHW to HWC
    img2_np = img2.permute(1, 2, 0).cpu().numpy()
    return ssim(img1_np, img2_np, channel_axis=-1, data_range=1)

def add_titles_to_image(image, titles, font=None, title_height=30):
    width, height = image.size
    title_img = Image.new("RGB", (width, title_height), (255, 255, 255))
    draw = ImageDraw.Draw(title_img)
    num_titles = len(titles)
    segment_width = width // num_titles
    for i, title in enumerate(titles):
        bbox = draw.textbbox((0,0), title, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = segment_width * i + (segment_width - w) // 2
        y = (title_height - h) // 2
        draw.text((x, y), title, fill=(0, 0, 0), font=font)
    combined = Image.new("RGB", (width, height + title_height), (255, 255, 255))
    combined.paste(title_img, (0, 0))
    combined.paste(image, (0, title_height))
    return combined

def evaluate_full_testset(model_path, dataset_path, batch_size=8, use_transformer=False, n_visuals=5, save_dir='eval_sample'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(use_transformer=use_transformer).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    _, _, test_loader = create_dataloaders(dataset_path, batch_size=batch_size)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    total_psnr = 0.0
    total_ssim = 0.0
    count = 0
    visualized = 0

    for batch in test_loader:
        if batch is None:
            continue
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            outputs = torch.clamp(outputs, 0.0, 1.0)

        # Calculate PSNR/SSIM for each image in batch
        for i in range(outputs.size(0)):
            pred = outputs[i].cpu()
            gt = targets[i].cpu()
            inp = inputs[i].cpu()
            psnr_val = calculate_psnr(pred, gt)
            ssim_val = calculate_ssim(pred, gt)
            total_psnr += psnr_val
            total_ssim += ssim_val
            count += 1

            # Save a few visual samples
            if visualized < n_visuals:
                pil_inp = to_pil_image(inp)
                pil_pred = to_pil_image(pred)
                pil_gt = to_pil_image(gt)
                size = 256
                pil_inp = pil_inp.resize((size, size))
                pil_pred = pil_pred.resize((size, size))
                pil_gt = pil_gt.resize((size, size))
                canvas = Image.new('RGB', (size * 3, size))
                canvas.paste(pil_inp, (0, 0))
                canvas.paste(pil_pred, (size, 0))
                canvas.paste(pil_gt, (2 * size, 0))
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except Exception:
                    font = ImageFont.load_default()
                canvas = add_titles_to_image(canvas, ["Input", "Enhanced", "Ground Truth"], font)
                img_name = os.path.join(save_dir, f'{model_name}_test_{visualized+1}.png')
                canvas.save(img_name)
                visualized += 1

    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    print(f"\nTest Results: Average PSNR (full test set): {avg_psnr:.4f}, Average SSIM: {avg_ssim:.4f}")

if __name__ == "__main__":
    model_file = 'unet_v2_best_model.pth'
    model_name = os.path.splitext(os.path.basename(model_file))[0]
    dataset_folder = "EUVP"
    evaluate_full_testset(model_file, dataset_folder, batch_size=8, use_transformer=True, n_visuals=5, save_dir='eval_sample')