import torch
import torchvision.transforms as T
from PIL import Image
import cv2

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])

def preprocess_frame(frame_path):
    image = Image.open(frame_path).convert("RGB")
    return transform(image).unsqueeze(0)  # shape: (1,3,H,W)

def postprocess_tensor(tensor):
    image = tensor.squeeze(0).detach().cpu().permute(1,2,0).numpy()
    image = (image * 255).clip(0,255).astype("uint8")
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
