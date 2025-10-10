import torch
import numpy as np
import cv2
from PIL import Image
import io

IMG_SIZE = (256, 256)

def preprocess_image(image: Image.Image):
    img = np.array(image.convert("RGB"))
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img_tensor

def postprocess_tensor(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    tensor = np.clip(tensor * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(tensor)

def tensor_to_download_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
