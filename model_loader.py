import torch
import os
from backend.oceanaut_model.fusion_cnn import FusionUNet

def load_fusion_model(weights_path, device):
    model = FusionUNet().to(device)
    if os.path.exists(weights_path):
        try:
            model.load_state_dict(torch.load(weights_path, map_location=device))
            print("Fusion model weights loaded successfully.")
        except Exception as e:
            print("Error loading weights:", e)
    else:
        print("No weights found at:", weights_path)
    model.eval()
    return model, os.path.exists(weights_path)
