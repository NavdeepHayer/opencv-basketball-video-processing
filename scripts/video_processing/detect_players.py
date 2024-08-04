import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import numpy as np

def load_model(weights_path=None, device='cuda'):
    """
    Load the Faster R-CNN model with optional pre-trained weights.

    Args:
    - weights_path (str): Path to the model weights file (if any).
    - device (str): Device to load the model on ('cuda' or 'cpu').

    Returns:
    - model: The loaded Faster R-CNN model.
    - device: The device used.
    """
    model = fasterrcnn_resnet50_fpn(pretrained=True)  # or use specific weights if needed
    model = model.to(device)
    model.eval()

    if weights_path:
        model.load_state_dict(torch.load(weights_path))

    return model, device
def detect_players(frame, model, device):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    
    # Transform the frame and transfer it to the GPU
    frame_tensor = transform(frame).unsqueeze(0).to(device)  # Ensure data is on GPU
    
    with torch.no_grad():
        prediction = model(frame_tensor)  # Model prediction should be on GPU
    
    player_boxes = []
    scores = []
    for idx, box in enumerate(prediction[0]['boxes']):
        score = prediction[0]['scores'][idx].item()
        if score > 0.7:  # Confidence threshold
            x1, y1, x2, y2 = box.detach().cpu().numpy().astype(int)  # Move to CPU for further processing
            player_boxes.append((x1, y1, x2, y2))
            scores.append(score)
    
    return player_boxes, scores
