import torch
import torchvision
import torchvision.transforms as T

def load_deep_learning_model(device='cuda'):
    # Load a pre-trained deep learning model, e.g., ResNet50
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    model = model.to(device)
    model.eval()
    return model

def extract_features(frame, player_boxes, model, device):
    """
    Extract features for each player using a deep learning model.

    Args:
    - frame: The image frame from which to extract features.
    - player_boxes: Bounding boxes of detected players in the frame.
    - model: The pre-trained deep learning model for feature extraction.
    - device: The device to run the model on ('cuda' or 'cpu').

    Returns:
    - features: A list of feature vectors for each player.
    """
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    features = []
    for box in player_boxes:
        x1, y1, x2, y2 = box
        player_region = frame[y1:y2, x1:x2]
        player_region = transform(player_region).unsqueeze(0).to(device)

        with torch.no_grad():
            feature_vector = model(player_region)
        
        features.append(feature_vector.cpu().numpy().flatten())

    return features

