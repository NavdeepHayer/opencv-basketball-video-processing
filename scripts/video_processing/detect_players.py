import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import numpy as np
import cv2
import torchvision.transforms as T

def load_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model = model.to(device)
    model.eval()
    return model, device

def detect_players(frame, model, device):
    transform = T.Compose([T.ToTensor()])
    frame_tensor = transform(frame).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        predictions = model(frame_tensor)

    player_boxes = []
    scores = []
    # Assuming the model returns boxes and scores separately
    if len(predictions[0]['boxes']) > 0:
        boxes = predictions[0]['boxes'].detach().cpu().numpy()
        scores_list = predictions[0]['scores'].detach().cpu().numpy()

        for box, score in zip(boxes, scores_list):
            if score > 0.7:  # Confidence threshold
                player_boxes.append(box)
                scores.append(score)

    return player_boxes, scores

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)
