import sys
import os
import cv2
import torch
import numpy as np

# Adjust the path based on your folder structure
path_to_yolov5 = os.path.join(os.path.dirname(__file__), '../../models/yolov5')
sys.path.insert(0, path_to_yolov5)

from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.torch_utils import select_device

def load_model(weights_path='../../models/yolov5/yolov5s.pt', device=''):
    device = select_device(device)  # Selects the appropriate device
    model = attempt_load(weights_path)  # Load the model
    model.to(device)  # Assign the model to the correct device
    model.eval()  # Set the model to evaluation mode
    return model, device

def detect_players(frame, model, device):
    """Detect players in a frame using the YOLOv5 model."""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img, ratio, (dw, dh) = letterbox(img, 640, stride=32)
    img = img.transpose((2, 0, 1))[None]
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0

    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, 0.25, 0.45)

    player_boxes = []
    for det in pred:
        if len(det):
            det[:, :4] = rescale_boxes(det[:, :4], frame.shape, img.shape[2:])
            for *xyxy, conf, cls in reversed(det):
                x1, y1, x2, y2 = map(int, xyxy)
                player_boxes.append((x1, y1, x2 - x1, y2 - y1))

    return player_boxes

def rescale_boxes(boxes, original_shape, current_shape):
    gain = min(current_shape[0] / original_shape[0], current_shape[1] / original_shape[1])
    pad = (current_shape[1] - original_shape[1] * gain) / 2, (current_shape[0] - original_shape[0] * gain) / 2
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes[:, :4] /= gain
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, original_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, original_shape[0])
    return boxes

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
