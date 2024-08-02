import sys
import os
import cv2
import torch

# Adjust the path based on your folder structure
# This should point to the root of the 'yolov5' directory
path_to_yolov5 = os.path.join(os.path.dirname(__file__), '../../models/yolov5')
sys.path.insert(0, path_to_yolov5)

from models.experimental import attempt_load
from utils.general import non_max_suppression, xywh2xyxy, xyxy2xywh
from utils.torch_utils import select_device

def load_model(weights_path='../../models/yolov5/yolov5s.pt', device=''):
    device = select_device(device)  # Selects the appropriate device
    model = attempt_load(weights_path)  # Load the model
    model.to(device)  # Assign the model to the correct device
    model.eval()  # Set the model to evaluation mode
    return model, device

def rescale_boxes(prediction, original_shape, current_shape):
    """Rescale bounding box coordinates from `current_shape` to `original_shape`."""
    gain = min(current_shape[0] / original_shape[0], current_shape[1] / original_shape[1])
    pad = (current_shape[1] - original_shape[1] * gain) / 2, (current_shape[0] - original_shape[0] * gain) / 2

    prediction[:, [0, 2]] -= pad[0]  # x padding
    prediction[:, [1, 3]] -= pad[1]  # y padding
    prediction[:, :4] /= gain
    prediction[:, [0, 2]] = prediction[:, [0, 2]].clamp(0, original_shape[1])  # clip x
    prediction[:, [1, 3]] = prediction[:, [1, 3]].clamp(0, original_shape[0])  # clip y

    return prediction

def process_frame(frame, model, device):
    # Convert BGR to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize and pad image to expected size (640 for yolov5s, or as needed)
    img, ratio, (dw, dh) = letterbox(img, 640, stride=32)

    # Convert
    img = img.transpose((2, 0, 1))[None]  # Add batch dimension
    img = torch.from_numpy(img).to(device)
    img = img.float()  # Convert to float
    img /= 255.0  # Normalize to [0, 1]

    pred = model(img, augment=False)[0]  # Inference
    pred = non_max_suppression(pred, 0.25, 0.45)  # Apply NMS

    # Extract bounding boxes from predictions
    player_boxes = []
    for det in pred:  # detections per image
        if len(det):
            # Rescale boxes from img_size to original frame size
            det = rescale_boxes(det, frame.shape, img.shape[2:])
            for *xyxy, conf, cls in reversed(det):
                x1, y1, x2, y2 = map(int, xyxy)
                player_boxes.append((x1, y1, x2-x1, y2-y1))  # Convert to (x, y, w, h)

    return player_boxes

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize image to a 32-pixel-multiple rectangle
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)
