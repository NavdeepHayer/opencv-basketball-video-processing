import numpy as np

def non_max_suppression(boxes, scores, threshold=0.3):
    """
    Perform Non-Maximum Suppression to eliminate redundant overlapping bounding boxes

    Args:
    boxes (array): Array of bounding boxes (x1, y1, x2, y2).
    scores (array): Confidence scores for each bounding box.
    threshold (float): Overlap threshold for suppression.

    Returns:
    list: Indices of bounding boxes to keep.
    """
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)  # Sort by score ascending

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Calculate the overlap ratio
        overlap = (w * h) / area[idxs[:last]]

        # Keep only boxes with an overlap ratio less than the threshold
        overlap_indices = np.where(overlap <= threshold)[0]

        # Ensure indices are within valid range
        if len(overlap_indices) == 0:
            idxs = idxs[:last]  # Remove the last element only
        else:
            idxs = idxs[overlap_indices]

    return pick

if __name__ == "__main__":
    # Example usage
    boxes = np.array([[100, 100, 210, 210], [120, 120, 230, 230], [80, 80, 200, 200]])
    scores = np.array([0.9, 0.75, 0.85])
    print("Indices of boxes to keep:", non_max_suppression(boxes, scores, 0.5))
