import numpy as np

class PlayerIDTracker:
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.next_id = 0
        self.player_positions = {}  # Maps player_id to bounding box
        self.player_features = {}  # Maps player_id to feature descriptors

    def assign_player_id(self, detected_boxes, detected_features):
        player_ids = []

        if not self.player_positions:
            # If no players tracked yet, assign new IDs
            for box in detected_boxes:
                player_id = self.next_id
                self.next_id += 1
                self.player_positions[player_id] = box
                player_ids.append(player_id)
        else:
            # Calculate IoU between detected boxes and tracked boxes
            existing_boxes = np.array(list(self.player_positions.values()))
            iou_matrix = self.compute_iou_matrix(detected_boxes, existing_boxes)

            for i, box in enumerate(detected_boxes):
                # Find best match for current detection
                max_iou_index = np.argmax(iou_matrix[i])
                max_iou = iou_matrix[i][max_iou_index]

                if max_iou > self.iou_threshold:
                    # Assign existing ID if IoU exceeds threshold
                    player_id = list(self.player_positions.keys())[max_iou_index]
                else:
                    # Assign new ID
                    player_id = self.next_id
                    self.next_id += 1

                self.player_positions[player_id] = box
                self.player_features[player_id] = detected_features[i]
                player_ids.append(player_id)

        return player_ids

    def compute_iou_matrix(self, boxes1, boxes2):
        """
        Compute the Intersection over Union (IoU) between two sets of bounding boxes.

        Args:
        boxes1: List of bounding boxes, e.g., [(x1, y1, x2, y2), ...]
        boxes2: List of bounding boxes, e.g., [(x1, y1, x2, y2), ...]

        Returns:
        IoU matrix of shape (len(boxes1), len(boxes2))
        """
        iou_matrix = np.zeros((len(boxes1), len(boxes2)))

        for i, box1 in enumerate(boxes1):
            for j, box2 in enumerate(boxes2):
                iou_matrix[i, j] = self.compute_iou(box1, box2)

        return iou_matrix

    def compute_iou(self, box1, box2):
        """
        Compute the Intersection over Union (IoU) of two bounding boxes.

        Args:
        box1, box2: Bounding boxes in the format (x1, y1, x2, y2).

        Returns:
        IoU value.
        """
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])

        inter_area = max(0, x2_inter - x1_inter + 1) * max(0, y2_inter - y1_inter + 1)

        box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

        iou = inter_area / float(box1_area + box2_area - inter_area)

        return iou
