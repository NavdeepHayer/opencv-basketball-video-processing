import cv2
import numpy as np

class PlayerTracker:
    def __init__(self):
        # Initialize tracking structures
        self.trackers = {}
        self.next_id = 0
        self.histograms = {}

    def update(self, frame, player_boxes):
        """
        Update the tracker with the new detections.

        Args:
        - frame (numpy.ndarray): Current frame.
        - player_boxes (list): List of bounding boxes from current frame detections.

        Returns:
        - tracked_boxes (list): Updated list of bounding boxes with IDs.
        - player_ids (list): List of IDs corresponding to tracked players.
        """
        tracked_boxes = []
        player_ids = []

        # Calculate histograms for current frame detections
        current_histograms = [self.compute_histogram(frame, box) for box in player_boxes]

        for box, current_hist in zip(player_boxes, current_histograms):
            player_id = self.match_with_existing(current_hist)
            if player_id is None:
                # Assign a new ID if no match found
                player_id = self.next_id
                self.next_id += 1
                self.trackers[player_id] = box
                self.histograms[player_id] = current_hist
            else:
                # Update tracker and histogram with existing ID
                self.trackers[player_id] = box
                self.histograms[player_id] = current_hist

            tracked_boxes.append(box)
            player_ids.append(player_id)

        return tracked_boxes, player_ids

    def compute_histogram(self, frame, box):
        """Compute the color histogram for the player box."""
        x1, y1, x2, y2 = map(int, box)
        player_region = frame[y1:y2, x1:x2]
        hsv_player_region = cv2.cvtColor(player_region, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_player_region], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist

    def match_with_existing(self, current_hist):
        """Match current histogram with existing player histograms."""
        best_match_id = None
        best_match_score = 0.7  # Threshold for histogram similarity, tune this as needed

        for player_id, existing_hist in self.histograms.items():
            score = cv2.compareHist(existing_hist, current_hist, cv2.HISTCMP_CORREL)
            if score > best_match_score:
                best_match_id = player_id
                best_match_score = score

        return best_match_id
