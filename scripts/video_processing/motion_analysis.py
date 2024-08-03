import cv2
import numpy as np

class KalmanPlayerTracker:
    def __init__(self):
        # Initialize tracking structures for each player
        self.kalman_filters = {}  # Maps player_id to Kalman filter
        self.next_id = 0
        self.descriptors = {}  # Store feature descriptors for each player
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    def create_kalman_filter(self, bbox):
        """Create a Kalman filter for a new player."""
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 5, 0], [0, 0, 0, 5]], np.float32) * 0.03
        kf.statePre = np.array([bbox[0], bbox[1], 0, 0], np.float32)  # Initial state
        return kf

    def update(self, frame, player_boxes, features):
        """
        Update the tracker with the new detections.

        Args:
        - frame (numpy.ndarray): Current frame.
        - player_boxes (list): List of bounding boxes from current frame detections.
        - features (list): List of feature descriptors for each player.

        Returns:
        - tracked_boxes (list): Updated list of bounding boxes with IDs.
        - player_ids (list): List of IDs corresponding to tracked players.
        """
        tracked_boxes = []
        player_ids = []

        for box, feature in zip(player_boxes, features):
            if feature is None:
                continue

            player_id = self.match_with_existing(feature)
            if player_id is None:
                # Assign a new ID if no match found
                player_id = self.next_id
                self.next_id += 1
                self.descriptors[player_id] = feature
                self.kalman_filters[player_id] = self.create_kalman_filter(box)
            else:
                # Update tracker and descriptors with existing ID
                self.descriptors[player_id] = feature

            # Update Kalman filter with the current measurement
            kf = self.kalman_filters[player_id]
            kf.correct(np.array([box[0], box[1]], np.float32))

            # Predict the next position
            predicted = kf.predict()
            tracked_box = (int(predicted[0]), int(predicted[1]), box[2], box[3])
            self.kalman_filters[player_id] = kf

            tracked_boxes.append(tracked_box)
            player_ids.append(player_id)

        return tracked_boxes, player_ids

    def match_with_existing(self, current_feature):
        """Match current feature with existing player features."""
        best_match_id = None
        best_match_count = 10  # Minimum number of good matches required

        for player_id, existing_feature in self.descriptors.items():
            matches = self.matcher.match(existing_feature, current_feature)
            good_matches = [m for m in matches if m.distance < 0.7 * max([m.distance for m in matches])]
            if len(good_matches) > best_match_count:
                best_match_id = player_id
                best_match_count = len(good_matches)

        return best_match_id
