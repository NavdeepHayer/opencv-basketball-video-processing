import cv2

def extract_features(frame, player_boxes):
    """
    Extract features from players in a frame.

    Args:
    - frame (numpy.ndarray): The image frame.
    - player_boxes (list): List of bounding boxes (x1, y1, x2, y2) for detected players.

    Returns:
    - list: List of feature descriptors for each player.
    """
    features = []
    sift = cv2.SIFT_create()

    for box in player_boxes:
        x1, y1, x2, y2 = map(int, box)  # Convert box coordinates to integers
        if x2 > x1 and y2 > y1:  # Ensure valid region
            player_region = frame[y1:y2, x1:x2]

            if player_region.size == 0:
                print(f"Warning: Skipping invalid player region with coordinates: {box}")
                features.append(None)
                continue

            keypoints, descriptors = sift.detectAndCompute(player_region, None)
            if descriptors is None:
                print(f"Warning: No descriptors found for region: {box}")
                features.append(None)
            else:
                features.append(descriptors)
        else:
            print(f"Warning: Skipping invalid player region with coordinates: {box}")
            features.append(None)

    return features
