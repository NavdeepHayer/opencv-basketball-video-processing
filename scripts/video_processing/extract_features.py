import cv2

def extract_features(frame, player_boxes):
    """
    Extract features from players in a frame.

    Args:
    - frame (numpy.ndarray): The image frame.
    - player_boxes (list): List of bounding boxes (x, y, width, height) for detected players.

    Returns:
    - list: List of feature descriptors for each player.
    """
    features = []
    sift = cv2.SIFT_create()

    for box in player_boxes:
        x, y, w, h = box
        player_region = frame[y:y+h, x:x+w]
        keypoints, descriptors = sift.detectAndCompute(player_region, None)
        features.append(descriptors)

    return features

if __name__ == "__main__":
    frame_path = "../frames/sample_video/frame_0.jpg"
    frame = cv2.imread(frame_path)
    player_boxes = [(100, 100, 200, 200)]  # Dummy box
    features = extract_features(frame, player_boxes)
    print(f"Extracted features from {len(features)} players.")
