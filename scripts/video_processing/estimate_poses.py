import cv2
import mediapipe as mp

def estimate_poses(frame, player_boxes):
    """
    Estimate poses for players in a frame.

    Args:
    - frame (numpy.ndarray): The image frame.
    - player_boxes (list): List of bounding boxes (x, y, width, height) for detected players.

    Returns:
    - list: List of pose landmarks for each player.
    """
    mp_pose = mp.solutions.pose
    pose_estimator = mp_pose.Pose(static_image_mode=True)

    poses = []
    for box in player_boxes:
        x, y, w, h = box
        player_region = frame[y:y+h, x:x+w]

        # Convert image to RGB
        player_region_rgb = cv2.cvtColor(player_region, cv2.COLOR_BGR2RGB)

        # Estimate pose
        results = pose_estimator.process(player_region_rgb)
        pose_landmarks = results.pose_landmarks

        # Convert pose landmarks to a list of dictionaries
        if pose_landmarks:
            landmarks = [
                {'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility}
                for lm in pose_landmarks.landmark
            ]
            poses.append(landmarks)
        else:
            poses.append([])

    pose_estimator.close()
    return poses

if __name__ == "__main__":
    frame_path = "../frames/sample_video/frame_0.jpg"
    frame = cv2.imread(frame_path)
    player_boxes = [(100, 100, 200, 200)]  # Dummy box
    poses = estimate_poses(frame, player_boxes)
    print(f"Estimated poses for {len(poses)} players.")
