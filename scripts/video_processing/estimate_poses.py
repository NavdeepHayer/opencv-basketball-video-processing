import cv2
import mediapipe as mp

def estimate_poses(frame, player_boxes):
    """
    Estimate poses for players in a frame.

    Args:
    - frame (numpy.ndarray): The image frame.
    - player_boxes (list): List of bounding boxes (x1, y1, x2, y2) for detected players.

    Returns:
    - list: List of pose landmarks for each player.
    """
    mp_pose = mp.solutions.pose
    pose_estimator = mp_pose.Pose(static_image_mode=True)

    poses = []
    for (x1, y1, x2, y2) in player_boxes:
        player_region = frame[y1:y2, x1:x2]
        if player_region.size == 0:
            print(f"Warning: Skipping invalid player region with coordinates: {(x1, y1, x2, y2)}")
            poses.append(None)
            continue

        # Convert image to RGB
        player_region_rgb = cv2.cvtColor(player_region, cv2.COLOR_BGR2RGB)

        # Estimate pose
        results = pose_estimator.process(player_region_rgb)
        pose_landmarks = results.pose_landmarks
        poses.append(pose_landmarks)

    pose_estimator.close()
    return poses
