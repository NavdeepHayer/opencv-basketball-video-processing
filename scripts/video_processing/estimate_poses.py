import cv2
import mediapipe as mp

def estimate_poses(frame, player_boxes):
    mp_pose = mp.solutions.pose
    pose_estimator = mp_pose.Pose(static_image_mode=True)
    poses = []

    for (x1, y1, x2, y2) in player_boxes:
        player_region = frame[y1:y2, x1:x2]
        if player_region.size == 0:
            print(f"Skipping invalid player region with coordinates: {(x1, y1, x2, y2)}")
            poses.append(None)
            continue

        player_region_rgb = cv2.cvtColor(player_region, cv2.COLOR_BGR2RGB)
        results = pose_estimator.process(player_region_rgb)
        
        if results.pose_landmarks:
            # Convert pose landmarks to a list of tuples or dictionaries
            pose_data = [{ 'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility } for lm in results.pose_landmarks.landmark]
            poses.append(pose_data)
        else:
            poses.append(None)

    pose_estimator.close()
    return poses
