import cv2
import os
import torch
import numpy as np
from tqdm import tqdm
from detect_players import load_model, detect_players
from extract_features import extract_features, load_deep_learning_model
from estimate_poses import estimate_poses
from store_in_db import store_in_db
from player_profile_db import store_player_profile
from kalman_filter_tracking import KalmanFilter
from player_id_tracker import PlayerIDTracker

def draw_poses(frame, player_boxes, poses):
    for box, pose in zip(player_boxes, poses):
        if pose is not None:
            for landmark in pose:
                x = int(landmark['x'] * (box[2] - box[0]) + box[0])
                y = int(landmark['y'] * (box[3] - box[1]) + box[1])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    return frame

def process_single_video(video_path, model, device, processed_directory, frames_directory, player_tracker, feature_extraction_model):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    player_trackers = {}
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_name = f"frame_{frame_count}.jpg"
        player_boxes, scores = detect_players(frame, model, device)

        if player_boxes and scores:
            # Convert list of numpy arrays to a single numpy array
            boxes_array = np.array(player_boxes)
            scores_array = np.array(scores)

            # Convert to tensors for PyTorch
            boxes_tensor = torch.tensor(boxes_array, dtype=torch.float32, device=device)
            scores_tensor = torch.tensor(scores_array, dtype=torch.float32, device=device)

            # Apply Non-Maximum Suppression
            nms_indices = torch.ops.torchvision.nms(boxes_tensor, scores_tensor, 0.3)
            refined_boxes = boxes_tensor[nms_indices].cpu().numpy().astype(int)
            refined_scores = scores_tensor[nms_indices].cpu().numpy()

            # Debugging: Print refined boxes and scores
            print(f"Frame {frame_count}: {len(refined_boxes)} boxes after NMS")

            # Extract features using the deep learning model
            features = extract_features(frame, refined_boxes, feature_extraction_model)

            # Assign player IDs
            player_ids = player_tracker.assign_player_id(refined_boxes, features)

            # Track players using Kalman filters and update visualization
            tracked_boxes = []
            for player_id, box in zip(player_ids, refined_boxes):
                x1, y1, x2, y2 = box
                if player_id not in player_trackers:
                    player_trackers[player_id] = KalmanFilter()

                kalman_filter = player_trackers[player_id]
                corrected = kalman_filter.correct([x1, y1])
                predicted = kalman_filter.predict()
                tracked_box = (int(predicted[0]), int(predicted[1]), x2, y2)
                tracked_boxes.append(tracked_box)

                # Draw the bounding box and ID on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {player_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            poses = estimate_poses(frame, tracked_boxes)
            frame_with_poses = draw_poses(frame, tracked_boxes, poses)

            for player_id, feature, pose in zip(player_ids, features, poses):
                if feature is not None and pose is not None:
                    store_player_profile(player_id, feature)
                    store_in_db(frame_name, feature, pose, os.path.basename(video_path))

            cv2.imwrite(os.path.join(frames_directory, frame_name), frame_with_poses)
            cv2.imshow('Tracking and Pose Visualization', frame_with_poses)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    os.rename(video_path, os.path.join(processed_directory, os.path.basename(video_path)))

if __name__ == "__main__":
    video_directory = "../../raw_videos"
    processed_directory = "../../processed_videos"
    frames_directory = "../../frames"
    model, device = load_model()
    player_tracker = PlayerIDTracker()  # Ensure this is correctly initialized
    feature_extraction_model = load_deep_learning_model()  # Load the feature extraction model
    videos = [f for f in os.listdir(video_directory) if f.endswith(".mp4")]
    for video_file in tqdm(videos, desc="Processing Videos"):
        video_path = os.path.join(video_directory, video_file)
        process_single_video(video_path, model, device, processed_directory, frames_directory, player_tracker, feature_extraction_model)
