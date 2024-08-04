import cv2
import os
import torch
import numpy as np
from tqdm import tqdm
from detect_players import load_model as load_detection_model, detect_players
from extract_features import extract_features, load_deep_learning_model
from estimate_poses import estimate_poses
from store_in_db import store_in_db
from player_profile_db import store_player_profile
from kalman_filter_tracking import KalmanFilter
from player_id_tracker import PlayerIDTracker
from trajectory_predictor import TrajectoryPredictor, load_model as load_lstm_model
from torch.cuda.amp import autocast
import torch.profiler

def draw_poses(frame, player_boxes, poses):
    for box, pose in zip(player_boxes, poses):
        if pose is not None:
            for landmark in pose:
                x = int(landmark['x'] * (box[2] - box[0]) + box[0])
                y = int(landmark['y'] * (box[3] - box[1]) + box[1])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    return frame

def pose_similarity(pose1, pose2, threshold=0.5):
    if pose1 is None or pose2 is None:
        return False

    num_points = min(len(pose1), len(pose2))
    if num_points == 0:
        return False

    distances = [np.linalg.norm(np.array([p1['x'], p1['y']]) - np.array([p2['x'], p2['y']])) for p1, p2 in zip(pose1, pose2)]
    average_distance = np.mean(distances)
    
    return average_distance < threshold

def predict_future_position(model, trajectory):
    model.eval()
    with torch.no_grad():
        trajectory = torch.tensor(trajectory, dtype=torch.float32).unsqueeze(0).to(next(model.parameters()).device)
        predicted_position = model(trajectory).cpu().numpy().flatten()
    return predicted_position

def collect_trajectory_data(trajectories, filename='trajectory_data.npy'):
    data = []
    for player_id, trajectory in trajectories.items():
        if len(trajectory) >= 10:
            data.append(trajectory[-10:])

    with open(filename, 'ab') as f:
        np.save(f, np.array(data))

def process_single_video(video_path, model, device, processed_directory, frames_directory, player_tracker, feature_extraction_model, lstm_model=None):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    player_trackers = {}
    trajectories = {}
    collected_data = []

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True
    ) as prof:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_name = f"frame_{frame_count}.jpg"
            player_boxes, scores = detect_players(frame, model, device)

            if player_boxes and scores:
                boxes_array = np.array(player_boxes)
                scores_array = np.array(scores)

                boxes_tensor = torch.tensor(boxes_array, dtype=torch.float32, device=device)
                scores_tensor = torch.tensor(scores_array, dtype=torch.float32, device=device)

                nms_indices = torch.ops.torchvision.nms(boxes_tensor, scores_tensor, 0.3)
                refined_boxes = boxes_tensor[nms_indices].cpu().numpy().astype(int)
                refined_scores = scores_tensor[nms_indices].cpu().numpy()

                # Use autocast for mixed precision inference
                with autocast():
                    features = extract_features(frame, refined_boxes, feature_extraction_model, device)
                    poses = estimate_poses(frame, refined_boxes)

                player_ids = player_tracker.assign_player_id(refined_boxes, features)

                for i, player_id in enumerate(player_ids):
                    if player_id not in trajectories:
                        trajectories[player_id] = []

                    x1, y1, x2, y2 = refined_boxes[i]
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    trajectories[player_id].append([center_x, center_y])

                    if len(trajectories[player_id]) > 1:
                        collected_data.append((trajectories[player_id][-2], [center_x, center_y]))

                    if len(trajectories[player_id]) > 10:
                        trajectories[player_id].pop(0)

                    if lstm_model and len(trajectories[player_id]) == 10:
                        predicted_position = predict_future_position(lstm_model, trajectories[player_id])
                        print(f"Predicted next position for Player {player_id}: {predicted_position}")

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

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {player_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                frame_with_poses = draw_poses(frame, refined_boxes, poses)

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

    collect_trajectory_data(trajectories)

    # Print profiling results
    print(prof.key_averages().table(sort_by="cuda_time_total"))

if __name__ == "__main__":
    video_directory = "../../raw_videos"
    processed_directory = "../../processed_videos"
    frames_directory = "../../frames"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model, _ = load_detection_model(weights_path=None, device=device)

    player_tracker = PlayerIDTracker()
    feature_extraction_model = load_deep_learning_model().to(device)

    lstm_weights_path = 'trajectory_predictor.pth'
    lstm_model = None
    if os.path.exists(lstm_weights_path):
        lstm_model = TrajectoryPredictor().to(device)
        lstm_model = load_lstm_model(lstm_model, lstm_weights_path)

    videos = [f for f in os.listdir(video_directory) if f.endswith(".mp4")]
    for video_file in tqdm(videos, desc="Processing Videos"):
        video_path = os.path.join(video_directory, video_file)
        process_single_video(video_path, model, device, processed_directory, frames_directory, player_tracker, feature_extraction_model, lstm_model)

