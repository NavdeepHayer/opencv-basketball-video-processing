import cv2
import os
import torch
import numpy as np
from tqdm import tqdm
import logging
from sklearn.model_selection import train_test_split
from datetime import datetime
from detect_players import load_model, detect_players
from extract_features import extract_features, load_deep_learning_model
from estimate_poses import estimate_poses
from kalman_filter_tracking import KalmanFilter
from player_id_tracker import PlayerIDTracker
from trajectory_predictor import load_lstm_model, predict_future_position

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def process_single_video(video_path, model, device, processed_directory, frame_subdirectory, player_tracker, feature_extraction_model, lstm_model, collect_data=True):
    logging.info(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    player_trackers = {}
    trajectories = {}
    collected_data = []
    pseudo_labeled_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logging.info("End of video reached or unable to read frame.")
            break

        frame_name = f"frame_{frame_count}.jpg"
        player_boxes, scores = detect_players(frame, model, device)

        if player_boxes and scores:
            logging.info(f"Detected {len(player_boxes)} players in frame {frame_count}.")
            boxes_array = np.array(player_boxes)
            scores_array = np.array(scores)

            boxes_tensor = torch.tensor(boxes_array, dtype=torch.float32, device=device)
            scores_tensor = torch.tensor(scores_array, dtype=torch.float32, device=device)

            nms_indices = torch.ops.torchvision.nms(boxes_tensor, scores_tensor, 0.3)
            refined_boxes = boxes_tensor[nms_indices].cpu().numpy().astype(int)
            refined_scores = scores_tensor[nms_indices].cpu().numpy()

            features = extract_features(frame, refined_boxes, feature_extraction_model, device)
            poses = estimate_poses(frame, refined_boxes)

            player_ids = player_tracker.assign_player_id(refined_boxes, features)

            for i, player_id in enumerate(player_ids):
                if player_id not in trajectories:
                    trajectories[player_id] = []

                x1, y1, x2, y2 = refined_boxes[i]
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1

                # Safely extract feature points
                feature_points = []
                if poses[i] is not None:
                    feature_points = [coord for fp in poses[i] if fp is not None for coord in (fp['x'], fp['y'])]

                # Add height, width, and feature points to the feature vector
                combined_data = [center_x, center_y, width, height] + features[i].tolist() + feature_points
                trajectories[player_id].append(combined_data)

                if collect_data:
                    if len(trajectories[player_id]) > 1:
                        collected_data.append((trajectories[player_id][-2], combined_data))

                if len(trajectories[player_id]) > 10:
                    trajectories[player_id].pop(0)

                if lstm_model is not None and len(trajectories[player_id]) == 10:
                    predicted_position = predict_future_position(lstm_model, trajectories[player_id])
                    logging.info(f"Predicted next position for Player {player_id}: {predicted_position}")

                    # Assuming a confidence threshold of 0.7
                    confidence = refined_scores[i]  # Use model's score for confidence
                    if confidence > 0.7:
                        pseudo_labeled_data.append((combined_data, predicted_position))

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

            # Save the frame in the subdirectory based on mode
            cv2.imwrite(os.path.join(frame_subdirectory, frame_name), frame_with_poses)
            cv2.imshow('Tracking and Pose Visualization', frame_with_poses)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1

        if collect_data and frame_count % 1000 == 0 and collected_data:
            with open('trajectory_data_with_features.npy', 'ab') as f:
                np.save(f, np.array(collected_data))
            collected_data.clear()

    cap.release()
    cv2.destroyAllWindows()
    os.rename(video_path, os.path.join(processed_directory, os.path.basename(video_path)))

    # Add pseudo-labeled data to the dataset
    if pseudo_labeled_data:
        with open('trajectory_data_with_features.npy', 'ab') as f:
            np.save(f, np.array(pseudo_labeled_data))

    if collect_data and collected_data:
        with open('trajectory_data_with_features.npy', 'ab') as f:
            np.save(f, np.array(collected_data))

def determine_input_size(data_filename):
    """ Determine the input size based on the saved training data. """
    if not os.path.exists(data_filename):
        raise FileNotFoundError(f"Data file {data_filename} not found.")
    
    data = np.load(data_filename, allow_pickle=True)
    # Assuming data is not empty and consistent
    sample_sequence = data[0]
    return len(sample_sequence[0])  # Length of one input sequence element

def load_existing_data(file_path):
    """ Load existing data from a file if it exists. """
    if os.path.exists(file_path):
        data = np.load(file_path, allow_pickle=True)
        X_existing = np.array([d[0] for d in data])
        y_existing = np.array([d[1] for d in data])
        return X_existing, y_existing
    else:
        return None, None

def merge_data(new_data, new_labels, existing_data, existing_labels):
    """ Combine new and existing data. """
    if existing_data is not None and existing_labels is not None:
        X_combined = np.concatenate((new_data, existing_data), axis=0)
        y_combined = np.concatenate((new_labels, existing_labels), axis=0)
    else:
        X_combined, y_combined = new_data, new_labels
    return X_combined, y_combined

def load_and_prepare_data():
    """ Simulate loading new data for demonstration purposes. """
    data = np.random.randn(1000, 10, 2)  # 1000 samples, 10 time steps, 2 features per step
    labels = np.random.randn(1000, 2)    # 1000 labels, 2 coordinates per label
    return train_test_split(data, labels, test_size=0.2, random_state=42)

def select_model(trained_models_path):
    """ Provide options to select a model or use the latest one. """
    trained_models = sorted(os.listdir(trained_models_path), reverse=True)
    
    print("Available trained models:")
    for idx, model_date in enumerate(trained_models, start=1):
        print(f"{idx}: {model_date}")
    print(f"{len(trained_models) + 1}: Use the latest model ({trained_models[0]})")

    model_choice = int(input("Select a model to use: "))
    if model_choice == len(trained_models) + 1:
        return trained_models[0]  # Use the latest model
    else:
        return trained_models[model_choice - 1]

if __name__ == "__main__":
    video_directory = "../../raw_videos"
    processed_directory = "../../processed_videos"
    frames_directory = "../../frames"
    model, device = load_model(weights_path=None, device='cuda' if torch.cuda.is_available() else 'cpu')
    player_tracker = PlayerIDTracker()
    feature_extraction_model = load_deep_learning_model(device=device)

    mode = input("Select mode: (1) Record new data, (2) Use trained model, (3) Add new data to trained dataset: ")
    
    if mode == "1":
        lstm_model = None
        collect_data = True
        # Subdirectory for raw feature extraction
        mode_directory = os.path.join(frames_directory, "raw_feature_extraction")
    elif mode in ["2", "3"]:
        # Select trained model
        trained_models_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir, 'trained_data'))
        raw_data_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir, 'raw_training_data'))
        
        # Model selection with option for the latest model
        model_date = select_model(trained_models_path)
        model_path = os.path.join(trained_models_path, model_date, 'enhanced_trajectory_predictor.pth')

        # Determine input size based on training data
        trajectory_data_filename = os.path.join(raw_data_path, model_date, 'trajectory_data_with_features.npy')
        input_size = determine_input_size(trajectory_data_filename)
        lstm_model = load_lstm_model(model_path, input_size=input_size, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        if mode == "2":
            collect_data = False
            # Subdirectory for AI trained data
            mode_directory = os.path.join(frames_directory, "AI_trained")
        elif mode == "3":
            collect_data = True
            # Subdirectory for AI trained added to raw
            mode_directory = os.path.join(frames_directory, "AI_trained_added_to_raw")

            # Load new data
            X_new, X_test, y_new, y_test = load_and_prepare_data()

            # Load existing data
            existing_data_path = os.path.join(os.getcwd(), 'trajectory_data_with_features.npy')
            X_existing, y_existing = load_existing_data(existing_data_path)

            # Merge new and existing data
            X_combined, y_combined = merge_data(X_new, y_new, X_existing, y_existing)

            # Save combined data
            combined_data = np.array(list(zip(X_combined, y_combined)))
            np.save(existing_data_path, combined_data)
    else:
        print("Invalid mode selected.")
        exit()

    # Create the mode-specific directory if it doesn't exist
    os.makedirs(mode_directory, exist_ok=True)

    videos = [f for f in os.listdir(video_directory) if f.endswith(".mp4")]
    for video_file in tqdm(videos, desc="Processing Videos"):
        video_path = os.path.join(video_directory, video_file)

        # Create a subdirectory for frames based on the video name and date within the mode directory
        video_name = os.path.splitext(video_file)[0]
        today_date = datetime.now().strftime('%Y-%m-%d')
        frame_subdirectory = os.path.join(mode_directory, f"{video_name}_{today_date}")
        os.makedirs(frame_subdirectory, exist_ok=True)

        process_single_video(video_path, model, device, processed_directory, frame_subdirectory, player_tracker, feature_extraction_model, lstm_model, collect_data)

