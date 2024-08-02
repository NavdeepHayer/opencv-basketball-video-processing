import os
import cv2
from tqdm import tqdm
from detect_players import load_model, detect_players
from extract_features import extract_features
from estimate_poses import estimate_poses
from store_in_db import store_in_db
from motion_analysis import PlayerTracker
from player_profile_db import store_player_profile

def process_all_videos(video_directory, processed_directory, frames_directory):
    model, device = load_model()  # Ensure the model is loaded once
    tracker = PlayerTracker()
    videos = [f for f in os.listdir(video_directory) if f.endswith(".mp4")]
    for video_file in tqdm(videos, desc="Processing Videos"):
        video_path = os.path.join(video_directory, video_file)
        process_single_video(video_path, model, device, processed_directory, frames_directory, tracker)

def process_single_video(video_path, model, device, processed_directory, frames_directory, tracker):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_name = f"frame_{frame_count}.jpg"
        player_boxes = detect_players(frame, model, device)

        # Update player IDs for each detected box using the tracker
        tracked_boxes, player_ids = tracker.update(player_boxes)
        
        features = extract_features(frame, tracked_boxes)
        poses = estimate_poses(frame, tracked_boxes)

        # Store each player's profile
        for player_id, feature, pose in zip(player_ids, features, poses):
            if feature is not None and pose is not None:
                store_player_profile(player_id, feature)
                store_in_db(frame_name, feature, pose, os.path.basename(video_path))
            else:
                print(f"Skipping storing data for frame {frame_name} due to missing features or poses.")

        frame_count += 1

        # Optionally, visualize tracking
        for (x1, y1, x2, y2), player_id in zip(tracked_boxes, player_ids):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {player_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    os.rename(video_path, os.path.join(processed_directory, os.path.basename(video_path)))

if __name__ == "__main__":
    process_all_videos("../../raw_videos", "../../processed_videos", "../../frames")
