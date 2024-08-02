import os
import cv2
from tqdm import tqdm
from detect_players import load_model, detect_players
from extract_features import extract_features
from estimate_poses import estimate_poses
from store_in_db import store_in_db

def process_all_videos(video_directory, processed_directory, frames_directory):
    model, device = load_model()  # Ensure the model is loaded once
    videos = [f for f in os.listdir(video_directory) if f.endswith(".mp4")]
    for video_file in tqdm(videos, desc="Processing Videos"):
        video_path = os.path.join(video_directory, video_file)
        process_single_video(video_path, model, device, processed_directory, frames_directory)

def process_single_video(video_path, model, device, processed_directory, frames_directory):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    video_name = os.path.basename(video_path).split('.')[0]
    os.makedirs(os.path.join(frames_directory, video_name), exist_ok=True)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_name = f"frame_{frame_count}.jpg"
        player_boxes = detect_players(frame, model, device)
        features = extract_features(frame, player_boxes)
        poses = estimate_poses(frame, player_boxes)

        # Draw bounding boxes and pose landmarks
        for box, pose in zip(player_boxes, poses):
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box
            if pose is not None:
                for landmark in pose:  # Iterate over each landmark in the list
                    # Access coordinates from the dictionary
                    px = int(landmark['x'] * w) + x
                    py = int(landmark['y'] * h) + y
                    cv2.circle(frame, (px, py), 2, (0, 0, 255), -1)  # Draw pose landmarks

        # Save the frame with annotations
        frame_path = os.path.join(frames_directory, video_name, frame_name)
        cv2.imwrite(frame_path, frame)

        # Store data in the database
        store_in_db(frame_name, features, poses, video_name)
        frame_count += 1

    cap.release()
    os.rename(video_path, os.path.join(processed_directory, os.path.basename(video_path)))

if __name__ == "__main__":
    process_all_videos("../../raw_videos", "../../processed_videos", "../../frames")
