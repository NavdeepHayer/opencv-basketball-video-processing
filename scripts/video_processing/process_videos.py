import os
import cv2
from tqdm import tqdm
from detect_players import load_model, detect_players
from extract_features import extract_features
from estimate_poses import estimate_poses
from store_in_db import store_in_db

def process_all_videos(video_directory, processed_directory):
    model, device = load_model()  # Ensure the model is loaded once
    videos = [f for f in os.listdir(video_directory) if f.endswith(".mp4")]
    for video_file in tqdm(videos, desc="Processing Videos"):
        video_path = os.path.join(video_directory, video_file)
        process_single_video(video_path, model, device, processed_directory)

def process_single_video(video_path, model, device, processed_directory):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_name = f"frame_{frame_count}.jpg"
        player_boxes = detect_players(frame, model, device)
        features = extract_features(frame, player_boxes)
        poses = estimate_poses(frame, player_boxes)
        
        # Display the frame with detected player boxes
        for box in player_boxes:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        cv2.imshow("Frame Preview", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        store_in_db(frame_name, features, poses, os.path.basename(video_path))
        frame_count += 1
    cap.release()
    cv2.destroyAllWindows()
    os.rename(video_path, os.path.join(processed_directory, os.path.basename(video_path)))

if __name__ == "__main__":
    process_all_videos("../../raw_videos", "../../processed_videos")
