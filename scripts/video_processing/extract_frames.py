import cv2
import os

def extract_frames(video_path, output_dir):
    """
    Extracts frames from a video and saves them as images in the output directory.

    Args:
    - video_path (str): Path to the video file.
    - output_dir (str): Directory where extracted frames will be saved.

    Returns:
    - list: List of frame filenames.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frame_filenames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = f"frame_{frame_count}.jpg"
        frame_path = os.path.join(output_dir, frame_filename)
        cv2.imwrite(frame_path, frame)
        frame_filenames.append(frame_filename)
        frame_count += 1

    cap.release()
    return frame_filenames

if __name__ == "__main__":
    video_path = "../raw_videos/basketball1.mp4"
    output_dir = "../frames/sample_video"
    frames = extract_frames(video_path, output_dir)
    print(f"Extracted {len(frames)} frames.")
