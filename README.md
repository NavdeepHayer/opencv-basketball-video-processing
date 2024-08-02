# Basketball Video Processing Project

This project is designed to process basketball game videos to extract frames, detect players, estimate poses, and store feature data for player re-identification.

## Directory Structure

- **raw_videos/**: Contains unprocessed videos.
- **processed_videos/**: Contains videos that have been processed.
- **frames/**: Stores extracted frames from videos.
- **features/**: Contains extracted feature points and pose data.
- **db/**: Contains database scripts and configurations.
- **scripts/**: 
  - **video_processing/**: Scripts for processing videos.
  - **model_training/**: Scripts for training and evaluating machine learning models.
  - **database/**: Scripts for database setup and management.
- **config/**: Configuration files for the project.
- **logs/**: Directory for log files.

## Scripts Overview

### Video Processing
- **process_videos.py**: Orchestrates the video processing pipeline.
- **extract_frames.py**: Extracts frames from videos.
- **detect_players.py**: Detects players and draws bounding boxes.
- **extract_features.py**: Extracts feature points from frames.
- **estimate_poses.py**: Estimates player poses using pose estimation models.
- **store_in_db.py**: Stores extracted data in the PostgreSQL database.

### Model Training
- **train_reid_model.py**: Script to train the re-identification model.
- **evaluate_model.py**: Script to evaluate the performance of the trained model.

### Database
- **setup_db.py**: Initializes the PostgreSQL database and creates necessary tables.
- **manage_data.py**: Manages data entry, updates, and queries within the database.

