import psycopg2
import json

def store_in_db(frame_name, features, poses, video_name):
    conn = psycopg2.connect("dbname=opencvdata user=opencvserver password=300715108Kai! host=localhost")
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO frames (frame_name, video_name, processed)
        VALUES (%s, %s, %s)
        ON CONFLICT (frame_name) DO NOTHING;
    """, (frame_name, video_name, True))

    for feature, pose in zip(features, poses):
        if feature is None or pose is None:
            print(f"Skipping storing data for frame {frame_name} due to missing features or poses.")
            continue

        cur.execute("""
            INSERT INTO features (frame_name, feature_data, pose_data)
            VALUES (%s, %s, %s);
        """, (frame_name, json.dumps(feature.tolist()), json.dumps(pose)))

    conn.commit()
    cur.close()
    conn.close()
