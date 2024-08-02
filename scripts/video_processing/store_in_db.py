import psycopg2
import json

def store_in_db(frame_name, features, poses, video_name):
    """Store features and poses data in the database."""
    # Connect to the PostgreSQL database
    conn = psycopg2.connect("dbname=opencvdata user=opencvserver password=300715108Kai! host=localhost")
    cur = conn.cursor()

    # Insert frame metadata into the frames table
    cur.execute("""
        INSERT INTO frames (frame_name, video_name, processed)
        VALUES (%s, %s, %s)
        ON CONFLICT (frame_name) DO NOTHING;
    """, (frame_name, video_name, True))

    # Insert feature and pose data into the features table
    for feature, pose in zip(features, poses):
        cur.execute("""
            INSERT INTO features (frame_name, feature_data, pose_data)
            VALUES (%s, %s, %s);
        """, (frame_name, json.dumps(feature.tolist()), json.dumps(pose)))

    # Commit the transaction and close the connection
    conn.commit()
    cur.close()
    conn.close()

if __name__ == "__main__":
    # Example usage
    store_in_db("frame_0.jpg", [], [], "video_name")
