def store_in_db(frame_name, features, poses, video_name):
    conn = psycopg2.connect("dbname=opencvdata user=opencvserver password=300715108Kai! host=localhost")
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO frames (frame_name, video_name, processed)
        VALUES (%s, %s, True)
        ON CONFLICT (frame_name) DO NOTHING;
    """, (frame_name, video_name))

    for feature, pose in zip(features, poses):
        feature_data = json.dumps([f.tolist() for f in feature])  # Ensure serialization
        pose_data = json.dumps({'landmarks': str(pose)})  # Simplify pose data for storage
        cur.execute("""
            INSERT INTO features (frame_name, feature_data, pose_data)
            VALUES (%s, %s, %s);
        """, (frame_name, feature_data, pose_data))

    conn.commit()
    cur.close()
    conn.close()
