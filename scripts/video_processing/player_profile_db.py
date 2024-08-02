import psycopg2
import json

def store_player_profile(player_id, feature_vector):
    conn = psycopg2.connect("dbname=opencvdata user=opencvserver password=300715108Kai! host=localhost")
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO player_profiles (player_id, feature_vector)
        VALUES (%s, %s)
        ON CONFLICT (player_id) DO UPDATE
        SET feature_vector = EXCLUDED.feature_vector;
    """, (player_id, json.dumps(feature_vector.tolist())))

    conn.commit()
    cur.close()
    conn.close()
