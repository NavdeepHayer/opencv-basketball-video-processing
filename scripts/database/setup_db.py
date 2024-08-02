import psycopg2
from psycopg2 import sql

def create_tables():
    try:
        # Connect to the openCVData database
        conn = psycopg2.connect(
            dbname="opencvdata", 
            user="opencvserver",         # Replace with your PostgreSQL username
            password="300715108Kai!", # Replace with your PostgreSQL password
            host="localhost"
        )
        cur = conn.cursor()
        
        # Create frames table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS frames (
                frame_id SERIAL PRIMARY KEY,
                frame_name VARCHAR(255) UNIQUE NOT NULL,
                video_name VARCHAR(255) NOT NULL,
                processed BOOLEAN DEFAULT FALSE
            );
        """)
        
        # Create features table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS features (
                feature_id SERIAL PRIMARY KEY,
                frame_name VARCHAR(255) REFERENCES frames(frame_name),
                feature_data JSONB,
                pose_data JSONB
            );
        """)

        conn.commit()
        cur.close()
        conn.close()
        print("Database tables created successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    create_tables()
