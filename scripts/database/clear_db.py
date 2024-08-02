import psycopg2

def clear_database():
    try:
        # Connect to the openCVData database
        conn = psycopg2.connect(
            dbname="opencvdata", 
            user="opencvserver",         # Replace with your PostgreSQL username
            password="300715108Kai!", # Replace with your PostgreSQL password
            host="localhost"
        )
        cur = conn.cursor()
        
        # Clear the tables
        cur.execute("TRUNCATE TABLE features RESTART IDENTITY CASCADE;")
        cur.execute("TRUNCATE TABLE frames RESTART IDENTITY CASCADE;")
        
        conn.commit()
        cur.close()
        conn.close()
        print("Database tables cleared successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    clear_database()
