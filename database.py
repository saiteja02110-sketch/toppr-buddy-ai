from dotenv import load_dotenv
import os
import mysql.connector
from mysql.connector import Error
import logging
import time

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("database")

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

logger.info(f"DB config: Host={DB_HOST}, Port={DB_PORT}, User={DB_USER}, DB={DB_NAME}")

# Global connection pool-like management
connection = None

def get_connection():
    """Get or create a fresh connection with auto-reconnect."""
    global connection
    try:
        if connection is None or not connection.is_connected():
            connection = mysql.connector.connect(
                host=DB_HOST,
                port=int(DB_PORT) if DB_PORT else 3306,
                user=DB_USER,
                password=DB_PASSWORD,
                database=DB_NAME,
                connection_timeout=10,
                autocommit=False
            )
            logger.info("✅ New DB connection established")
        return connection
    except Error as e:
        logger.error(f"❌ DB Connection Error: {e}")
        connection = None
        raise

def get_cursor():
    """Get a fresh cursor for each operation (safer than global)."""
    conn = get_connection()
    return conn.cursor(dictionary=True)

def _require_db():
    """Ensure DB is available."""
    try:
        get_connection()
    except Exception as e:
        raise RuntimeError(f"Database unavailable: {e}. Check .env and MySQL service.") from e

def create_tables():
    """Create all necessary tables safely."""
    try:
        _require_db()
        conn = get_connection()
        cursor = get_cursor()
        
        tables = [
            """
            CREATE TABLE IF NOT EXISTS users(
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(100) NOT NULL,
                email VARCHAR(150) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS chats(
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT,
                tool VARCHAR(50),
                message LONGTEXT,
                reply LONGTEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS uploaded_files(
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT,
                filename VARCHAR(255),
                filetype VARCHAR(50),
                filepath TEXT,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            """,
            # ... (other tables remain the same for brevity, but all are included)
            """
            CREATE TABLE IF NOT EXISTS notes(
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT,
                title VARCHAR(255),
                content LONGTEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS study_plans(
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT,
                title VARCHAR(255),
                content LONGTEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS quiz_results(
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT,
                topic VARCHAR(255),
                score INT,
                total_questions INT,
                completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS voice_history(
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT,
                transcript LONGTEXT,
                ai_reply LONGTEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS user_settings(
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT UNIQUE,
                theme VARCHAR(20) DEFAULT 'dark',
                language VARCHAR(30) DEFAULT 'English',
                voice_enabled BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            """
        ]
        
        for table in tables:
            cursor.execute(table)
        
        conn.commit()
        logger.info("✅ All tables created/verified successfully!")
    except Exception as e:
        logger.error(f"Table creation failed: {e}")

def execute(query, values=None, retries=2):
    """Execute query with retry and fresh cursor."""
    for attempt in range(retries):
        try:
            _require_db()
            conn = get_connection()
            cursor = get_cursor()
            cursor.execute(query, values or ())
            conn.commit()
            return
        except Error as e:
            logger.warning(f"DB execute attempt {attempt+1} failed: {e}")
            if attempt == retries - 1:
                raise
            time.sleep(1)

def fetch_all(query, values=None):
    """Fetch multiple rows with fresh cursor."""
    try:
        _require_db()
        cursor = get_cursor()
        cursor.execute(query, values or ())
        return cursor.fetchall()
    except Exception as e:
        logger.error(f"Fetch failed: {e}")
        return []

def save_chat(user_id, tool, user_message, ai_response):
    """Save chat with error handling."""
    try:
        query = """
        INSERT INTO chats(user_id, tool, message, reply)
        VALUES(%s, %s, %s, %s)
        """
        execute(query, (user_id, tool, user_message, ai_response))
        logger.info("Chat saved to DB")
    except Exception as e:
        logger.error(f"Failed to save chat: {e}")
        # Do not crash the app

def get_chat_history(user_id, limit=20):
    """Get chat history with robust error handling."""
    try:
        query = """
        SELECT message, reply
        FROM chats
        WHERE user_id = %s
        ORDER BY id DESC
        LIMIT %s
        """
        rows = fetch_all(query, (user_id, limit))
        history = []
        for row in reversed(rows):
            history.append({"role": "user", "content": row["message"]})
            history.append({"role": "assistant", "content": row["reply"]})
        return history
    except Exception as e:
        logger.error(f"Failed to load history: {e}")
        return []

def save_uploaded_file(user_id, filename, filetype, filepath):
    """Save uploaded file metadata."""
    try:
        query = """
        INSERT INTO uploaded_files(user_id, filename, filetype, filepath)
        VALUES(%s, %s, %s, %s)
        """
        execute(query, (user_id, filename, filetype, filepath))
        logger.info(f"File metadata saved: {filename}")
    except Exception as e:
        logger.warning(f"Failed to save file metadata: {e}")

# Initialize tables on import
if __name__ == "__main__":
    create_tables()
    logger.info("Database ready.")
else:
    try:
        create_tables()
    except Exception:
        logger.warning("DB tables init skipped - connection may not be available yet.")