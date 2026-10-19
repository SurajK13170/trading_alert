import redis
import urllib.parse
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

load_dotenv()

def clean_env(name, default=None):
    value = os.getenv(name, default)
    if isinstance(value, str):
        return value.split("  #", 1)[0].strip()
    return value

# Build MySQL URL safely when the password contains special characters.
mysql_password = clean_env("MYSQL_PASSWORD", "")
encoded_password = urllib.parse.quote_plus(mysql_password)
mysql_user = clean_env("MYSQL_USER", "root")
mysql_host = clean_env("MYSQL_HOST", "localhost")
mysql_port = clean_env("MYSQL_PORT", "3306")
mysql_db = clean_env("MYSQL_DB") or clean_env("MYSQL_DATABASE")

MYSQL_URL = clean_env("MYSQL_URL")
if mysql_user and mysql_password and mysql_host and mysql_db:
    MYSQL_URL = f"mysql+pymysql://{mysql_user}:{encoded_password}@{mysql_host}:{mysql_port}/{mysql_db}"

if not MYSQL_URL:
    raise RuntimeError("MySQL configuration is missing. Set MYSQL_URL or MYSQL_* values in .env.")

engine = create_engine(MYSQL_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Redis Setup
# redis-py 4.6 defaults to Redis 5-compatible RESP2.
redis_client = redis.from_url(clean_env("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
