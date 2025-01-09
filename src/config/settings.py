# settings.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Reddit API Configuration
REDDIT_HEADERS = {'User-Agent': 'Mozilla/5.0'}
REDDIT_URL = 'https://www.reddit.com/r/worldnews/top/.json?t=year'
MAX_POSTS = 700
TIME_PERIOD_DAYS = 90  # Last 3 months

# MySQL Configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', 'root'),
    'database': os.getenv('DB_NAME', 'reddit_sentiment')
}

# Visualization Settings
VISUALIZATION_PATH = 'src/visualizations'
HYPOTHESIS_PATH = 'src/questions_and_hypothesis'