from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
import pandas as pd
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from main import RedditSentimentAnalyzer

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 9),
    'email': ['mr.zac94@gmail.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def scrape_reddit_data(**context):
    analyzer = RedditSentimentAnalyzer()
    df = analyzer.scrape_reddit_data()
    df.to_csv('data/raw/temp_scraped_data.csv', index=False)
    return "Data scraped successfully"

def clean_data(**context):
    analyzer = RedditSentimentAnalyzer()
    df = pd.read_csv('data/raw/temp_scraped_data.csv')
    cleaned_df = analyzer.clean_data(df)
    return "Data cleaned successfully"

def preprocess_text(**context):
    analyzer = RedditSentimentAnalyzer()
    df = pd.read_csv('data/processed/reddit_worldnews_data_cleaned.csv')
    preprocessed_df = analyzer.preprocess_text(df)
    return "Text preprocessing completed"

def perform_sentiment_analysis(**context):
    analyzer = RedditSentimentAnalyzer()
    df = pd.read_csv('data/processed/reddit_worldnews_data_preprocessed.csv')
    sentiment_df = analyzer.perform_sentiment_analysis(df)
    return "Sentiment analysis completed"

def generate_visualizations(**context):
    analyzer = RedditSentimentAnalyzer()
    df = pd.read_csv('data/processed/reddit_worldnews_data_wrangled.csv')
    analyzer.generate_visualizations(df)
    return "Visualizations generated"

def save_to_database(**context):
    analyzer = RedditSentimentAnalyzer()
    df = pd.read_csv('data/processed/reddit_worldnews_data_wrangled.csv')
    analyzer.save_to_database(df)
    return "Data saved to database"

with DAG(
    'reddit_sentiment_analysis',
    default_args=default_args,
    description='Reddit Sentiment Analysis Pipeline',
    schedule_interval='0 0 * * *',  # Run daily at midnight
    catchup=False
) as dag:

    # Define tasks
    scrape_task = PythonOperator(
        task_id='scrape_reddit_data',
        python_callable=scrape_reddit_data
    )

    clean_task = PythonOperator(
        task_id='clean_data',
        python_callable=clean_data
    )

    preprocess_task = PythonOperator(
        task_id='preprocess_text',
        python_callable=preprocess_text
    )

    sentiment_task = PythonOperator(
        task_id='perform_sentiment_analysis',
        python_callable=perform_sentiment_analysis
    )

    visualization_task = PythonOperator(
        task_id='generate_visualizations',
        python_callable=generate_visualizations
    )

    database_task = PythonOperator(
        task_id='save_to_database',
        python_callable=save_to_database
    )

    # Set task dependencies
    scrape_task >> clean_task >> preprocess_task >> sentiment_task >> visualization_task >> database_task