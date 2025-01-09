# dags/reddit_sentiment_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from reddit_sentiment_tasks import (
    scrape_reddit_data,
    clean_data,
    preprocess_text,
    perform_sentiment_analysis,
    generate_visualizations,
    save_to_database
)

# Define default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 9),
    'email': ['your_email@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

# Create DAG
dag = DAG(
    'reddit_sentiment_pipeline',
    default_args=default_args,
    description='Reddit Sentiment Analysis Pipeline',
    schedule_interval='0 0 * * *',  # Run daily at midnight
    catchup=False
)

# Define tasks
t1_scrape = PythonOperator(
    task_id='scrape_reddit_data',
    python_callable=scrape_reddit_data,
    dag=dag
)

t2_clean = PythonOperator(
    task_id='clean_data',
    python_callable=clean_data,
    dag=dag
)

t3_preprocess = PythonOperator(
    task_id='preprocess_text',
    python_callable=preprocess_text,
    dag=dag
)

t4_sentiment = PythonOperator(
    task_id='perform_sentiment_analysis',
    python_callable=perform_sentiment_analysis,
    dag=dag
)

t5_visualize = PythonOperator(
    task_id='generate_visualizations',
    python_callable=generate_visualizations,
    dag=dag
)

t6_save = PythonOperator(
    task_id='save_to_database',
    python_callable=save_to_database,
    dag=dag
)

# Set task dependencies
t1_scrape >> t2_clean >> t3_preprocess >> t4_sentiment >> t5_visualize >> t6_save