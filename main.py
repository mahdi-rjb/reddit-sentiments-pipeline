import pandas as pd
import numpy as np
import mysql.connector
import requests
import json
import re
import time
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
import statsmodels.api as sm
from src.config.settings import *


class RedditSentimentAnalyzer:
    def __init__(self):
        """Initialize the analyzer with necessary components"""
        self.headers = REDDIT_HEADERS
        self.url = REDDIT_URL
        self.max_posts = MAX_POSTS
        self.sia = SentimentIntensityAnalyzer()

        # Create necessary directories
        os.makedirs('data/raw', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        os.makedirs(VISUALIZATION_PATH, exist_ok=True)
        os.makedirs(HYPOTHESIS_PATH, exist_ok=True)

    def scrape_reddit_data(self):
        """Scrape data from Reddit worldnews"""
        data = []
        post_count = 0
        url = self.url

        while url and post_count < self.max_posts:
            response = requests.get(url, headers=self.headers)
            if response.status_code != 200:
                print(f"Error: {response.status_code}")
                break

            json_data = response.json()
            posts = json_data['data']['children']

            for post in posts:
                if post_count >= self.max_posts:
                    break
                post_data = post['data']
                created_utc = datetime.utcfromtimestamp(post_data['created_utc'])

                if created_utc >= datetime.now() - timedelta(days=90):
                    data.append({
                        'title': post_data['title'],
                        'score': post_data['score'],
                        'comments': post_data['num_comments'],
                        'created_utc': created_utc,
                        'selftext': post_data['selftext'],
                        'url': post_data['url'],
                        'author': post_data['author'],
                        'upvote_ratio': post_data['upvote_ratio'],
                        'flair': post_data.get('link_flair_text', ''),
                        'num_awards': post_data.get('total_awards_received', 0),
                        'subreddit': post_data['subreddit']
                    })
                    post_count += 1

            after = json_data['data']['after']
            if after:
                url = f'https://www.reddit.com/r/worldnews/top/.json?t=year&after={after}'
            else:
                url = None

            time.sleep(2)

        df = pd.DataFrame(data)
        df.to_csv('src/data/raw/reddit_worldnews_raw.csv', index=False)
        return df

    def clean_data(self, df):

        """Clean the scraped data"""
        df.fillna({'selftext': '', 'flair': '', 'author': '', 'upvote_ratio': 0, 'num_awards': 0}, inplace=True)
        df['content'] = df['title'] + ' ' + df['selftext']
        df['content'] = df['content'].apply(lambda x: re.sub(r'[^A-Za-z\s]', '', x))
        df.to_csv('src/data/processed/reddit_worldnews_data_cleaned.csv', index=False)
        return df

    def preprocess_text(self, df):
        """Preprocess text data"""
        stop_words = set(stopwords.words('english'))

        def preprocess(text):
            tokens = word_tokenize(text.lower())
            tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
            return ' '.join(tokens)

        df['processed_content'] = df['content'].apply(preprocess)
        df.to_csv('src/data/processed/reddit_worldnews_data_preprocessed.csv', index=False)
        return df

    def perform_sentiment_analysis(self, df):
        """Perform sentiment analysis on processed text"""
        df['sentiment'] = df['processed_content'].apply(lambda x: self.sia.polarity_scores(x)['compound'])
        df['sentiment_label'] = df['sentiment'].apply(
            lambda x: 'positive' if x > 0.1 else ('negative' if x < -0.1 else 'neutral')
        )

        # Add time-based features
        df['created_date'] = pd.to_datetime(df['created_utc']).dt.date
        df['week_day'] = pd.to_datetime(df['created_utc']).dt.weekday
        df['month'] = pd.to_datetime(df['created_utc']).dt.month
        df['hour'] = pd.to_datetime(df['created_utc']).dt.hour

        # Add event-related features
        events_keywords = ['war', 'peace', 'election']
        df['event_related'] = df['processed_content'].apply(lambda x: any(keyword in x for keyword in events_keywords))

        # Add sentiment strength features
        df['extreme_sentiment'] = df['sentiment'].apply(lambda x: 'extreme' if abs(x) > 0.5 else 'moderate')
        df['strong_sentiment'] = df['sentiment'].apply(lambda x: 'strong' if abs(x) > 0.5 else 'weak')

        df.to_csv('src/data/processed/reddit_worldnews_data_wrangled.csv', index=False)
        return df

    def generate_visualizations(self, df):
        """Generate all visualizations"""
        # Basic sentiment visualizations
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x='sentiment_label', palette='coolwarm')
        plt.title('Sentiment Distribution')
        plt.savefig(f'{VISUALIZATION_PATH}/sentiment_distribution.png')
        plt.close()

        # Word cloud
        text = ' '.join(df['processed_content'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(f'{VISUALIZATION_PATH}/wordcloud.png')
        plt.close()

        # Time-based Sentiment Trends
        df['created_date'] = pd.to_datetime(df['created_utc']).dt.date
        sentiment_trends = df.groupby(['created_date', 'sentiment_label']).size().unstack(fill_value=0)
        sentiment_trends.plot(kind='line', figsize=(12, 6))
        plt.title('Sentiment Trends Over Time')
        plt.xlabel('Date')
        plt.ylabel('Number of Posts')
        plt.legend(title='Sentiment')
        plt.savefig(f'{VISUALIZATION_PATH}/sentiment_trends.png')
        plt.close()

        # Generate hypothesis visualizations
        self._generate_hypothesis_visualizations(df)

        # Generate question-specific visualizations
        self._generate_question_visualizations(df)

    def _generate_hypothesis_visualizations(self, df):
        """Generate visualizations for hypotheses"""
        # H1: Sentiment trends over time
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x='created_date', y='sentiment', marker='o')
        plt.title('Daily Sentiment Trends Over Time')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{HYPOTHESIS_PATH}/h1_sentiment_trends.png')
        plt.close()

        # H2: Sentiment and upvotes
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='sentiment_label', y='score')
        plt.title('Upvotes by Sentiment Category')
        plt.xlabel('Sentiment Label')
        plt.ylabel('Upvotes')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{HYPOTHESIS_PATH}/h2_upvotes_by_sentiment.png')
        plt.close()

        # H3: Keywords and sentiment
        keywords = ['war', 'peace', 'election']
        keyword_data = df[df['content'].str.contains('|'.join(keywords), case=False)]
        plt.figure(figsize=(10, 6))
        sns.countplot(data=keyword_data, x='sentiment_label', hue='sentiment_label', palette='viridis')
        plt.title('Sentiment Distribution for Posts Containing Keywords')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{HYPOTHESIS_PATH}/h3_keyword_sentiment_distribution.png')
        plt.close()

        # H4: Sentiment polarity and comments
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='strong_sentiment', y='comments')
        plt.title('Comments by Sentiment Polarity')
        plt.xlabel('Polarity Strength')
        plt.ylabel('Number of Comments')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{HYPOTHESIS_PATH}/h4_comments_by_sentiment_polarity.png')
        plt.close()

        # H5: Engagement by time of day
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='hour', y='score')
        plt.title('Engagement Levels by Hour of the Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Upvotes')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{HYPOTHESIS_PATH}/h5_engagement_by_hour.png')
        plt.close()

    def _generate_question_visualizations(self, df):
        """Generate visualizations for specific research questions"""
        # Q1: Sentiment trends over time (Event Days)
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x='created_date', y='sentiment', marker='o', label='Sentiment')
        plt.title('Daily Sentiment Trends Over Time')
        plt.xlabel('Date')
        plt.ylabel('Sentiment Score')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{HYPOTHESIS_PATH}/q1_sentiment_trends.png')
        plt.close()

        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=df, x='created_date', y='sentiment', hue='sentiment', palette='coolwarm')
        plt.title('Sentiment Trends on Event Days')
        plt.xlabel('Date')
        plt.ylabel('Sentiment Score')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{HYPOTHESIS_PATH}/q1_sentiment_event_days.png')
        plt.close()

        # Q2: Extreme Sentiment Categories and Upvotes
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='sentiment_label', y='score')
        plt.title('Upvotes by Sentiment Category')
        plt.xlabel('Sentiment Label')
        plt.ylabel('Upvotes')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{HYPOTHESIS_PATH}/q2_upvotes_sentiment.png')
        plt.close()

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='sentiment', y='score', hue='sentiment_label')
        plt.title('Relationship Between Sentiment and Upvotes')
        plt.xlabel('Sentiment Score')
        plt.ylabel('Upvotes')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{HYPOTHESIS_PATH}/q2_sentiment_vs_upvotes.png')
        plt.close()

        # Q3: Time of Day Analysis
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='hour', y='score')
        plt.title('Engagement Levels by Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Upvotes')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{HYPOTHESIS_PATH}/q3_hourly_engagement.png')
        plt.close()

        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x='hour', y='score', marker='o')
        plt.title('Engagement Patterns Throughout the Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Upvotes')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{HYPOTHESIS_PATH}/q3_engagement_patterns.png')
        plt.close()

        # Q4: Sentiment vs Comments
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='sentiment', y='comments', hue='sentiment_label')
        plt.title('Sentiment Score vs Comments')
        plt.xlabel('Sentiment Score')
        plt.ylabel('Number of Comments')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{HYPOTHESIS_PATH}/q4_sentiment_comments.png')
        plt.close()

        # Q5: Topic Analysis
        keywords = ['war', 'peace', 'election']
        keyword_data = df[df['content'].str.contains('|'.join(keywords), case=False)]
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=keyword_data, x='sentiment_label', y='score')
        plt.title('Engagement for Topic-Specific Posts')
        plt.xlabel('Sentiment Label')
        plt.ylabel('Upvotes')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{HYPOTHESIS_PATH}/q5_topic_engagement.png')
        plt.close()

        plt.figure(figsize=(10, 6))
        sns.countplot(data=keyword_data, x='sentiment_label', hue='sentiment_label', palette='viridis')
        plt.title('Sentiment Distribution for Keywords')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{HYPOTHESIS_PATH}/q5_keywords_sentiment_distribution.png')
        plt.close()

    def save_to_database(self, df):
        """Save processed data to MySQL database"""
        try:
            connection = mysql.connector.connect(**DB_CONFIG)
            cursor = connection.cursor()

            # Insert data into Authors table
            authors = df['author'].unique()
            for author in authors:
                if pd.notna(author):
                    cursor.execute("INSERT IGNORE INTO Authors (author_name) VALUES (%s)", (author,))

            # Insert data into Flairs table
            flairs = df['flair'].unique()
            for flair in flairs:
                if pd.notna(flair):
                    cursor.execute("INSERT IGNORE INTO Flairs (flair_name) VALUES (%s)", (flair,))

            # Insert sentiment labels
            sentiments = ['Positive', 'Negative', 'Neutral']
            for sentiment in sentiments:
                cursor.execute("INSERT IGNORE INTO SentimentLabels (label_name) VALUES (%s)", (sentiment,))

            # Insert sentiment levels including extreme and strong sentiment values
            levels = list(set(['Low', 'Medium', 'High', 'extreme', 'moderate', 'strong', 'weak']))
            for level in levels:
                cursor.execute("INSERT IGNORE INTO SentimentLevels (level_name) VALUES (%s)", (level,))

            # Insert data into Posts table
            for index, row in df.iterrows():
                if pd.notna(row['author']):
                    cursor.execute("SELECT author_id FROM Authors WHERE author_name = %s", (row['author'],))
                    author_id = cursor.fetchone()[0]

                if pd.notna(row['flair']):
                    cursor.execute("SELECT flair_id FROM Flairs WHERE flair_name = %s", (row['flair'],))
                    flair_id = cursor.fetchone()[0]
                else:
                    continue

                if pd.notna(row['sentiment_label']):
                    cursor.execute("SELECT label_id FROM SentimentLabels WHERE label_name = %s",
                                   (row['sentiment_label'],))
                    sentiment_label_id = cursor.fetchone()[0]

                # Get IDs for extreme and strong sentiment
                cursor.execute("SELECT level_id FROM SentimentLevels WHERE level_name = %s",
                               (row['extreme_sentiment'],))
                extreme_sentiment_id = cursor.fetchone()[0]

                cursor.execute("SELECT level_id FROM SentimentLevels WHERE level_name = %s", (row['strong_sentiment'],))
                strong_sentiment_id = cursor.fetchone()[0]

                cursor.execute("""
                    INSERT INTO Posts (
                        title, score, comments, created_utc, url, author_id, 
                        upvote_ratio, flair_id, num_awards, subreddit,
                        content, processed_content, sentiment, sentiment_label_id, 
                        created_date, week_day, month, event_related,
                        extreme_sentiment_id, strong_sentiment_id
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    row['title'], row['score'], row['comments'], row['created_utc'],
                    row['url'], author_id, row['upvote_ratio'], flair_id,
                    row['num_awards'], row['subreddit'], row['content'],
                    row['processed_content'], row['sentiment'], sentiment_label_id,
                    row['created_date'], row['week_day'], row['month'],
                    row['event_related'], extreme_sentiment_id, strong_sentiment_id
                ))

            connection.commit()
            print("Data successfully saved to database")

        except Exception as e:
            print(f"Error saving to database: {e}")
            if connection.is_connected():
                connection.rollback()

        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
def main():
    analyzer = RedditSentimentAnalyzer()

    # Execute pipeline
    df = analyzer.scrape_reddit_data()
    print("Data scraped successfully")

    df = analyzer.clean_data(df)
    print("Data cleaned successfully")

    df = analyzer.preprocess_text(df)
    print("Text preprocessing completed")

    df = analyzer.perform_sentiment_analysis(df)
    print("Sentiment analysis completed")

    analyzer.generate_visualizations(df)
    print("Visualizations generated")

    analyzer.save_to_database(df)
    print("Data saved to database")

if __name__ == "__main__":
    main()