import os
import tweepy
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

def extract_crisis_data(keywords, days_back=7, max_tweets=1000):
    """
    Extract crisis-related tweets based on keywords
    """

    bearer_token = os.getenv("BEARER_TOKEN")
    client = tweepy.Client(bearer_token=bearer_token)

    query = " OR ".join([f'"{keyword}"' for keyword in keywords])
    query += " -is:retweet lang:en"

    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days_back)

    tweets = []
    for tweet in tweepy.Paginator(client.search_recent_tweets, 
                                 query=query,
                                 start_time=start_time,
                                 end_time=end_time,
                                 tweet_fields=['created_at', 'public_metrics', 'geo'],
                                 max_results=100).flatten(limit=max_tweets):
        tweets.append({
            'id': tweet.id,
            'text': tweet.text,
            'created_at': tweet.created_at,
            'likes': tweet.public_metrics['like_count'],
            'retweets': tweet.public_metrics['retweet_count'],
            'replies': tweet.public_metrics['reply_count'],
            'geo': tweet.geo if hasattr(tweet, 'geo') else None
        })
    
    return pd.DataFrame(tweets)

if __name__ == "__main__":
    keywords = [
        "feeling depressed", "addiction help", "overwhelmed", "suicidal thoughts", 
        "mental health struggle", "anxiety attack", "severe stress", "feeling lonely",
        "hopeless", "self-harm", "don't want to live", "overdose", "need therapy",
        "panic attack", "can't go on"
    ]

    data = extract_crisis_data(keywords)
    
    data.to_csv("crisis_posts_raw.csv", index=False)
    print(f"Extracted {len(data)} crisis-related posts and saved to crisis_posts_raw.csv")
