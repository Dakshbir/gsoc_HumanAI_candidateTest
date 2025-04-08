import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text):
        """Clean and normalize text for NLP processing"""
        text = text.lower()

        text = re.sub(r'http\S+', '', text)

        text = re.sub(r'@\w+', '', text)

        text = re.sub(r'#(\w+)', r'\1', text)

        text = re.sub(r'[^a-zA-Z\s]', '', text)

        tokens = word_tokenize(text)
        filtered_tokens = [w for w in tokens if w not in self.stop_words]

        return ' '.join(filtered_tokens)
    
    def process_dataset(self, df, text_column='text'):
        """Process entire dataset"""
        df['cleaned_text'] = df[text_column].apply(self.clean_text)
        return df

if __name__ == "__main__":
    try:
        df = pd.read_csv("crisis_posts_raw.csv")
    except FileNotFoundError:
        from datetime import datetime, timedelta
        import numpy as np
        
        print("Raw data file not found. Generating sample data...")
        
        crisis_phrases = [
            "feeling depressed", "addiction help", "overwhelmed", "suicidal thoughts", 
            "mental health struggle", "anxiety attack", "severe stress", "feeling lonely",
            "hopeless", "self-harm", "don't want to live", "overdose", "need therapy",
            "panic attack", "can't go on"
        ]
        
        data = []
        now = datetime.now()
        
        for i in range(500):
            created_at = now - timedelta(days=np.random.randint(0, 30), hours=np.random.randint(0, 24))
            text = np.random.choice(crisis_phrases)
            likes = np.random.randint(0, 100)
            retweets = np.random.randint(0, 30)
            replies = np.random.randint(0, 20)
            data.append({
                'id': 1000000000 + i,
                'text': text,
                'created_at': created_at,
                'likes': likes,
                'retweets': retweets,
                'replies': replies
            })
        
        df = pd.DataFrame(data)

    preprocessor = TextPreprocessor()
    df_cleaned = preprocessor.process_dataset(df)

    df_cleaned.to_csv("crisis_posts_cleaned.csv", index=False)
    print(f"Processed {len(df_cleaned)} posts and saved to crisis_posts_cleaned.csv")
