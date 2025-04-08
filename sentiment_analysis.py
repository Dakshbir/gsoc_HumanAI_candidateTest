import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class CrisisClassifier:
    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.high_risk_terms = {
            'suicide': 0.9, 'kill myself': 0.95, 'end my life': 0.9,
            'overdose': 0.8, 'no reason to live': 0.85, 'can\'t go on': 0.7,
            'want to die': 0.9, 'better off dead': 0.85, 'suicidal': 0.85,
            'self harm': 0.8, 'self-harm': 0.8, 'dont want to live': 0.9,
            'don\'t want to live': 0.9, 'hopeless': 0.6
        }
        
    def analyze_sentiment(self, text):
        """Analyze sentiment of text using VADER"""
        scores = self.sentiment_analyzer.polarity_scores(text)

        if scores['compound'] <= -0.2:
            return 'Negative'
        elif scores['compound'] >= 0.2:
            return 'Positive'
        else:
            return 'Neutral'
    
    def calculate_risk_score(self, text):
        """Calculate risk score based on presence of high-risk terms"""
        text = text.lower()
        risk_score = 0

        for term, weight in self.high_risk_terms.items():
            if term in text:
                risk_score += weight
        
        return min(1.0, risk_score) 
    
    def classify_risk_level(self, text, sentiment):
        """Classify text into risk levels based on content and sentiment"""
        risk_score = self.calculate_risk_score(text)

        if risk_score > 0.6 or (risk_score > 0.3 and sentiment == 'Negative'):
            return 'High-Risk'
        elif risk_score > 0.2 or sentiment == 'Negative':
            return 'Moderate Concern'
        else:
            return 'Low Concern'
    
    def process_dataset(self, df, text_column='cleaned_text'):
        """Process entire dataset with sentiment and risk classification"""
        df['sentiment'] = df[text_column].apply(self.analyze_sentiment)
        df['risk_score'] = df[text_column].apply(self.calculate_risk_score)
        df['risk_level'] = df.apply(lambda x: self.classify_risk_level(
            x[text_column], x['sentiment']), axis=1)
        return df
    
    def visualize_distribution(self, df, save_path='sentiment_risk_distribution.png'):
        """Create visualization of sentiment and risk distribution"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        sentiment_counts = df['sentiment'].value_counts()
        colors = {'Negative': 'red', 'Neutral': 'gray', 'Positive': 'green'}
        ax1.bar(sentiment_counts.index, sentiment_counts.values, 
                color=[colors[x] for x in sentiment_counts.index])
        ax1.set_title('Sentiment Distribution')
        ax1.set_ylabel('Count')

        risk_counts = df['risk_level'].value_counts()
        colors = {'High-Risk': 'red', 'Moderate Concern': 'orange', 'Low Concern': 'green'}
        ax2.bar(risk_counts.index, risk_counts.values, 
                color=[colors[x] for x in risk_counts.index])
        ax2.set_title('Risk Level Distribution')
        ax2.set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(save_path)
        return fig

if __name__ == "__main__":

    try:
        df = pd.read_csv("crisis_posts_cleaned.csv")
    except FileNotFoundError:
        print("Cleaned data file not found. Please run text_preprocessing.py first.")
        exit(1)

    classifier = CrisisClassifier()
    df_analyzed = classifier.process_dataset(df)

    df_analyzed.to_csv("crisis_posts_analyzed.csv", index=False)
    print(f"Analyzed {len(df_analyzed)} posts and saved to crisis_posts_analyzed.csv")

    classifier.visualize_distribution(df_analyzed)
    print("Created visualization at sentiment_risk_distribution.png")

    print("\nSentiment Distribution:")
    print(df_analyzed['sentiment'].value_counts())
    
    print("\nRisk Level Distribution:")
    print(df_analyzed['risk_level'].value_counts())
