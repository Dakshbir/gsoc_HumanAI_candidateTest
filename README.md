# AI-Powered Crisis Detection

This project implements an AI-driven public health monitoring system that can detect and address emerging suicide, substance use, and mental health crises in real-time.

## Features

- Social media data extraction for crisis-related content
- Sentiment analysis and risk classification
- Geospatial mapping of crisis hotspots
- Interactive visualization of crisis trends

## Setup

1. Clone the repository
2. Install dependencies: \pip install -r requirements.txt\
3. Create a \.env\ file with your Twitter API credentials
4. Run the scripts in sequence:
   - \python data_extraction.py\
   - \python text_preprocessing.py\
   - \python sentiment_analysis.py\
   - \python geospatial_analysis.py\

## Outputs

- \crisis_posts_cleaned.csv\: Preprocessed data ready for analysis
- \crisis_posts_analyzed.csv\: Data with sentiment and risk classifications
- \crisis_posts_geocoded.csv\: Data with location information
- \Risk Level Distribution.png\: Visualization of risk distributions
- \Sentiment Distribution.png\: Visualization of sentiment distributions
- \crisis_heatmap.html\: Interactive map of crisis hotspots
- \top_locations.csv\: Gives us the top locations 

