import pandas as pd
import folium
from folium.plugins import HeatMap
import spacy

class GeospatialAnalyzer:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            import sys
            print("Downloading spaCy model...")
            from spacy.cli import download
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        self.location_coords = {
            'New York': (40.7128, -74.0060),
            'Los Angeles': (34.0522, -118.2437),
            'Chicago': (41.8781, -87.6298),
            'Houston': (29.7604, -95.3698),
            'Phoenix': (33.4484, -112.0740),
            'Philadelphia': (39.9526, -75.1652),
            'San Antonio': (29.4241, -98.4936),
            'San Diego': (32.7157, -117.1611),
            'Dallas': (32.7767, -96.7970),
            'Austin': (30.2672, -97.7431),
            'Boston': (42.3601, -71.0589),
            'Seattle': (47.6062, -122.3321),
            'Denver': (39.7392, -104.9903),
            'Atlanta': (33.7490, -84.3880),
            'Miami': (25.7617, -80.1918)
        }
        
    def extract_locations(self, text):
        """Extract location mentions from text using NER"""
        doc = self.nlp(text)
        locations = [ent.text for ent in doc.ents if ent.label_ == "GPE" or ent.label_ == "LOC"]
        return locations[0] if locations else None
    
    def geocode_location(self, location_name):
        """Convert location name to coordinates"""
        if not location_name:
            return None

        for loc, coords in self.location_coords.items():
            if loc.lower() in location_name.lower():
                return coords
        
        return None
    
    def process_dataset(self, df):
        """Extract and geocode locations from dataset"""
        df['extracted_location'] = df['text'].apply(self.extract_locations)

        df['coordinates'] = df['extracted_location'].apply(self.geocode_location)
        
        return df
    
    def create_crisis_heatmap(self, df, risk_column='risk_level', save_path='crisis_heatmap.html'):
        """Create heatmap visualization of crisis hotspots"""

        df_with_loc = df[df['coordinates'].notna() & 
                         df[risk_column].isin(['High-Risk', 'Moderate Concern'])]
 
        crisis_map = folium.Map(location=[39.8, -98.5], zoom_start=4)

        heat_data = []
        for _, row in df_with_loc.iterrows():
            coords = row['coordinates']
            if coords:
                weight = 1.0 if row[risk_column] == 'High-Risk' else 0.5
                heat_data.append([coords[0], coords[1], weight])

        HeatMap(heat_data).add_to(crisis_map)

        location_counts = df_with_loc['extracted_location'].value_counts().head(5)

        for loc, count in location_counts.items():
            coords = self.geocode_location(loc)
            if coords:
                folium.Marker(
                    location=coords,
                    popup=f"{loc}: {count} crisis mentions",
                    icon=folium.Icon(color='red', icon='info-sign')
                ).add_to(crisis_map)

        crisis_map.save(save_path)
        
        return crisis_map, location_counts

if __name__ == "__main__":
    try:
        df = pd.read_csv("crisis_posts_analyzed.csv")
    except FileNotFoundError:
        print("Analyzed data file not found. Please run sentiment_analysis.py first.")
        exit(1)

    geo_analyzer = GeospatialAnalyzer()
    df_geo = geo_analyzer.process_dataset(df)

    df_geo.to_csv("crisis_posts_geocoded.csv", index=False)
    print(f"Geocoded {len(df_geo)} posts and saved to crisis_posts_geocoded.csv")

    crisis_map, top_locations = geo_analyzer.create_crisis_heatmap(df_geo)
    print("Created heatmap visualization at crisis_heatmap.html")

    print("\nTop 5 Locations with Highest Crisis Mentions:")
    print(top_locations)

    location_count = df_geo['coordinates'].notna().sum()
    print(f"\nTotal posts with valid location data: {location_count} ({location_count/len(df_geo)*100:.1f}%)")
