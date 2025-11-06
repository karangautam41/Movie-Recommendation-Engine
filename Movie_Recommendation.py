# 1. IMPORTS
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
import requests # Not strictly needed anymore, but often useful
import re       # For cleaning text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pathlib # Used for robust, OS-agnostic path handling

# 2. CONFIGURATION
# Define the base directory as the directory where this script (app.py) is located.
# This ensures paths are reliable regardless of the user's current working directory.
BASE_DIR = pathlib.Path(__file__).parent 

# Define global constants for file paths. All processed files are stored in the 'data' directory.
DATA_DIR = BASE_DIR / 'data' 
# Path to the raw dataset from Kaggle
RAW_DATASET_PATH = DATA_DIR / 'netflix_titles.csv'
# Path to save the cleaned pandas DataFrame
PROCESSED_DATA_PATH = DATA_DIR / 'processed_data.pkl'
# Path to save the computed TF-IDF feature matrix
TFIDF_MATRIX_PATH = DATA_DIR / 'tfidf_matrix.pkl'
# Path to save the fitted TF-IDF vectorizer object
VECTORIZER_PATH = DATA_DİR / 'vectorizer.pkl'

# TMDB_API_KEY is now completely removed as the poster functionality is not required.

# 3. PREPROCESSING LOGIC
# These functions are responsible for loading the raw data, cleaning it,
# and transforming it into a format suitable for the recommendation engine.

def _clean_names(text):
    """
    Helper function to clean director or cast names.
    Example: "John Woo, James McTeigue" -> "johnwoo jamesmcteigue"
    This turns multiple names into a single string of unique tokens for vectorization.
    """
    if pd.isna(text) or text == '':
        return ''
    # Split names by comma
    names = re.split(r',\s*', text)
    # Clean each name (lowercase, no spaces, remove special characters)
    cleaned_names = [re.sub(r'[^a-z0-9]', '', str(name).lower().replace(' ', '')) for name in names]
    # Join them back with a space
    return ' '.join(filter(None, cleaned_names)) # Filter out empty strings

def _clean_genres(text):
    """
    Helper function to clean the 'listed_in' (genres) column.
    Example: "Action & Adventure, Comedies" -> "action adventure comedies"
    """
    if pd.isna(text) or text == '':
        return ''
    # Lowercase, remove '&', and replace commas/spaces with a single space
    return str(text).lower().replace('&', '').replace(',', ' ')

def _create_feature_soup(row):
    """
    Helper function to combine all key text features into a single
    string (a "soup" of features). This combined string will be fed into
    the TF-IDF Vectorizer.
    """
    # Ensure all components are strings before concatenation
    return (
        str(row['description_clean']) + ' ' +
        str(row['director_clean']) + ' ' +
        str(row['cast_clean']) + ' ' +
        str(row['listed_in_clean'])
    )

def run_preprocessing_internal(df_path=RAW_DATASET_PATH):
    """
    Main preprocessing pipeline.
    This function is executed automatically if the model artifacts are missing.
    It loads the raw CSV, cleans it, vectorizes it, and saves the artifacts.
    """
    print(f"Starting automatic data preprocessing using {df_path}...")

    # 1. Load Data
    # Use .exists() from pathlib for reliable path checking
    if not df_path.exists():
        # This is the line causing the "error" - it's a safety check!
        # It correctly stops the app if the required data file is missing.
        raise FileNotFoundError(f"ERROR: Dataset not found at {df_path}. Please download 'netflix_titles.csv' and place it in the 'data' directory.")

    df = pd.read_csv(df_path)
    # We only want to recommend movies
    df = df[df['type'] == 'Movie'].copy()
    
    # 2. Clean Data
    # Fill missing values in key text columns to avoid errors
    fill_cols = ['description', 'director', 'cast', 'listed_in']
    for col in fill_cols:
        df[col] = df[col].fillna('')

    print("Cleaning and combining features...")
    # Apply the cleaning helper functions
    df['description_clean'] = df['description'].astype(str).str.lower()
    df['director_clean'] = df['director'].apply(_clean_names)
    # Only take the top 3 cast members for relevance
    df['cast_clean'] = df['cast'].apply(lambda x: ' '.join(_clean_names(x).split()[:3]))
    df['listed_in_clean'] = df['listed_in'].apply(_clean_genres)
    
    # 3. Create Feature Soup
    df['features'] = df.apply(_create_feature_soup, axis=1)
    
    # Drop any movies that have no features at all
    df = df[df['features'].str.strip().str.len() > 0].copy()
    df.reset_index(drop=True, inplace=True)
    
    print(f"Data loaded and cleaned. Final shape: {df.shape}")

    # 4. Vectorize Features
    print("Generating TF-IDF matrix from 'features'...")
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['features'])

    # 5. Save Artifacts
    # Create the 'data' directory if it doesn't exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save artifacts using their pathlib paths
    with open(PROCESSED_DATA_PATH, 'wb') as f:
        pickle.dump(df, f)
    with open(TFIDF_MATRIX_PATH, 'wb') as f:
        pickle.dump(tfidf_matrix, f)
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)

    print(f"Preprocessing complete. Artifacts saved to '{DATA_DIR}'.")
    return True 

# 4. RECOMMENDATION ENGINE
class RecommendationEngine:
    """
    This class encapsulates all the recommendation logic.
    It loads the preprocessed data and provides methods to get
    recommendations based on a movie title or a text query.
    """
    
    def __init__(self):
        """
        Initializes the engine by loading the saved artifacts. It checks for
        the existence of the pickle files and automatically runs preprocessing
        if any of them are missing.
        """
        # A check to see if all necessary files exist
        if not all([PROCESSED_DATA_PATH.exists(), 
                    TFIDF_MATRIX_PATH.exists(), 
                    VECTORIZER_PATH.exists()]):
            # If any file is missing, automatically run the preprocessing step
            run_preprocessing_internal() 
            
        # Now, load the artifacts (they should exist)
        try:
            # Load the cleaned movie data
            with open(PROCESSED_DATA_PATH, 'rb') as f:
                self.df = pickle.load(f)
            # Load the pre-computed TF-IDF feature matrix
            with open(TFIDF_MATRIX_PATH, 'rb') as f:
                self.tfidf_matrix = pickle.load(f)
            # Load the fitted vectorizer
            with open(VECTORIZER_PATH, 'rb') as f:
                self.vectorizer = pickle.load(f)
        except Exception as e:
            # Catch all loading errors, including those from run_preprocessing_internal
            raise IOError(f"Failed to load model artifacts. Check if '{RAW_DATASET_PATH}' exists and is accessible. Error: {e}")

    def _get_recommendations_from_vector(self, query_vector, top_n, exclude_index=None):
        """
        Internal helper function to find the most similar movies to a
        given query vector using cosine similarity.
        """
        # Calculate cosine similarity between the query vector and all movie vectors
        sim_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # If recommending by title, exclude the movie itself
        if exclude_index is not None:
            sim_scores[exclude_index] = 0 
        
        # Get the indices of the movies with the highest similarity scores
        similar_movie_indices = sim_scores.argsort()[::-1][:top_n]
        
        # Return the DataFrame rows corresponding to these indices
        return self.df.iloc[similar_movie_indices]

    def get_recommendations_by_title(self, title, top_n=5):
        """
        Generates movie recommendations based on a given movie title.
        """
        try:
            # Find the index of the selected movie
            idx = self.df[self.df['title'] == title].index[0]
        except IndexError:
            # Return an empty DataFrame if the movie title isn't found
            return pd.DataFrame()
            
        # Get the pre-computed TF-IDF vector for this movie
        query_vector = self.tfidf_matrix[idx]
        
        return self._get_recommendations_from_vector(query_vector, top_n, exclude_index=idx)

    def get_recommendations_by_query(self, query_text, top_n=5):
        """
        Generates movie recommendations based on a user's text description.
        """
        if not query_text or not query_text.strip():
            return pd.DataFrame()
            
        # Transform the raw text query into a TF-IDF vector
        query_vector = self.vectorizer.transform([query_text.lower()])
        
        return self._get_recommendations_from_vector(query_vector, top_n)

# 5. STREAMLIT APPLICATION
def run_app():
    """
    Defines and runs the Streamlit web application.
    """

    @st.cache_resource(show_spinner="Loading model and data. This may take a moment if preprocessing is running...")
    def load_engine():
        """
        Loads the RecommendationEngine object. This function is cached. 
        The engine's __init__ will automatically trigger preprocessing 
        if the model files are missing.
        """
        try:
            return RecommendationEngine()
        except Exception as e:
            # This is where your FileNotFoundError is caught and displayed
            st.error(
                f"A fatal error occurred during model loading or preprocessing. Please ensure you have the required file ('{RAW_DATASET_PATH}') in the 'data' directory. Error: {e}"
            )
            return None

    def display_recommendations(recommendations):
        """
        Helper function to display the recommended movies in a clean
        text-only, list format. (API/Poster logic removed)
        """
        if recommendations.empty:
            st.warning("Could not find any recommendations.")
            return

        # Display results in a clean, stacked list format
        for i, (_, row) in enumerate(recommendations.iterrows()):
            st.subheader(f"#{i+1}: {row['title']} ({row['release_year']})")
            
            # Use columns to align key details neatly
            col_genre, col_director = st.columns([1, 1])
            
            with col_genre:
                st.markdown(f"**Genre:** {row.get('listed_in', 'N/A')}")
            
            with col_director:
                st.markdown(f"**Director:** {row.get('director', 'N/A')}")

            # Use an expander for the full description/synopsis
            with st.expander("Show Synopsis"):
                st.markdown(row.get('description', 'No description available.'))
            
            st.markdown("---") # Separator between movies
    
    # Main Application Logic    
    st.set_page_config(
        page_title="Movie Recommendation System",
        page_icon="🎬",
        layout="wide"
    )

    # Load the engine. This is cached and handles preprocessing if needed.
    engine = load_engine()

    # Only run the app if the engine loaded successfully
    if engine:
        st.title("🎬 Content-Based Movie Recommender")
        st.markdown(f"---") 

        # Create two tabs for the two recommendation modes
        tab1, tab2 = st.tabs(["**Recommend by Movie**", "**Recommend by Description**"])

        # Tab 1: Recommend by Movie Title
        with tab1:
            st.header("Find movies similar to one you like:")
            # Get the list of all movie titles for the dropdown
            movie_titles = sorted(engine.df['title'].tolist()) 
            
            col1, col2 = st.columns([3, 1])
            with col1:
                selected_movie = st.selectbox("Choose a movie:", options=movie_titles)
            with col2:
                num_recommendations = st.slider("Number of recommendations:", 3, 10, 5, key='slider1')

            if st.button("Get Recommendations", key='button1', type="primary", use_container_width=True):
                if selected_movie:
                    st.markdown(f"### Results for movies similar to '{selected_movie}':")
                    with st.spinner('Finding similar movies...'):
                        recommendations = engine.get_recommendations_by_title(selected_movie, top_n=num_recommendations)
                        display_recommendations(recommendations)
                else:
                    st.warning("Please select a movie to get recommendations.")


        # Tab 2: Recommend by Text Query
        with tab2:
            st.header("Find movies based on what you're in the mood for:")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                query_text = st.text_area("Describe the movie you want to see:", "a fast-paced action movie with spies", height=100)
            with col2:
                num_query_recommendations = st.slider("Number of recommendations:", 3, 10, 5, key='slider2')
            
            if st.button("Get Recommendations", key='button2', type="primary", use_container_width=True):
                if query_text:
                    st.markdown(f"### Results for description: '{query_text}'")
                    with st.spinner('Finding recommendations...'):
                        recommendations = engine.get_recommendations_by_query(query_text, top_n=num_query_recommendations)
                        display_recommendations(recommendations)
                else:
                    st.warning("Please enter a description to get recommendations.")

# 6. MAIN EXECUTION BLOCK
if __name__ == "__main__":
    # The application now always runs the Streamlit app.
    # The 'load_engine' function handles the automatic preprocessing.
    run_app()
