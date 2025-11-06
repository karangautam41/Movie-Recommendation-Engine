# app.py
# Streamlit movie recommender with robust handling for missing dataset + uploader fallback
# Author: ChatGPT (edited for you)
# Usage: streamlit run app.py

import streamlit as st
import pandas as pd
import pickle
import re
import pathlib
import sys # For better error handling/debugging

# ---------- CONFIGURATION ----------
# Robustly determine base dir
try:
    BASE_DIR = pathlib.Path(__file__).parent
except NameError:
    BASE_DIR = pathlib.Path.cwd()

DATA_DIR = BASE_DIR / "data"
RAW_DATASET_PATH = DATA_DIR / "netflix_titles.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed_data.pkl"
TFIDF_MATRIX_PATH = DATA_DIR / "tfidf_matrix.pkl"
VECTORIZER_PATH = DATA_DIR / "vectorizer.pkl"

# Ensure data directory exists early
DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- NEW: EMBEDDED STARTER DATA FOR GUARANTEED FIRST RUN ---
# This synthetic data mimics the structure of netflix_titles.csv
STARTER_DATA = {
    'show_id': ['s1', 's2', 's3', 's4', 's5', 's6'],
    'type': ['Movie', 'Movie', 'TV Show', 'Movie', 'Movie', 'Movie'],
    'title': ['The Great Adventure', 'A Night in Paris', 'Cyber Spies', 'Zombie Apocalypse', 'Love in the City', 'Deep Sea Mystery'],
    'director': ['John Doe', 'Jane Smith', 'Alex Chen', 'John Doe', 'Maria Garcia', 'Kenji Tanaka'],
    'cast': ['Tom Hanks, Meryl Streep', 'Ryan Gosling, Emma Stone', 'Keanu Reeves, Zendaya', 'Brad Pitt, Angelina Jolie', 'Penelope Cruz, Javier Bardem', 'Tom Hanks, Ken Watanabe'],
    'country': ['United States', 'France', 'United States', 'United States', 'Spain', 'Japan'],
    'release_year': [2021, 2020, 2022, 2019, 2021, 2018],
    'listed_in': ['Action & Adventure, Thrillers', 'Comedies, Romantic Movies', 'Sci-Fi & Fantasy, TV Thrillers', 'Horror Movies, Action & Adventure', 'Romantic Movies, Dramas', 'Thrillers, Documentaries'],
    'description': ['A thrilling journey to find a lost treasure.', 'Two strangers fall in love during one magical night.', 'Hackers team up to stop a global cyber threat.', 'Survivors fight to stay alive after a zombie outbreak.', 'A story of love and heartbreak in a bustling city.', 'A documentary crew explores an ancient underwater secret.'],
}
STARTER_DF = pd.DataFrame(STARTER_DATA)
# -----------------------------------------------------------


# ---------- HELPER CLEANING FUNCTIONS ----------
def _clean_names(text):
    """Clean director/cast names into compact tokens."""
    if pd.isna(text) or text == "":
        return ""
    names = re.split(r",\s*", str(text))
    cleaned = [re.sub(r"[^a-z0-9]", "", n.lower().replace(" ", "")) for n in names]
    return " ".join([c for c in cleaned if c])


def _clean_genres(text):
    """Normalize the 'listed_in' column to simple tokens."""
    if pd.isna(text) or text == "":
        return ""
    return str(text).lower().replace("&", "").replace(",", " ")


def _create_feature_soup(row):
    """Combine textual features into a single string for TF-IDF."""
    return " ".join(
        [
            str(row.get("description_clean", "")),
            str(row.get("director_clean", "")),
            str(row.get("cast_clean", "")),
            str(row.get("listed_in_clean", "")),
        ]
    ).strip()


# ---------- PREPROCESSING ----------
# Use st.cache_data to only run this once until the file changes
@st.cache_data(show_spinner=False)
def run_preprocessing_internal(df_path: pathlib.Path):
    """
    Load CSV, filter movies, clean text, build TF-IDF matrix,
    and save artifacts to disk (pickle).
    """
    if not df_path.exists():
        st.error(f"FATAL ERROR: Dataset not found at {df_path} during preprocessing run.")
        return False

    try:
        df = pd.read_csv(df_path)
        
        # Ensure 'type' column exists, if not, assume all are 'Movie' for this app
        if "type" not in df.columns:
            st.warning("Column 'type' not found. Assuming all entries are movies.")
            df['type'] = 'Movie' # Add the column
            
        df = df[df["type"] == "Movie"].copy()  # only movies

        # Fill missing text columns
        for col in ["description", "director", "cast", "listed_in"]:
            if col not in df.columns:
                df[col] = "" # Add missing column
            df[col] = df[col].fillna("")

        # Clean fields
        df["description_clean"] = df["description"].astype(str).str.lower()
        df["director_clean"] = df["director"].apply(_clean_names)
        df["cast_clean"] = df["cast"].apply(lambda x: " ".join(_clean_names(x).split()[:3]))
        df["listed_in_clean"] = df["listed_in"].apply(_clean_genres)

        # Combine features
        df["features"] = df.apply(_create_feature_soup, axis=1)
        
        # Filter out rows with no features
        df = df[df["features"].str.strip().str.len() > 0].copy()
        df.reset_index(drop=True, inplace=True)
        
        if df.empty:
            st.error("No processable movie data found after cleaning. Cannot build model.")
            return False

        # Vectorize
        tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
        tfidf_matrix = tfidf_vectorizer.fit_transform(df["features"])

        # Save artifacts
        with open(PROCESSED_DATA_PATH, "wb") as f:
            pickle.dump(df, f)
        with open(TFIDF_MATRIX_PATH, "wb") as f:
            pickle.dump(tfidf_matrix, f)
        with open(VECTORIZER_PATH, "wb") as f:
            pickle.dump(tfidf_vectorizer, f)

        return True
    
    except pd.errors.EmptyDataError:
        st.error(f"The file at {df_path} is empty. Please upload a valid CSV.")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred during preprocessing: {e}")
        st.exception(e) # Print full traceback
        return False


# ---------- RECOMMENDATION ENGINE ----------
class RecommendationEngine:
    """
    Loads precomputed artifacts (or raises). Exposes two helper methods:
     - get_recommendations_by_title
     - get_recommendations_by_query
    """

    def __init__(self):
        # load artifacts, or raise
        if not all([PROCESSED_DATA_PATH.exists(), TFIDF_MATRIX_PATH.exists(), VECTORIZER_PATH.exists()]):
            raise FileNotFoundError("One or more artifact files are missing. Please preprocess the dataset first.")

        try:
            with open(PROCESSED_DATA_PATH, "rb") as f:
                self.df = pickle.load(f)
            with open(TFIDF_MATRIX_PATH, "rb") as f:
                self.tfidf_matrix = pickle.load(f)
            with open(VECTORIZER_PATH, "rb") as f:
                self.vectorizer = pickle.load(f)
                
            if self.df.empty:
                raise ValueError("Loaded dataset is empty.")
                
        except (pickle.UnpicklingError, EOFError):
            st.error("Error loading model artifacts. They may be corrupted. Please re-process the data.")
            # Clear corrupted files
            for pth in [PROCESSED_DATA_PATH, TFIDF_MATRIX_PATH, VECTORIZER_PATH]:
                if pth.exists(): pth.unlink()
            raise FileNotFoundError("Corrupted artifacts removed. Please restart.")
        except Exception as e:
            st.error(f"An unexpected error occurred loading artifacts: {e}")
            raise e

    def _get_recommendations_from_vector(self, query_vector, top_n=5, exclude_index=None):
        sim_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        if exclude_index is not None:
            sim_scores[exclude_index] = 0
        
        # Ensure we don't request more recommendations than available
        top_n = min(top_n, len(sim_scores) - (1 if exclude_index is not None else 0))
        
        indices = sim_scores.argsort()[::-1][:top_n]
        return self.df.iloc[indices]

    def get_recommendations_by_title(self, title, top_n=5):
        try:
            # Find the index of the movie
            idx_list = self.df[self.df["title"] == title].index
            if not idx_list.any():
                st.warning(f"Movie '{title}' not found in the dataset.")
                return pd.DataFrame()
            
            idx = int(idx_list[0])
            query_vector = self.tfidf_matrix[idx]
            return self._get_recommendations_from_vector(query_vector, top_n=top_n, exclude_index=idx)
        
        except Exception as e:
            st.error(f"Error getting recommendation by title: {e}")
            return pd.DataFrame()

    def get_recommendations_by_query(self, query_text, top_n=5):
        if not query_text or not query_text.strip():
            return pd.DataFrame()
        try:
            query_vector = self.vectorizer.transform([query_text.lower()])
            return self._get_recommendations_from_vector(query_vector, top_n=top_n)
        except Exception as e:
            st.error(f"Error getting recommendation by query: {e}")
            return pd.DataFrame()


# ---------- UTIL: Try load or return None ----------
def try_load_engine():
    """Attempt to instantiate RecommendationEngine and return it, or None if artifacts missing."""
    try:
        engine = RecommendationEngine()
        return engine
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Unexpected error while loading engine: {e}")
        return None


# ---------- STREAMLIT UI ----------
def display_recommendations(recommendations: pd.DataFrame):
    if recommendations is None or recommendations.empty:
        st.warning("No recommendations found.")
        return
        
    st.markdown("### Your Recommendations")
    for i, (_, row) in enumerate(recommendations.iterrows()):
        st.subheader(f"#{i+1}: {row.get('title', 'Untitled')} ({row.get('release_year', 'N/A')})")
        
        # Use columns for better layout
        col_genre, col_director = st.columns(2)
        with col_genre:
            st.markdown(f"**Genre:** {row.get('listed_in', 'N/A')}")
        with col_director:
            st.markdown(f"**Director:** {row.get('director', 'N/A')}")
            
        with st.expander("Show Synopsis"):
            st.write(row.get("description", "No description available."))
        st.markdown("---")


def run_app():
    st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")
    st.title("🎬 Content-Based Movie Recommender")

    # --- NEW: AUTO-INIT LOGIC ---
    # This block handles the first-run experience
    artifacts_exist = all([p.exists() for p in [PROCESSED_DATA_PATH, TFIDF_MATRIX_PATH, VECTORIZER_PATH]])
    
    if not artifacts_exist:
        if not RAW_DATASET_PATH.exists():
            # If NEITHER artifacts NOR raw data exist, create starter data
            st.info("Welcome! No dataset found. Creating a small starter dataset to get you started.")
            with st.spinner("Saving starter data..."):
                try:
                    STARTER_DF.to_csv(RAW_DATASET_PATH, index=False)
                    st.success(f"Starter dataset saved to `{RAW_DATASET_PATH}`")
                except Exception as e:
                    st.error(f"Failed to save starter data: {e}")
                    st.stop()
        
        # At this point, RAW_DATASET_PATH *must* exist (either starter or user-placed)
        st.info("Preprocessed model artifacts not found. Building them now...")
        with st.spinner("Running preprocessing... This may take a moment."):
            try:
                success = run_preprocessing_internal(RAW_DATASET_PATH)
                if success:
                    st.success("Preprocessing complete! Model is ready.")
                    st.cache_data.clear() # Clear cache after successful run
                    st.rerun() # Rerun script to load the engine
                else:
                    st.error("Preprocessing failed. The app cannot load. See errors above.")
                    st.stop()
            except Exception as e:
                st.error(f"Preprocessing failed: {e}")
                st.stop()

    # --- END AUTO-INIT LOGIC ---

    # Try to load existing artifacts (fast path on subsequent runs)
    engine = try_load_engine()

    # If engine is still None after auto-init, something is wrong
    if engine is None:
        st.error("Failed to load the recommendation engine even after preprocessing. Please check the logs.")
        if st.button("Clear Cache and Retry Preprocessing"):
            for pth in [PROCESSED_DATA_PATH, TFIDF_MATRIX_PATH, VECTORIZER_PATH, RAW_DATASET_PATH]:
                if pth.exists(): pth.unlink()
            st.cache_data.clear()
            st.rerun()
        st.stop()

    # If engine is present, build the main UI
    st.markdown("---")
    tab1, tab2 = st.tabs(["Recommend by Movie", "Recommend by Description"])

    with tab1:
        st.header("Find movies similar to one you like")
        movie_titles = sorted(engine.df["title"].dropna().unique().tolist())
        
        if not movie_titles:
            st.warning("No movie titles found in the loaded data.")
        else:
            col1, col2 = st.columns([3, 1])
            with col1:
                selected_movie = st.selectbox("Choose a movie:", options=movie_titles)
            with col2:
                num_recs = st.slider("Number of recommendations:", 3, 10, 5, key="slider1")

            if st.button("Get Recommendations", key="btn_title"):
                if selected_movie:
                    with st.spinner("Finding similar movies..."):
                        recs = engine.get_recommendations_by_title(selected_movie, top_n=num_recs)
                        display_recommendations(recs)
                else:
                    st.warning("Please select a movie.")

    with tab2:
        st.header("Find movies based on what you're in the mood for")
        col1, col2 = st.columns([3, 1])
        with col1:
            query_text = st.text_area("Describe the movie you want:", "a fast-paced action movie with spies", height=100)
        with col2:
            num_recs_q = st.slider("Number of recommendations:", 3, 10, 5, key="slider2")

        if st.button("Get Recommendations", key="btn_query"):
            if query_text and query_text.strip():
                with st.spinner("Finding recommendations..."):
                    recs = engine.get_recommendations_by_query(query_text, top_n=num_recs_q)
                    display_recommendations(recs)
            else:
                st.warning("Please enter a description.")

    # --- Uploader Section (moved to an expander) ---
    st.markdown("---")
    with st.expander("Upload Your Own Dataset (e.g., full netflix_titles.csv)"):
        st.warning("Uploading a new file will replace the current dataset and require a new preprocessing step.")
        uploaded_file = st.file_uploader("Upload a new CSV file", type=["csv"])
        
        if uploaded_file is not None:
            if st.button("Upload & Replace"):
                with st.spinner("Saving new file and clearing old model..."):
                    try:
                        # Save new file
                        with open(RAW_DATASET_PATH, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                            
                        # Clear old artifacts
                        for pth in [PROCESSED_DATA_PATH, TFIDF_MATRIX_PATH, VECTORIZER_PATH]:
                            if pth.exists(): pth.unlink()
                        st.cache_data.clear()
                        
                        st.success("New file uploaded! The app will now preprocess it.")
                        st.info("Rerunning app...")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Failed to save uploaded file: {e}")


if __name__ == "__main__":
    run_app()
