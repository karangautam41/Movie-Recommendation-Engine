# app.py
# Streamlit movie recommender with robust handling for missing dataset + uploader fallback
# Author: ChatGPT (edited for you)
# Usage: streamlit run app.py

import streamlit as st
import pandas as pd
import pickle
import re
import pathlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- CONFIGURATION ----------
# Robustly determine base dir (some runtimes don't set __file__)
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
def run_preprocessing_internal(df_path: pathlib.Path = RAW_DATASET_PATH):
    """
    Load CSV, filter movies, clean text, build TF-IDF matrix,
    and save artifacts to disk (pickle).
    """
    st.info(f"Starting preprocessing using: {df_path}")
    if not df_path.exists():
        raise FileNotFoundError(f"Dataset not found at {df_path}")

    df = pd.read_csv(df_path)
    if "type" in df.columns:
        df = df[df["type"] == "Movie"].copy()  # only movies
    else:
        st.warning("'type' column not found — proceeding with the full dataset")

    # Fill missing text columns
    for col in ["description", "director", "cast", "listed_in"]:
        if col in df.columns:
            df[col] = df[col].fillna("")
        else:
            df[col] = ""

    # Clean fields
    df["description_clean"] = df["description"].astype(str).str.lower()
    df["director_clean"] = df["director"].apply(_clean_names)
    df["cast_clean"] = df["cast"].apply(lambda x: " ".join(_clean_names(x).split()[:3]))
    df["listed_in_clean"] = df["listed_in"].apply(_clean_genres)

    # Combine features
    df["features"] = df.apply(_create_feature_soup, axis=1)
    df = df[df["features"].str.strip().str.len() > 0].copy()
    df.reset_index(drop=True, inplace=True)

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

    st.success(f"Preprocessing complete — saved artifacts to '{DATA_DIR}'.")
    return True


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

        with open(PROCESSED_DATA_PATH, "rb") as f:
            self.df = pickle.load(f)
        with open(TFIDF_MATRIX_PATH, "rb") as f:
            self.tfidf_matrix = pickle.load(f)
        with open(VECTORIZER_PATH, "rb") as f:
            self.vectorizer = pickle.load(f)

    def _get_recommendations_from_vector(self, query_vector, top_n=5, exclude_index=None):
        sim_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        if exclude_index is not None:
            sim_scores[exclude_index] = 0
        indices = sim_scores.argsort()[::-1][:top_n]
        return self.df.iloc[indices]

    def get_recommendations_by_title(self, title, top_n=5):
        try:
            idx = int(self.df[self.df["title"] == title].index[0])
        except Exception:
            return pd.DataFrame()
        query_vector = self.tfidf_matrix[idx]
        return self._get_recommendations_from_vector(query_vector, top_n=top_n, exclude_index=idx)

    def get_recommendations_by_query(self, query_text, top_n=5):
        if not query_text or not query_text.strip():
            return pd.DataFrame()
        query_vector = self.vectorizer.transform([query_text.lower()])
        return self._get_recommendations_from_vector(query_vector, top_n=top_n)


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
    for i, (_, row) in enumerate(recommendations.iterrows()):
        st.subheader(f"#{i+1}: {row.get('title', 'Untitled')} ({row.get('release_year', 'N/A')})")
        col_genre, col_director = st.columns([1, 1])
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

    # Try to load existing artifacts (fast path)
    engine = try_load_engine()

    if engine is None:
        st.warning(
            "Preprocessed model artifacts not found. You must provide the 'netflix_titles.csv' dataset "
            "so the app can preprocess it and build TF-IDF artifacts."
        )

        st.markdown("**Options to provide the data:**")
        st.markdown(
            "- Place `netflix_titles.csv` into the `data/` folder (same folder as this app), **or**\n"
            "- Upload the CSV using the uploader below (the app will save it to `data/netflix_titles.csv` and preprocess)."
        )

        uploaded_file = st.file_uploader("Upload netflix_titles.csv", type=["csv"], help="Upload the Kaggle CSV export.")
        if uploaded_file is not None:
            st.info("File uploaded. Click **Upload & Preprocess** to save and start preprocessing.")
            if st.button("Upload & Preprocess"):
                # Save uploaded file to disk and run preprocessing, then load engine
                with open(RAW_DATASET_PATH, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                with st.spinner("Running preprocessing (this can take a while for large CSVs)..."):
                    try:
                        run_preprocessing_internal(RAW_DATASET_PATH)
                        engine = try_load_engine()
                        if engine:
                            st.success("Model built and loaded successfully — you can now use the app.")
                        else:
                            st.error("Preprocessing finished but engine failed to load. Check logs above.")
                    except Exception as e:
                        st.error(f"Preprocessing failed: {e}")

        # If still no engine, stop further UI construction
        if engine is None:
            st.stop()

    # If engine is present, build the main UI
    st.markdown("---")
    tab1, tab2 = st.tabs(["Recommend by Movie", "Recommend by Description"])

    with tab1:
        st.header("Find movies similar to one you like")
        movie_titles = sorted(engine.df["title"].dropna().unique().tolist())
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


if __name__ == "__main__":
    run_app()
