# app.py
# Streamlit movie recommender with automatic fallback dataset, uploader, and robust preprocessing.
# Usage: streamlit run app.py

import streamlit as st
import pandas as pd
import pickle
import re
import io
import pathlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------
# 0) BUILT-IN SAMPLE DATASET
# --------------------------
# Minimal, valid sample matching the Kaggle schema so the app "just works".
SAMPLE_NETFLIX_CSV = """show_id,type,title,director,cast,country,date_added,release_year,rating,duration,listed_in,description
s1,Movie,Inception,Christopher Nolan,Leonardo DiCaprio|Joseph Gordon-Levitt|Ellen Page,United States,September 1, 2010,2010,PG-13,148 min,Action & Adventure, A thief who enters dreams must plant an idea to redeem himself.
s2,Movie,The Matrix,Lana Wachowski|Lilly Wachowski,Keanu Reeves|Laurence Fishburne|Carrie-Anne Moss,United States,April 1, 1999,1999,R,136 min,Sci-Fi & Fantasy, A hacker learns reality is a simulation and fights its controllers.
s3,Movie,Interstellar,Christopher Nolan,Matthew McConaughey|Anne Hathaway|Jessica Chastain,United States,November 7, 2014,2014,PG-13,169 min,Sci-Fi & Fantasy, Explorers travel through a wormhole in space in an attempt to ensure humanity's survival.
s4,Movie,The Irishman,Martin Scorsese,Robert De Niro|Al Pacino|Joe Pesci,United States,November 27, 2019,2019,R,209 min,Dramas, A hitman reflects on his life and ties to organized crime.
s5,Movie,Roma,Alfonso Cuarón,Yalitza Aparicio|Marina de Tavira,Mexico,December 14, 2018,2018,R,135 min,Dramas, A year in the life of a middle-class family's maid in 1970s Mexico City.
s6,Movie,Bird Box,Susanne Bier,Sandra Bullock|Trevante Rhodes|John Malkovich,United States,December 21, 2018,2018,R,124 min,Thrillers, A mother journeys blindfolded to protect her children from unseen entities.
s7,Movie,Marriage Story,Noah Baumbach,Scarlett Johansson|Adam Driver,United States,December 6, 2019,2019,R,137 min,Dramas, A couple navigates a divorce that pushes them to extremes.
s8,Movie,Extraction,Sam Hargrave,Chris Hemsworth|Rudhraksh Jaiswal,United States,April 24, 2020,2020,R,117 min,Action & Adventure, A mercenary undertakes a deadly mission to rescue a drug lord's kidnapped son.
"""

def sample_csv_bytes() -> bytes:
    # Normalize line endings and return UTF-8 bytes
    return SAMPLE_NETFLIX_CSV.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


# --------------------------
# 1) PATHS & CONFIG
# --------------------------
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

try:
    BASE_DIR = pathlib.Path(__file__).parent  # typical script usage
except NameError:
    BASE_DIR = pathlib.Path.cwd()             # fallback (rare environments)

DATA_DIR = BASE_DIR / "data"
RAW_DATASET_PATH = DATA_DIR / "netflix_titles.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed_data.pkl"
TFIDF_MATRIX_PATH = DATA_DIR / "tfidf_matrix.pkl"
VECTORIZER_PATH = DATA_DIR / "vectorizer.pkl"

DATA_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------
# 2) CLEANING HELPERS
# --------------------------
def _clean_names(text):
    """Clean director/cast into compact tokens."""
    if pd.isna(text) or text == "":
        return ""
    # Original Kaggle uses comma-separated; some rows above use pipe to show robustness.
    # Support both separators.
    parts = re.split(r"[|,]\s*", str(text))
    tokens = [re.sub(r"[^a-z0-9]", "", p.lower()) for p in parts]
    return " ".join([t for t in tokens if t])


def _clean_genres(text):
    """Normalize genre string into tokens."""
    if pd.isna(text) or text == "":
        return ""
    return str(text).lower().replace("&", "").replace(",", " ")


def _create_feature_soup(row):
    """Combine textual fields for TF-IDF."""
    return " ".join(
        [
            str(row.get("description_clean", "")),
            str(row.get("director_clean", "")),
            str(row.get("cast_clean", "")),
            str(row.get("listed_in_clean", "")),
        ]
    ).strip()


# --------------------------
# 3) PREPROCESSING
# --------------------------
def run_preprocessing_internal(csv_path: pathlib.Path) -> None:
    """
    Reads CSV, cleans, builds TF-IDF, and saves artifacts.
    Raises on hard errors; UI catches and shows clean messages.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # Ensure required columns exist (create blanks if missing so app still works)
    required = ["title", "type", "description", "director", "cast", "listed_in", "release_year"]
    for col in required:
        if col not in df.columns:
            df[col] = ""

    # Keep only movies when possible
    try:
        df = df[df["type"].astype(str).str.lower() == "movie"].copy()
    except Exception:
        pass

    # Fill NAs
    for col in ["description", "director", "cast", "listed_in"]:
        df[col] = df[col].fillna("")

    # Clean fields
    df["description_clean"] = df["description"].astype(str).str.lower()
    df["director_clean"] = df["director"].apply(_clean_names)
    df["cast_clean"] = df["cast"].apply(lambda x: " ".join(_clean_names(x).split()[:3]))
    df["listed_in_clean"] = df["listed_in"].apply(_clean_genres)

    # Feature soup
    df["features"] = df.apply(_create_feature_soup, axis=1)
    df = df[df["features"].str.strip().str.len() > 0].copy()
    df.reset_index(drop=True, inplace=True)

    # Vectorize
    vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
    tfidf_matrix = vectorizer.fit_transform(df["features"])

    # Save artifacts
    with open(PROCESSED_DATA_PATH, "wb") as f:
        pickle.dump(df, f)
    with open(TFIDF_MATRIX_PATH, "wb") as f:
        pickle.dump(tfidf_matrix, f)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)


def ensure_artifacts(auto_use_sample: bool = True) -> None:
    """
    Ensure artifacts exist. If RAW CSV missing and auto_use_sample is True,
    write the built-in sample and preprocess automatically.
    """
    have_artifacts = all(p.exists() for p in [PROCESSED_DATA_PATH, TFIDF_MATRIX_PATH, VECTORIZER_PATH])
    if have_artifacts:
        return

    # Ensure a CSV exists
    if not RAW_DATASET_PATH.exists() and auto_use_sample:
        RAW_DATASET_PATH.write_bytes(sample_csv_bytes())

    # If still no CSV, bail (caller will show uploader UI)
    if not RAW_DATASET_PATH.exists():
        return

    # Build artifacts
    run_preprocessing_internal(RAW_DATASET_PATH)


# --------------------------
# 4) ENGINE
# --------------------------
class RecommendationEngine:
    def __init__(self):
        with open(PROCESSED_DATA_PATH, "rb") as f:
            self.df = pickle.load(f)
        with open(TFIDF_MATRIX_PATH, "rb") as f:
            self.tfidf_matrix = pickle.load(f)
        with open(VECTORIZER_PATH, "rb") as f:
            self.vectorizer = pickle.load(f)

    def _from_vector(self, query_vector, top_n=5, exclude_index=None):
        sims = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        if exclude_index is not None and 0 <= exclude_index < sims.size:
            sims[exclude_index] = 0
        idxs = sims.argsort()[::-1][:top_n]
        return self.df.iloc[idxs]

    def by_title(self, title, top_n=5):
        try:
            idx = int(self.df[self.df["title"] == title].index[0])
        except Exception:
            return pd.DataFrame()
        return self._from_vector(self.tfidf_matrix[idx], top_n=top_n, exclude_index=idx)

    def by_query(self, text, top_n=5):
        if not text or not str(text).strip():
            return pd.DataFrame()
        v = self.vectorizer.transform([str(text).lower()])
        return self._from_vector(v, top_n=top_n)


@st.cache_resource(show_spinner=False)
def load_engine_cached():
    # Ensure artifacts (auto-fallback to sample if needed)
    try:
        ensure_artifacts(auto_use_sample=True)
    except Exception as e:
        # Let UI rebuild on demand; don't cache failures
        st.session_state["_ensure_error"] = str(e)
        return None

    # If artifacts exist, load engine
    if all(p.exists() for p in [PROCESSED_DATA_PATH, TFIDF_MATRIX_PATH, VECTORIZER_PATH]):
        try:
            return RecommendationEngine()
        except Exception as e:
            st.session_state["_engine_error"] = str(e)
            return None
    return None


# --------------------------
# 5) UI HELPERS
# --------------------------
def display_recs(df: pd.DataFrame):
    if df is None or df.empty:
        st.warning("No recommendations found.")
        return
    for i, (_, row) in enumerate(df.iterrows()):
        st.subheader(f"#{i+1}: {row.get('title', 'Untitled')} ({row.get('release_year', 'N/A')})")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown(f"**Genre:** {row.get('listed_in', 'N/A')}")
        with c2:
            st.markdown(f"**Director:** {row.get('director', 'N/A')}")
        with st.expander("Show Synopsis"):
            st.write(row.get("description", "No description available."))
        st.markdown("---")


def rebuild_artifacts_from_current_csv():
    try:
        run_preprocessing_internal(RAW_DATASET_PATH)
        st.success("Artifacts rebuilt successfully.")
        st.cache_resource.clear()  # clear the engine cache
    except Exception as e:
        st.error(f"Rebuild failed: {e}")


# --------------------------
# 6) APP
# --------------------------
st.title("🎬 Content-Based Movie Recommender")

# Top utility bar (download sample + rebuild)
with st.expander("Data & Utilities", expanded=False):
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        st.download_button(
            "Download sample CSV",
            data=sample_csv_bytes(),
            file_name="netflix_titles_sample.csv",
            mime="text/csv",
            use_container_width=True,
            help="Grab a small built-in sample you can inspect or modify."
        )
    with c2:
        if st.button("Rebuild artifacts from current CSV", use_container_width=True):
            rebuild_artifacts_from_current_csv()

    st.caption(f"Current data folder: `{DATA_DIR}`")

# Load engine (auto-preprocess with sample if needed)
engine = load_engine_cached()

# If engine still missing, show uploader and a "Use sample now" button
if engine is None:
    st.warning(
        "Model artifacts are not available yet. Provide a dataset below or use the built-in sample."
    )

    up = st.file_uploader(
        "Upload your netflix_titles.csv",
        type=["csv"],
        help="Kaggle Netflix Titles CSV. The app will save it to data/netflix_titles.csv and build artifacts."
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Use built-in sample now", use_container_width=True):
            try:
                RAW_DATASET_PATH.write_bytes(sample_csv_bytes())
                run_preprocessing_
