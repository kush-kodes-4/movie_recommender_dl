import os
os.environ["KERAS_BACKEND"] = "torch"
import streamlit as st
import numpy as np
import keras
import pickle
import pandas as pd

st.set_page_config(
    page_title="Movie Recommender AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# DATA & MODEL LOADERS
# ---------------------------------------------------------
@st.cache_data
def get_movie_data():
    extract_dir = "ml-100k_data"
    item_file = os.path.join(extract_dir, "ml-100k", "u.item")
    GENRES = ["unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + GENRES
    movies_df = pd.read_csv(item_file, sep='|', header=None, encoding='latin-1', names=cols)
    
    movie_dict = {}
    for _, row in movies_df.iterrows():
        movie_genres = [g for g in GENRES if row[g] == 1 and g != "unknown"]
        movie_dict[row['movie_id']] = {
            'title': row['title'],
            'release_date': row['release_date'],
            'genres': movie_genres
        }
    return movie_dict, [g for g in GENRES if g != "unknown"]

@st.cache_data
def get_user_profiles():
    extract_dir = "ml-100k_data"
    user_file = os.path.join(extract_dir, "ml-100k", "u.user")
    users_df = pd.read_csv(user_file, sep='|', header=None, names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])
    return users_df.set_index('user_id').to_dict('index')

@st.cache_data
def get_ratings_data():
    extract_dir = "ml-100k_data"
    data_file = os.path.join(extract_dir, "ml-100k", "u.data")
    ratings_df = pd.read_csv(data_file, sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
    return ratings_df

@st.cache_resource
def load_model():
    return keras.models.load_model("movie_recommender.keras")

@st.cache_data
def load_mappings():
    with open("mappings.pkl", "rb") as f:
        mappings = pickle.load(f)
    return mappings

# ---------------------------------------------------------
# CUSTOM STYLING (THEME-ADAPTIVE & MODERN)
# ---------------------------------------------------------
st.markdown("""
<style>
    .movie-card {
        background-color: var(--secondary-background-color, #f8fafc);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }
    .movie-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.12);
        border-color: #ff4b4b;
    }
    .rank-pill {
        display: inline-block;
        background: linear-gradient(135deg, #ff4b4b, #e02424);
        color: white;
        font-weight: 700;
        font-size: 0.75rem;
        padding: 2px 9px;
        border-radius: 6px;
        margin-bottom: 0.5rem;
    }
    .movie-title {
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 0.35rem;
        line-height: 1.3;
    }
    .movie-rating {
        font-size: 0.95rem;
        font-weight: 500;
        color: #f59e0b;
        margin-bottom: 0.4rem;
    }
    .genre-badge {
        display: inline-block;
        background: rgba(128, 128, 128, 0.15);
        color: inherit;
        font-size: 0.72rem;
        font-weight: 500;
        padding: 2px 8px;
        border-radius: 9999px;
        margin-right: 4px;
        margin-top: 4px;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    .profile-box {
        background-color: var(--secondary-background-color, #f8fafc);
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .profile-badge {
        display: inline-block;
        background: rgba(255, 75, 75, 0.12);
        color: #ff4b4b;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# LOAD ASSETS
# ---------------------------------------------------------
with st.spinner("Initializing AI Recommender Engine..."):
    model = load_model()
    mappings = load_mappings()
    movie_data, all_genres = get_movie_data()
    user_profiles = get_user_profiles()
    ratings_df = get_ratings_data()

user2user_encoded = mappings["user2user_encoded"]
movie_encoded2movie = mappings["movie_encoded2movie"]
num_movies = mappings["num_movies"]

# ---------------------------------------------------------
# SIDEBAR: USER SELECTION & CONTROLS
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎬 **Control Panel**")
    
    user_id_input = st.number_input(
        "Select User ID (1 - 943)",
        min_value=1,
        max_value=943,
        step=1,
        value=1,
        help="MovieLens contains 943 real users, each with at least 20 ratings."
    )
    user_id = int(user_id_input)
    
    st.markdown("---")
    st.markdown("### ⚙️ **Recommendation Settings**")
    
    # Feature 1: Dynamic Top-N slider
    top_n = st.slider(
        "Number of Recommendations",
        min_value=5,
        max_value=30,
        value=10,
        step=5,
        help="Choose how many top recommended movies to display."
    )
    
    genre_selection = st.selectbox(
        "Filter by Genre",
        ["All Genres"] + all_genres
    )
    
    get_recs_btn = st.button("Generate Recommendations", type="primary", use_container_width=True)

# ---------------------------------------------------------
# MAIN AREA: HERO & USER PROFILE INSPECTOR (Feature 3)
# ---------------------------------------------------------
st.title("🎬 Neural Movie Recommender")
st.markdown("Personalized movie recommendations powered by deep collaborative filtering neural embeddings.")

if user_id in user_profiles:
    profile = user_profiles[user_id]
    gender_full = "Male" if profile['gender'] == 'M' else ("Female" if profile['gender'] == 'F' else profile['gender'])
    occupation_str = str(profile['occupation']).title()
    age = profile['age']
    
    user_history = ratings_df[ratings_df['user_id'] == user_id]
    num_ratings = len(user_history)
    avg_rating = user_history['rating'].mean() if num_ratings > 0 else 0.0

    st.markdown(f"""
    <div class="profile-box">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;">
            <div>
                <span class="profile-badge">👤 User #{user_id}</span>
                <span style="font-weight: 600; font-size: 1.05rem; margin-left: 4px;">{age} years old • {gender_full} • {occupation_str}</span>
            </div>
            <div style="margin-top: 6px;">
                <span style="font-size: 0.9rem; opacity: 0.85;">Total Ratings: <strong>{num_ratings}</strong> | Avg Given: <strong>{avg_rating:.2f} ★</strong></span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander(f"⭐ View User #{user_id}'s Top Historical Favorites (from training set)", expanded=False):
        top_favorites = user_history.sort_values(by=['rating', 'timestamp'], ascending=[False, False]).head(5)
        fav_cols = st.columns(min(5, len(top_favorites)))
        for idx, (_, fav_row) in enumerate(top_favorites.iterrows()):
            m_id = fav_row['movie_id']
            m_info = movie_data.get(m_id, {})
            m_title = m_info.get('title', f"Movie #{m_id}")
            m_genres = ", ".join(m_info.get('genres', [])[:2])
            with fav_cols[idx]:
                st.markdown(f"""
                <div style="background: rgba(128,128,128,0.1); padding: 0.75rem; border-radius: 8px; text-align: center; height: 100%;">
                    <div style="font-size: 0.9rem; font-weight: 600; margin-bottom: 0.25rem;">{m_title}</div>
                    <div style="color: #f59e0b; font-size: 0.85rem;">{"★" * int(fav_row['rating'])} ({fav_row['rating']}/5)</div>
                    <div style="font-size: 0.7rem; opacity: 0.7; margin-top: 0.2rem;">{m_genres}</div>
                </div>
                """, unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------
# RECOMMENDATION COMPUTATION & GRID DISPLAY
# ---------------------------------------------------------
if user_id not in user2user_encoded:
    st.error(f"User ID {user_id} not found in the training data.")
else:
    user_encoded = user2user_encoded[user_id]
    
    user_encoded_arr = np.array([user_encoded] * num_movies)
    movie_encoded_arr = np.array(range(num_movies))
    
    with st.spinner(f"Computing neural predictions across {num_movies:,} movies..."):
        predictions = model.predict([user_encoded_arr, movie_encoded_arr], verbose=0).flatten()
        
        # Filter by genre if requested
        if genre_selection != "All Genres":
            valid_indices = []
            for i in range(num_movies):
                m_id = movie_encoded2movie[i]
                if genre_selection in movie_data.get(m_id, {}).get('genres', []):
                    valid_indices.append(i)
            
            if not valid_indices:
                st.warning(f"No movies found for genre {genre_selection}.")
                st.stop()
            
            valid_indices = np.array(valid_indices)
            valid_preds = predictions[valid_indices]
            
            top_k = min(top_n, len(valid_preds))
            top_relative_indices = valid_preds.argsort()[-top_k:][::-1]
            top_indices = valid_indices[top_relative_indices]
        else:
            top_k = min(top_n, len(predictions))
            top_indices = predictions.argsort()[-top_k:][::-1]
    
    st.subheader(f"🎯 Top {len(top_indices)} Recommendations {f'({genre_selection})' if genre_selection != 'All Genres' else ''}")
    
    # 2-Column Responsive Card Grid
    col_left, col_right = st.columns(2)
    
    for rank, i in enumerate(top_indices, 1):
        target_col = col_left if rank % 2 != 0 else col_right
        
        m_id = movie_encoded2movie[i]
        movie_info = movie_data.get(m_id, {})
        title = movie_info.get('title', "Unknown Movie")
        genres = movie_info.get('genres', [])
        pred_rating = predictions[i]
        
        # Star visualization
        stars_count = max(1, min(5, round(pred_rating)))
        stars_visual = "★" * stars_count + "☆" * (5 - stars_count)
        
        # Genre badge HTML
        genre_badges_html = "".join([f'<span class="genre-badge">{g}</span>' for g in genres]) if genres else '<span class="genre-badge">General</span>'
        
        with target_col:
            st.markdown(f"""
            <div class="movie-card">
                <span class="rank-pill">#{rank}</span>
                <div class="movie-title">{title}</div>
                <div class="movie-rating">{stars_visual} <span style="font-weight: 600;">{pred_rating:.2f}</span> / 5.00</div>
                <div style="margin-top: 0.4rem;">{genre_badges_html}</div>
            </div>
            """, unsafe_allow_html=True)
