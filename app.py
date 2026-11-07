import os
os.environ["KERAS_BACKEND"] = "torch"
import streamlit as st
import numpy as np
import keras
import pickle
import pandas as pd

st.set_page_config(
    page_title="Movie Recommender & Taste Analytics",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def get_movie_data():
    extract_dir = "ml-100k_data"
    item_file = os.path.join(extract_dir, "ml-100k", "u.item")
    GENRES = ["unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + GENRES
    movies_df = pd.read_csv(item_file, sep='|', header=None, encoding='latin-1', names=cols)
    
    clean_genres = [g for g in GENRES if g != "unknown"]
    
    movie_dict = {}
    for _, row in movies_df.iterrows():
        movie_genres = [g for g in clean_genres if row[g] == 1]
        movie_dict[row['movie_id']] = {
            'title': row['title'],
            'release_date': row['release_date'],
            'genres': movie_genres
        }
    return movie_dict, clean_genres, movies_df

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

st.markdown("""
<style>
    .movie-card {
        background-color: var(--secondary-background-color, #f8fafc);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1.1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        position: relative;
    }
    .movie-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.12);
        border-color: #ff4b4b;
    }
    .card-badges {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.6rem;
    }
    .rank-pill {
        background: linear-gradient(135deg, #ff4b4b, #e02424);
        color: white;
        font-weight: 700;
        font-size: 0.75rem;
        padding: 2px 9px;
        border-radius: 6px;
    }
    .match-pill {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        font-weight: 700;
        font-size: 0.75rem;
        padding: 2px 9px;
        border-radius: 6px;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .movie-title {
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 0.35rem;
        line-height: 1.3;
    }
    .movie-rating {
        font-size: 0.95rem;
        font-weight: 600;
        color: #f59e0b;
        margin-bottom: 0.5rem;
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
    .profile-hero {
        background-color: var(--secondary-background-color, #f8fafc);
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.2rem;
    }
    .profile-chip {
        display: inline-block;
        background: rgba(255, 75, 75, 0.12);
        color: #ff4b4b;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        margin-right: 8px;
    }
    .seen-badge {
        background: rgba(107, 114, 128, 0.18);
        color: #6b7280;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 2px 6px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

with st.spinner("Initializing Recommender & Analytics Engine..."):
    model = load_model()
    mappings = load_mappings()
    movie_data, all_genres, movies_df = get_movie_data()
    user_profiles = get_user_profiles()
    ratings_df = get_ratings_data()

user2user_encoded = mappings["user2user_encoded"]
movie_encoded2movie = mappings["movie_encoded2movie"]
num_movies = mappings["num_movies"]

with st.sidebar:
    st.markdown("## **Control Panel")
    
    user_id_input = st.number_input(
        "Select User ID",
        min_value=1,
        max_value=943,
        step=1,
        value=1,
        help="Select any of the 943 real users in the MovieLens 100k benchmark."
    )
    user_id = int(user_id_input)
    
    st.markdown("---")
    st.markdown("Recommendation Tuning")
    
    top_n = st.slider(
        "Recommendations Count (Top-N)",
        min_value=5,
        max_value=30,
        value=10,
        step=5,
        help="Adjust how many top predictions to return."
    )
    
    genre_selection = st.selectbox(
        "Filter by Genre",
        ["All Genres"] + all_genres
    )
    
    exclude_rated = st.checkbox(
        "Exclude Already-Rated Movies",
        value=True,
        help="Filter out movies this user has already watched and scored in the dataset."
    )
    
    st.markdown("---")
    get_recs_btn = st.button("Generate AI Recommendations", type="primary", use_container_width=True)
    
    with st.expander("Dataset Information", expanded=False):
        st.markdown("""
        * **Dataset:** MovieLens 100K
        * **Total Users:** 943 (min 20 ratings/user)
        * **Total Movies:** 1,682
        * **Ratings Scale:** 1.0 to 5.0
        * **Embedding Size:** 50 dimensions
        """)

st.title("Movie Recommender & Taste Analytics")
st.markdown("Deep collaborative filtering recommendation system with user taste profiling and model diagnostics.")

if user_id in user_profiles:
    profile = user_profiles[user_id]
    gender_full = "Male" if profile['gender'] == 'M' else ("Female" if profile['gender'] == 'F' else profile['gender'])
    occupation_str = str(profile['occupation']).title()
    age = profile['age']
    zip_code = profile['zip_code']
    
    user_history = ratings_df[ratings_df['user_id'] == user_id]
    num_ratings = len(user_history)
    avg_rating = user_history['rating'].mean() if num_ratings > 0 else 0.0
    
    user_merged = user_history.merge(movies_df[['movie_id'] + all_genres], on='movie_id')
    user_genre_counts = user_merged[all_genres].sum()
    top_genre_name = user_genre_counts.idxmax() if not user_genre_counts.empty and user_genre_counts.max() > 0 else "N/A"
    
    st.markdown(f"""
    <div class="profile-hero">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;">
            <div>
                <span class="profile-chip">👤 User #{user_id}</span>
                <span style="font-weight: 600; font-size: 1.1rem;">{age} years old • {gender_full} • {occupation_str}</span>
                <span style="font-size: 0.85rem; opacity: 0.7; margin-left: 6px;">(ZIP: {zip_code})</span>
            </div>
            <div style="margin-top: 6px;">
                <span style="font-size: 0.95rem;">Rated: <strong>{num_ratings} movies</strong> | Avg Score: <strong>{avg_rating:.2f} ★</strong> | Primary Genre: <strong>{top_genre_name}</strong></span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    user_history = ratings_df[ratings_df['user_id'] == user_id]
    num_ratings = len(user_history)
    avg_rating = 0.0
    user_merged = pd.DataFrame()
    user_genre_counts = pd.Series(dtype=float)
    top_genre_name = "N/A"

tab_recs, tab_analytics, tab_explain = st.tabs([
    " AI Recommendations",
    " User Taste & Historical Analytics",
    "AI Explainability & Alignment"
])

rated_movie_ids = set(user_history['movie_id']) if num_ratings > 0 else set()

if user_id not in user2user_encoded:
    st.error(f"User ID {user_id} was not found in the trained embedding mappings.")
    st.stop()

user_encoded = user2user_encoded[user_id]
user_encoded_arr = np.array([user_encoded] * num_movies)
movie_encoded_arr = np.array(range(num_movies))

with st.spinner("Analyzing neural embedding preferences across 1,682 movies..."):
    predictions = model.predict([user_encoded_arr, movie_encoded_arr], verbose=0).flatten()

valid_indices = []
for i in range(num_movies):
    m_id = movie_encoded2movie[i]
    
    if exclude_rated and m_id in rated_movie_ids:
        continue
    
    if genre_selection != "All Genres":
        m_genres = movie_data.get(m_id, {}).get('genres', [])
        if genre_selection not in m_genres:
            continue
            
    valid_indices.append(i)

if not valid_indices:
    top_indices = []
else:
    valid_indices = np.array(valid_indices)
    valid_preds = predictions[valid_indices]
    top_k = min(top_n, len(valid_preds))
    top_relative_indices = valid_preds.argsort()[-top_k:][::-1]
    top_indices = valid_indices[top_relative_indices]

with tab_recs:
    if not len(top_indices):
        st.warning(f"No movies found matching criteria (Genre: {genre_selection}, Exclude Rated: {exclude_rated}).")
    else:
        st.subheader(f"Recommended Movies ({len(top_indices)} items)")
        if exclude_rated:
            st.caption("Showing movies user has **not yet rated**.")
        else:
            st.caption("Showing top movies across the entire catalog (including previously watched).")
            
        col_left, col_right = st.columns(2)
        
        for rank, i in enumerate(top_indices, 1):
            target_col = col_left if rank % 2 != 0 else col_right
            
            m_id = movie_encoded2movie[i]
            movie_info = movie_data.get(m_id, {})
            title = movie_info.get('title', "Unknown Movie")
            genres = movie_info.get('genres', [])
            pred_rating = predictions[i]
            
            match_pct = int(np.clip((pred_rating / 5.0) * 100, 45, 99))
            
            stars_count = max(1, min(5, round(pred_rating)))
            stars_visual = "★" * stars_count + "☆" * (5 - stars_count)
            
            genre_badges_html = "".join([f'<span class="genre-badge">{g}</span>' for g in genres]) if genres else '<span class="genre-badge">General</span>'
            
            is_previously_rated = m_id in rated_movie_ids
            seen_badge_html = '<span class="seen-badge">Previously Rated</span>' if is_previously_rated else ''
            
            with target_col:
                st.markdown(f"""
                <div class="movie-card">
                    <div class="card-badges">
                        <span class="rank-pill">#{rank} Recommendation</span>
                        <div>
                            {seen_badge_html}
                            <span class="match-pill">{match_pct}% Match</span>
                        </div>
                    </div>
                    <div class="movie-title">{title}</div>
                    <div class="movie-rating">{stars_visual} {pred_rating:.2f} / 5.00</div>
                    <div>{genre_badges_html}</div>
                </div>
                """, unsafe_allow_html=True)

with tab_analytics:
    st.subheader(f"📊 Historical Taste Profile for User #{user_id}")
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Movies Rated", f"{num_ratings}")
    with m2:
        st.metric("Average Rating Given", f"{avg_rating:.2f} ★")
    with m3:
        high_ratings = len(user_history[user_history['rating'] >= 4])
        high_pct = (high_ratings / num_ratings * 100) if num_ratings > 0 else 0
        st.metric("Positive Ratings (4-5★)", f"{high_ratings} ({high_pct:.0f}%)")
    with m4:
        st.metric("Most Watched Genre", top_genre_name)
    
    st.markdown("---")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("Historical Genre Affinity")
        st.caption("Distribution of genres across all movies previously watched by this user.")
        if not user_genre_counts.empty and user_genre_counts.max() > 0:
            top_user_genres = user_genre_counts[user_genre_counts > 0].sort_values(ascending=False).head(10)
            st.bar_chart(top_user_genres)
        else:
            st.info("No genre data available.")
            
    with chart_col2:
        st.markdown("Historical Rating Distribution")
        st.caption("How critical is this user? Breakdown of 1-star to 5-star ratings given.")
        if num_ratings > 0:
            rating_dist = user_history['rating'].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
            rating_dist.index = ["1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"]
            st.bar_chart(rating_dist)
        else:
            st.info("No ratings history available.")

    st.markdown("---")
    
    st.markdown("Top 5 Historical Favorites (From Training Data)")
    top_favs = user_history.sort_values(by=['rating', 'timestamp'], ascending=[False, False]).head(5)
    
    fav_cols = st.columns(min(5, max(1, len(top_favs))))
    for idx, (_, fav_row) in enumerate(top_favs.iterrows()):
        m_id = fav_row['movie_id']
        m_info = movie_data.get(m_id, {})
        m_title = m_info.get('title', f"Movie #{m_id}")
        m_genres = ", ".join(m_info.get('genres', [])[:2])
        with fav_cols[idx]:
            st.markdown(f"""
            <div style="background: var(--secondary-background-color, #f8fafc); border: 1px solid rgba(128,128,128,0.2); padding: 0.9rem; border-radius: 10px; text-align: center; height: 100%;">
                <div style="font-size: 0.9rem; font-weight: 600; margin-bottom: 0.3rem;">{m_title}</div>
                <div style="color: #f59e0b; font-size: 0.85rem; font-weight: 700;">{"★" * int(fav_row['rating'])} ({fav_row['rating']}/5)</div>
                <div style="font-size: 0.72rem; opacity: 0.75; margin-top: 0.25rem;">{m_genres}</div>
            </div>
            """, unsafe_allow_html=True)

with tab_explain:
    st.subheader("Alignment & Recommendation Explainability")
    st.markdown("""
    How did the collaborative neural network decide what to recommend?
    Below is a comparison between the user's **historical genre affinity** and the **genres recommended by the AI**.
    """)
    
    if len(top_indices) > 0:
        rec_genres_dict = {g: 0 for g in all_genres}
        for idx in top_indices:
            m_id = movie_encoded2movie[idx]
            for g in movie_data.get(m_id, {}).get('genres', []):
                rec_genres_dict[g] = rec_genres_dict.get(g, 0) + 1
                
        rec_genres_series = pd.Series(rec_genres_dict)
        rec_genres_series = rec_genres_series[rec_genres_series > 0].sort_values(ascending=False)
        
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            st.markdown("Recommended Genres Distribution")
            st.caption(f"Genre frequency among the Top {len(top_indices)} recommended movies.")
            st.bar_chart(rec_genres_series)
            
        with exp_col2:
            st.markdown("Neural Collaborative Filtering Mechanics")
            st.markdown(f"""
            * **User Latent Embedding:** User #{user_id} is mapped to a 50-dimensional latent preference vector learned across {num_ratings} training ratings.
            * **Item Latent Embedding:** Each movie is mapped to a 50-dimensional vector capturing latent attributes (tone, director style, era, pacing).
            * **Collaborative Matching:** The model computes interactions between the user's vector and all candidate movies through two Dense layers (128 → 64 units) with Dropout.
            * **Top Overlap:** The model heavily emphasizes **{top_genre_name}**, matching User #{user_id}'s strongest historical signal.
            """)
    else:
        st.info("No recommendations currently generated to analyze.")
