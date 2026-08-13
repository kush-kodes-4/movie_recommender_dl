import os
os.environ["KERAS_BACKEND"] = "torch"
import streamlit as st
import numpy as np
import keras
import pickle
import pandas as pd


st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="centered"
)

@st.cache_data
def get_movie_data():
    extract_dir = "ml-100k_data"
    item_file = os.path.join(extract_dir, "ml-100k", "u.item")
    GENRES = ["unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + GENRES
    movies_df = pd.read_csv(item_file, sep='|', header=None, encoding='latin-1', names=cols)
    
    movie_dict = {}
    for _, row in movies_df.iterrows():
        movie_genres = [g for g in GENRES if row[g] == 1]
        movie_dict[row['movie_id']] = {'title': row['title'], 'genres': movie_genres}
    return movie_dict, GENRES

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
    .reportview-container .main .block-container{
        padding-top: 2rem;
    }
    .movie-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    .movie-card:hover {
        transform: translateY(-5px);
    }
    .movie-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .movie-rating {
        font-size: 1rem;
        color: #4b5563;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #ff3333;
    }
</style>
""", unsafe_allow_html=True)


st.title("🎬 Movie Recommender AI")
st.markdown("Discover your next favorite movie! Enter your User ID below to get personalized recommendations powered by a neural network.")

with st.spinner("Loading model and data..."):
    model = load_model()
    mappings = load_mappings()
    movie_data, all_genres = get_movie_data()

user2user_encoded = mappings["user2user_encoded"]
movie_encoded2movie = mappings["movie_encoded2movie"]
num_movies = mappings["num_movies"]

st.divider()

col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    user_id_input = st.number_input("Enter User ID", min_value=1, step=1, value=1)

with col2:
    genre_selection = st.selectbox("Filter by Genre", ["All"] + all_genres)

with col3:
    st.write("")
    st.write("")
    get_recs_btn = st.button("Get Recommendations")

if get_recs_btn:
    user_id = int(user_id_input)
    
    if user_id not in user2user_encoded:
        st.error(f"User ID {user_id} not found in the training data. Please try another ID.")
    else:
        st.success(f"Finding top movies for User {user_id}...")
        
        user_encoded = user2user_encoded[user_id]
        
        
        user_encoded_arr = np.array([user_encoded] * num_movies)
        movie_encoded_arr = np.array(range(num_movies))
        
        with st.spinner("Analyzing preferences..."):
            predictions = model.predict([user_encoded_arr, movie_encoded_arr], verbose=0).flatten()
            
            if genre_selection != "All":
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
                
                top_k = min(10, len(valid_preds))
                top_relative_indices = valid_preds.argsort()[-top_k:][::-1]
                top_indices = valid_indices[top_relative_indices]
            else:
                top_indices = predictions.argsort()[-10:][::-1]
        
        st.subheader(f"🌟 Top 10 {genre_selection if genre_selection != 'All' else ''} Recommendations".replace('  ', ' '))
        
        
        for i in top_indices:
            m_id = movie_encoded2movie[i]
            title = movie_data.get(m_id, {}).get('title', "Unknown")
            genres = ", ".join(movie_data.get(m_id, {}).get('genres', []))
            rating = predictions[i]
            
            
            st.markdown(f"""
                <div class="movie-card">
                    <div class="movie-title">{title}</div>
                    <div class="movie-rating">⭐ Predicted Rating: {rating:.2f} / 5.00</div>
                    <div style="color: #6b7280; font-size: 0.875rem; margin-top: 0.25rem;">🎭 Genres: {genres}</div>
                </div>
            """, unsafe_allow_html=True)
