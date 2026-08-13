import os
os.environ["KERAS_BACKEND"] = "torch"
import numpy as np
import keras
import pickle
import sys
import pandas as pd

def get_movie_names():
    extract_dir = "ml-100k_data"
    item_file = os.path.join(extract_dir, "ml-100k", "u.item")
    movies_df = pd.read_csv(item_file, sep='|', header=None, encoding='latin-1', usecols=[0, 1], names=['movie_id', 'title'])
    return dict(zip(movies_df['movie_id'], movies_df['title']))

if len(sys.argv) < 2:
    print("Usage: python recommend.py <user_id>")
    sys.exit(1)

user_id = int(sys.argv[1])

model = keras.models.load_model("movie_recommender.keras")

with open("mappings.pkl", "rb") as f:
    mappings = pickle.load(f)

user2user_encoded = mappings["user2user_encoded"]
movie_encoded2movie = mappings["movie_encoded2movie"]
num_movies = mappings["num_movies"]

if user_id not in user2user_encoded:
    print("User ID not found.")
    sys.exit(1)

user_encoded = user2user_encoded[user_id]
movie_names = get_movie_names()

user_encoded_arr = np.array([user_encoded] * num_movies)
movie_encoded_arr = np.array(range(num_movies))

predictions = model.predict([user_encoded_arr, movie_encoded_arr], verbose=0).flatten()
top_indices = predictions.argsort()[-10:][::-1]

for i in top_indices:
    m_id = movie_encoded2movie[i]
    title = movie_names.get(m_id, "Unknown")
    print(f"{title} (Predicted Rating: {predictions[i]:.2f})")
