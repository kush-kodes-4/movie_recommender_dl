import pandas as pd
import numpy as np
import urllib.request
import zipfile
import os

def load_data():
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    zip_path = "ml-100k.zip"
    extract_dir = "ml-100k_data"
    
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(url, zip_path)
    
    if not os.path.exists(extract_dir):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            
    data_file = os.path.join(extract_dir, "ml-100k", "u.data")
    df = pd.read_csv(data_file, sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
    
    user_ids = df['user_id'].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    userencoded2user = {i: x for i, x in enumerate(user_ids)}
    
    movie_ids = df['movie_id'].unique().tolist()
    movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
    movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}
    
    df['user'] = df['user_id'].map(user2user_encoded)
    df['movie'] = df['movie_id'].map(movie2movie_encoded)
    
    num_users = len(user2user_encoded)
    num_movies = len(movie2movie_encoded)
    
    return df, num_users, num_movies, user2user_encoded, userencoded2user, movie2movie_encoded, movie_encoded2movie
