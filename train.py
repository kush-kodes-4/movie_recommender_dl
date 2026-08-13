import os
os.environ["KERAS_BACKEND"] = "torch"
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from data_loader import load_data
from model import create_model
import keras

df, num_users, num_movies, user2user_encoded, userencoded2user, movie2movie_encoded, movie_encoded2movie = load_data()

x = df[['user', 'movie']].values
y = df['rating'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

model = create_model(num_users, num_movies)

model.fit(
    x=[x_train[:, 0], x_train[:, 1]],
    y=y_train,
    batch_size=64,
    epochs=5,
    validation_data=([x_test[:, 0], x_test[:, 1]], y_test)
)

model.save("movie_recommender.keras")

with open("mappings.pkl", "wb") as f:
    pickle.dump({
        "user2user_encoded": user2user_encoded,
        "userencoded2user": userencoded2user,
        "movie2movie_encoded": movie2movie_encoded,
        "movie_encoded2movie": movie_encoded2movie,
        "num_users": num_users,
        "num_movies": num_movies
    }, f)
