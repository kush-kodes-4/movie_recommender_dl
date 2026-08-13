import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras import layers

def create_model(num_users, num_movies, embedding_size=50):
    user_input = layers.Input(shape=(1,))
    user_embedding = layers.Embedding(num_users, embedding_size)(user_input)
    user_vec = layers.Flatten()(user_embedding)
    
    movie_input = layers.Input(shape=(1,))
    movie_embedding = layers.Embedding(num_movies, embedding_size)(movie_input)
    movie_vec = layers.Flatten()(movie_embedding)
    
    concat = layers.Concatenate()([user_vec, movie_vec])
    
    dense_1 = layers.Dense(128, activation='relu')(concat)
    dropout_1 = layers.Dropout(0.2)(dense_1)
    dense_2 = layers.Dense(64, activation='relu')(dropout_1)
    dropout_2 = layers.Dropout(0.2)(dense_2)
    output = layers.Dense(1)(dropout_2)
    
    model = keras.Model(inputs=[user_input, movie_input], outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model
