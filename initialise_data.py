import pandas as pd
import numpy as np
import json
from matrix_generator import compute_similarities

# Load the movie data
df = pd.read_parquet("data/movies_cleaned_hard.parquet")

# Build and save the similarity matrix
cosine_sim_combined = compute_similarities(df, 0.4, 0.6)
np.save("data/cosine_sim_combined.parquet.npy", cosine_sim_combined)

# Grab all unique genres and save em
def get_all_genres(movies_df):
    unique_genres = set(genre.strip().title() for genres in movies_df['genres'].dropna() for genre in genres.split(','))
    return sorted(unique_genres)

available_genres = get_all_genres(df)
with open("data/unique_genres.json", "w") as f:
    json.dump(available_genres, f)

print("Data files sorted—good to go!")