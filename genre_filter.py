import pandas as pd

def filter_movies_by_genre(movies_df, selected_genre):
    # Filters movies by genre, ignores case, grabs top 10
    cleaned_genre = selected_genre.lower().strip()
    genre_filter = movies_df["genres"].apply(lambda x: cleaned_genre in [g.lower().strip() for g in x.split(",")])
    return movies_df.loc[genre_filter].head(10).round(2)  # Rounds scores to 2 decimals