import streamlit as st
import pandas as pd
from genre_filter import filter_movies_by_genre
from recommendation import get_recommendations_filtered
import json
import numpy as np
import difflib

# Load genres for dropdown
with open("data/unique_genres.json", "r") as f:
    available_genres = json.load(f)

st.set_page_config(page_title="Movie Picker", layout="wide")
st.title(":clapper: Welcome to Movie Picker")
st.subheader(":rainbow[Your Cineast Mates Help Ya Pick a Flick!] :sunglasses:", divider="gray")

@st.cache_data
def load_data():
    # Load movie data, clean titles for matching
    df = pd.read_parquet("data/movies_cleaned_hard.parquet")
    df["title_cleaned"] = df["title"].str.strip().str.lower()
    return df

df = load_data()
cosine_sim_combined = np.load("data/cosine_sim_combined.parquet.npy")

# Set up session state for pagination
if "movie_offset" not in st.session_state:
    st.session_state.movie_offset = 0
if "mode" not in st.session_state:
    st.session_state.mode = None

def main():
    title_to_index = pd.Series(df.index, index=df["title_cleaned"]).to_dict()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎬 Discover by Genre", use_container_width=True):
            st.session_state.mode = "By Genre"
    with col2:
        if st.button("🔎 Find by Movie Name", use_container_width=True):
            st.session_state.mode = "By Movie"

    st.markdown("---")

    if st.session_state.mode == "By Genre":
        selected_genre = st.selectbox("Choose a Genre", ["-- Pick one --"] + available_genres)

        if selected_genre != "-- Pick one --":
            movies = filter_movies_by_genre(df, selected_genre)
            total_movies = len(movies)
            movies_to_show = movies.iloc[st.session_state.movie_offset:st.session_state.movie_offset + 10]

            st.write(f"### {selected_genre} Movies ({st.session_state.movie_offset + 1}-{st.session_state.movie_offset + len(movies_to_show)} of {total_movies})")
            st.write(f"Genre: {selected_genre}")

            for _, row in movies_to_show.iterrows():
                st.subheader(row['title'])
                st.text(f"Rating: {row['score']} ⭐")
                st.text(f"Director: {row['director']}")
                st.text(f"Actors: {row['cast']}")
                st.write(row['overview'])
                if 'poster_url' in row:
                    st.image(row['poster_url'])
                st.markdown("---")

            movie_input = st.text_input(f"Enter a {selected_genre.lower()} movie title:", key="movie_search")

            if movie_input:
                movie_input_cleaned = movie_input.strip().lower()
                if movie_input_cleaned not in title_to_index:
                    st.error(f"Movie not found in {selected_genre.lower()}—try another!")
                else:
                    filtered_recommendations = get_recommendations_filtered(df, df.loc[title_to_index[movie_input_cleaned], "title"], cosine_sim_combined=cosine_sim_combined, selected_genre=selected_genre, top_n=10)

                    if isinstance(filtered_recommendations, str) or filtered_recommendations.empty:
                        st.error(f"No {selected_genre.lower()} matches—check your spelling!")
                    else:
                        st.write(f"### Movies like {movie_input}")
                        for _, row in filtered_recommendations.iterrows():
                            st.subheader(row['title'])
                            st.text(f"Rating: {row['score']} ⭐")
                            st.text(f"Director: {row['director']}")
                            st.text(f"Actors: {row['cast']}")
                            st.text(f"Genres: {row['genres']}")
                            st.write(row['overview'])
                            if 'poster_url' in row:
                                st.image(row['poster_url'])
                            st.markdown("---")

    elif st.session_state.mode == "By Movie":
        movie_input = st.text_input("Enter a Movie Title:")

        if movie_input:
            suggestions = difflib.get_close_matches(movie_input, title_to_index, n=10, cutoff=0.3)
            if suggestions:
                st.write("Or did ya mean...")
                cols = st.columns(len(suggestions), gap="small")
                for i, suggestion in enumerate(suggestions):
                    with cols[i]:
                        st.text(f"{suggestion.title()} ")
                st.markdown("---")

            movie_input_cleaned = movie_input.strip().lower()
            if movie_input_cleaned not in title_to_index:
                st.error("Movie not in database—give it another go!")
            else:
                recommendations = get_recommendations_filtered(df, df.loc[title_to_index[movie_input_cleaned], "title"], cosine_sim_combined=cosine_sim_combined, top_n=10)
                if isinstance(recommendations, str) or recommendations.empty:
                    st.error("No recs found—check ya input!")
                else:
                    st.write(f"### Movies like {movie_input}")
                    for _, row in recommendations.iterrows():
                        st.subheader(row['title'])
                        st.text(f"Rating: {row['score']} ⭐")
                        st.text(f"Director: {row['director']}")
                        st.text(f"Actors: {row['cast']}")
                        st.write(row['overview'])
                        st.write(row['genres'])
                        if 'poster_url' in row:
                            st.image(row['poster_url'])
                        st.markdown("---")

if __name__ == "__main__":
    main()