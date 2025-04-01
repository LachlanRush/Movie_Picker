from genre_filter import filter_movies_by_genre
from recommendation import get_recommendations_filtered
import numpy as np
import pandas as pd
import json
from tabulate import tabulate

# Load precomputed data
cosine_sim_combined = np.load("data/cosine_sim_combined.parquet.npy")
with open("data/unique_genres.json", "r") as f:
    available_genres = json.load(f)

def main():
    df = pd.read_parquet("data/movies_cleaned_hard.parquet")
    available_genres_lower = [genre.lower() for genre in available_genres]

    while True:
        print("\n" + "=" * 50)
        print("Welcome to the Movie Recommendation System")
        print("Pick an option:")
        print("1. Explore by genre")
        print("2. Search by movie title")
        user_choice = input("Enter 1 or 2: ").strip()

        if user_choice == "1":
            print("\nAvailable Genres:\n" + ", ".join(available_genres) + "\n")
            selected_genre = search_genre(df, available_genres_lower, available_genres)
            if selected_genre:
                search_titles(df, selected_genre, cosine_sim_combined)
            if ask_restart():
                continue

        elif user_choice == "2":
            search_titles(df, None, cosine_sim_combined)
            if ask_restart():
                continue

        else:
            print("Invalid choice. Please enter 1 or 2.")

def ask_restart():
    restart = input("\nAnother search? (y/n): ").strip().lower()
    return restart in ('y', 'yes')

def search_genre(df, available_genres_lower, available_genres):
    while True:
        user_genre = input("Enter a genre: ").strip().lower()
        if user_genre in available_genres_lower:
            selected_genre = available_genres[available_genres_lower.index(user_genre)]
            filtered_movies = filter_movies_by_genre(df, selected_genre)
            if not filtered_movies.empty:
                print(f"\nTop 10 {selected_genre} Movies (by Weighted Rating):\n")
                print(tabulate(filtered_movies[['title', 'score', 'vote_average', 'vote_count']],
                               headers="keys", tablefmt="pretty", showindex=False))
                return selected_genre
            print("\nNo movies found in this genre.")
        else:
            partial_matches = [genre for genre in available_genres if user_genre in genre.lower()]
            if partial_matches:
                print("\nDid you mean one of these? " + ", ".join(partial_matches))


def search_titles(df, genre, cosine_sim_combined):
    while True:
        user_title = input("\nEnter a movie: ").strip().lower()
        result_count = 10

        matched_titles = df[df['title'].str.lower() == user_title]
        if not matched_titles.empty:
            selected_title = matched_titles.iloc[0]['title']
            filtered_recommendations = get_recommendations_filtered(df, selected_title, genre, cosine_sim_combined, top_n=result_count)

            if isinstance(filtered_recommendations, str):
                print(f"\n{filtered_recommendations}")
            else:
                if genre:
                    print(f"\nTop 10 {genre} Movies like {selected_title}:")
                else:
                    print(f"\nTop 10 Movies like {selected_title}:")
                display_columns = ['title', 'score', 'vote_average', 'vote_count']
                if 'similarity_score' in filtered_recommendations.columns:
                    display_columns.append('similarity_score')
                    filtered_recommendations['similarity_score'] = filtered_recommendations['similarity_score'].apply(lambda x: round(x, 2))
                print(tabulate(filtered_recommendations[display_columns], headers="keys", tablefmt="pretty", showindex=False))
            return

        partial_matches = df[df['title'].str.lower().str.contains(user_title, na=False)]['title']
        if not partial_matches.empty:
            print("\nDid you mean one of these?\n" + "\n".join(partial_matches[:5]))

if __name__ == "__main__":
    main()

