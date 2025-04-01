from genre_filter import filter_movies_by_genre
from recommendation import get_recommendations_filtered
import numpy as np
import pandas as pd
import json
from tabulate import tabulate

# Load precomputed goodies
cosine_sim_combined = np.load("data/cosine_sim_combined.parquet.npy")
with open("data/unique_genres.json", "r") as f:
    available_genres = json.load(f)

def main():
    df = pd.read_parquet("data/movies_cleaned_hard.parquet")
    available_genres_lower = [genre.lower() for genre in available_genres]

    while True:
        print("\n" + "="*50)
        print("Welcome to the Movie Recommendation System")
        print("Pick an option:")
        print("1. Explore by genre")
        print("2. Search by movie title")
        print("3. Exit")
        user_choice = input("Enter 1, 2, or 3: ").strip()

        if user_choice == "1":
            print("\nAvailable Genres:\n" + ", ".join(available_genres) + "\n")
            selected_genre = search_genre(df, available_genres_lower, available_genres)
            if selected_genre:
                print(f"\nWanna search for more {selected_genre} flicks? Let’s go by movie.")
                search_titles(df, selected_genre, cosine_sim_combined)
            if ask_restart():
                continue
            break

        elif user_choice == "2":
            search_titles(df, None, cosine_sim_combined)
            if ask_restart():
                continue
            break

        elif user_choice == "3":
            print("Cheers for using the system—see ya!")
            break

        else:
            print("Nah, mate, pick 1, 2, or 3.")

def ask_restart():
    # Quick check if user wants another go
    restart = input("\nAnother search? (y/n): ").strip().lower()
    return restart == 'y' or restart == 'yes'

def get_result_count():
    while True:
        try:
            count = int(input("\nHow many recs? (5-50): ").strip())
            if 5 <= count <= 50:
                return count
            print("Gimme a number between 5 and 50.")
        except ValueError:
            print("That’s not a number, try again.")

def search_genre(df, available_genres_lower, available_genres, user_genre=None):
    if not user_genre:
        user_genre = input("Enter a genre: ").strip().lower()
    if user_genre in available_genres_lower:
        selected_genre = available_genres[available_genres_lower.index(user_genre)]
        filtered_movies = filter_movies_by_genre(df, selected_genre)
        if not filtered_movies.empty:
            print(f"\nTop 10 {selected_genre} Movies (by Weighted Rating):\n")
            print(tabulate(filtered_movies[['title', 'score', 'vote_average', 'vote_count']],
                           headers="keys", tablefmt="pretty", showindex=False))
            return selected_genre
        print("\nNo flicks found here.")
        return None

    # Suggest similar genres if no exact match
    partial_matches = [genre for genre in available_genres if user_genre in genre.lower()]
    if partial_matches:
        print("\nDid ya mean one of these? " + ", ".join(partial_matches))
        user_retry = input("Try again: ").strip().lower()
        return search_genre(df, available_genres_lower, available_genres, user_retry)

    user_retry = input("\nGenre not found—have another crack: ").strip().lower()
    return search_genre(df, available_genres_lower, available_genres, user_retry)

def search_titles(df, genre, cosine_sim_combined, user_title=None):
    if not user_title:
        user_title = input("\nEnter a movie: ").strip().lower()

    result_count = get_result_count()

    if user_title in df['title'].str.lower().values:
        selected_title = df['title'][df['title'].str.lower() == user_title].iloc[0]
        filtered_recommendations = get_recommendations_filtered(df, selected_title, genre, cosine_sim_combined, top_n=result_count)

        if isinstance(filtered_recommendations, str):
            print(f"\n{filtered_recommendations}")
            return
        if genre:
            print(f"\nTop {result_count} {genre} Movies like {selected_title}:\n")
        else:
            print(f"\nTop {result_count} Movies like {selected_title}:\n")

        display_columns = ['title', 'score', 'vote_average', 'vote_count']
        if 'similarity_score' in filtered_recommendations.columns:
            display_columns.append('similarity_score')
            filtered_recommendations['similarity_score'] = filtered_recommendations['similarity_score'].apply(lambda x: round(x, 2))

        print(tabulate(filtered_recommendations[display_columns],
                       headers="keys", tablefmt="pretty", showindex=False))
        return

    # Offer partial matches if title’s close
    partial_matches = df['title'][df['title'].str.lower().str.contains(user_title, na=False)]
    if not partial_matches.empty:
        print(f"\nDid ya mean one of these?\n{'\n'.join(partial_matches[:5])}")
        user_retry = input("Have another go: ").strip().lower()
        return search_titles(df, genre, cosine_sim_combined, user_retry)

    user_retry = input("\nTitle not found—try again: ").strip().lower()
    return search_titles(df, genre, cosine_sim_combined, user_retry)

if __name__ == "__main__":
    main()