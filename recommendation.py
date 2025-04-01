import pandas as pd

# Keeps track of recommendation cycles
index_shift = 0
last_title = ""

def get_recommendations_filtered(df, title, selected_genre=None, cosine_sim_combined=None, top_n=10):
    # Gets movie recs based on title, can filter by genre if you want
    global index_shift, last_title

    # If same title, shift to next set of recs
    if title == last_title:
        index_shift += 10
    else:
        index_shift = 0
        last_title = title

    # Find movie index from title
    indices = pd.Series(df.index, index=df['title']).to_dict()
    if title not in indices:
        return "Title not found, mate."
    idx = indices[title]

    # Grab similarity scores from matrix
    sim_scores = list(enumerate(cosine_sim_combined[idx]))
    filtered_sim_scores = [(i, score) for i, score in sim_scores if i != idx]  # Skip the movie itself

    # Filter by genre if given
    if selected_genre:
        filtered_sim_scores = [
            (i, score) for i, score in filtered_sim_scores if selected_genre in df.iloc[i]['genres']
        ]

    # Sort by similarity, apply offset for pagination
    filtered_sim_scores = sorted(filtered_sim_scores, key=lambda x: x[1], reverse=True)
    filtered_sim_scores = filtered_sim_scores[index_shift:index_shift + top_n] if index_shift < len(filtered_sim_scores) else []

    # Build result dataframe with similarity scores
    movie_indices = [i[0] for i in filtered_sim_scores]
    similarities = [score for _, score in filtered_sim_scores]
    result_df = df.iloc[movie_indices].copy()
    result_df['similarity_score'] = similarities

    return result_df.round(2)  # Keeps it neat with 2 decimal places