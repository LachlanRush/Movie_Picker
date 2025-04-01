from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import re

def preprocess_text(text):
    # Cleans up text for vectorizing—lowercase, no punctuation, tidy spaces
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def compute_similarities(df, cos1=0.4, cos2=0.6):
    # Computes combined similarity matrix using overview and metadata
    df['overview'] = df['overview'].fillna('').apply(preprocess_text)

    # TF-IDF for movie overviews
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000, min_df=2)
    tfidf_matrix = tfidf.fit_transform(df['overview'])
    cosine_sim1 = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Weight genres—main one gets a bit more love
    def process_genres(genres):
        genres_list = genres.split(',')
        main_genres = genres_list[:1]
        return ' '.join(main_genres + genres_list)

    df['genres_weighted'] = df['genres'].apply(process_genres)

    # Weight cast and director for metadata soup
    df['cast_weighted'] = df['cast'].fillna('').apply(
        lambda x: ' '.join([actor.strip() for actor in x.split(',')[:3]])
    )
    df['director_weighted'] = df['director'].fillna('').apply(
        lambda x: ' '.join([x.strip() + ' ' + x.strip()])  # Double up director
    )
    df['keywords_weighted'] = df['keywords'].fillna('').apply(
        lambda x: ' '.join([kw.strip() for kw in x.split(',')])
    )

    # Combine into one big metadata soup
    df['metadata_soup'] = (
        df['genres'] + ' ' +
        df['director_weighted'] + ' ' +
        df['cast_weighted'] + ' ' +
        df['keywords_weighted'] + ' ' +
        df['title'].fillna('')
    )
    df['metadata_soup'] = df['metadata_soup'].apply(preprocess_text)

    # CountVectorizer for metadata
    count_vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000, min_df=2)
    count_matrix = count_vectorizer.fit_transform(df['metadata_soup'])
    cosine_sim2 = cosine_similarity(count_matrix)

    # Normalise matrices so they play nice together
    scaler = MinMaxScaler()
    cosine_sim1 = scaler.fit_transform(cosine_sim1)
    cosine_sim2 = scaler.fit_transform(cosine_sim2)

    # Combine with weights—cos1 for overview, cos2 for metadata
    cosine_sim_combined = cos1 * cosine_sim1 + cos2 * cosine_sim2

    return cosine_sim_combined