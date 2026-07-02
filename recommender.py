import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("dataset/movies.csv")
ratings = pd.read_csv("dataset/ratings.csv")

print("Movies Dataset")
print(movies.head())

print("\nRatings Dataset")
print(ratings.head())

print("\nMissing Values")
print(movies.isnull().sum())

movies = movies[['movieId', 'title', 'genres']]
print("\nProcessed Movies Dataset")
print(movies.head())

movies['genres'] = movies['genres'].str.replace('|', ' ', regex=False)
print("\nCleaned Genres")
print(movies[['title', 'genres']].head())

cv = CountVectorizer()
genre_matrix = cv.fit_transform(movies['genres'])

similarity = cosine_similarity(genre_matrix)
print("\nSimilarity Matrix Shape:")
print(similarity.shape)

def recommend(movie_name):

    movie_index = movies[movies['title'] == movie_name].index

    if len(movie_index) == 0:
        return ["Movie not found."]

    movie_index = movie_index[0]

    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )

    recommended_movies = []

    for movie in movie_list[1:6]:
        movie_title = movies.iloc[movie[0]].title
        recommended_movies.append(movie_title)

    return recommended_movies

if __name__ == "__main__":
    print("\nRecommended Movies:")
    recommendations = recommend("Toy Story (1995)")

    for movie in recommendations:
        print(movie)