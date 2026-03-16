from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
movies = [
    "Action adventure and hero story",
    "Romantic love story",
    "Adventure and fantasy world"
]
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(movies)
similarity = cosine_similarity(tfidf)
print(similarity)
