from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
movies = [
    "Romantic comedy with a fun storyline",
    "Action thriller with suspense",
    "Adventure in a magical world"
]
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(movies)
similarity = cosine_similarity(tfidf)
print(similarity)
