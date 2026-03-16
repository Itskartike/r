from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
documents = [
    "Introduction to Python programming",
    "Advanced machine learning concepts",
    "Basics of data science"
]
query = ["Python programming"]
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(documents + query)
similarity = (tfidf * tfidf.T).toarray()
print("Most relevant document index:",np.argmax(similarity[-1][:-1]))
