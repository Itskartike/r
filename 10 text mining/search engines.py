from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
documents = [
    "Data science and machine learning",
    "Introduction to text mining",
    "Python for data analysis"
]
query = ["text mining"]
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(documents + query)
similarity = (tfidf * tfidf.T).toarray()
print("Most relevant document index:", np.argmax(similarity[-1][:-1]))
