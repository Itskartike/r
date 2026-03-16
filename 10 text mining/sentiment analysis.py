from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
reviews = [
    "The product is amazing",
    "Very bad quality",
    "I am happy with the purchase",
    "Worst experience ever"
]
sentiment = ["Positive", "Negative", "Positive", "Negative"]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(reviews)
model = LogisticRegression()
model.fit(X, sentiment)
print(model.predict(vectorizer.transform(["The product quality is good"])))

