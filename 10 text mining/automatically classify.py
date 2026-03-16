from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
# Sample data
texts = [
    "Football match ends with thrilling goal",
    "Government passes new education bill",
    "Cricket world cup semi-final today",
    "Election campaigns begin in several states"
]
labels = ["Sports", "Politics", "Sports", "Politics"]
# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
# Model training
model = MultinomialNB()
model.fit(X, labels)
# Prediction
prediction = model.predict(
vectorizer.transform(["Big cricket match today"])
)
print(prediction)
