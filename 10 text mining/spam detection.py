from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
messages = [
    "Win a free lottery now",
    "Meeting scheduled tomorrow",
    "Urgent offer claim now",
    "Project discussion today"
]
labels = ["Spam", "Not Spam", "Spam", "Not Spam"]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(messages)
model = MultinomialNB()
model.fit(X, labels)
print(model.predict(vectorizer.transform(["Free offer just for you"])))

