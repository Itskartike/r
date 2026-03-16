from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
messages = [
    "Congratulations! You won a free gift",
    "Team meeting scheduled for tomorrow",
    "Claim your free prize now",
    "Project update report"
]
labels = ["Spam", "Not Spam", "Spam", "Not Spam"]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(messages)
model = MultinomialNB()
model.fit(X, labels)
print(
    model.predict(
        vectorizer.transform(["Free prize waiting for you"])
    )
)
