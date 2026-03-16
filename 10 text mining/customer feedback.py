from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
feedback = [
    "Excellent service and friendly staff",
    "Very disappointed with the delivery",
    "Product quality is fantastic",
    "Customer support was terrible"
]
sentiment = ["Positive", "Negative", "Positive", "Negative"]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(feedback)
model = LogisticRegression()
model.fit(X, sentiment)
print(
    model.predict(
    vectorizer.transform(["Service was very good"])
    )
)
