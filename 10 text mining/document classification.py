from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
texts = [ 
    "Meeting scheduled with project team",
    "Family dinner this weekend",
    "Project deadline extended",
    "Birthday party invitation"
]
labels = ["Work", "Personal", "Work", "Personal"]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
model = MultinomialNB()
model.fit(X, labels)
prediction = model.predict(vectorizer.transform(["Project meeting tomorrow"]))
print(prediction)
