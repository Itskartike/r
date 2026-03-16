#AIM:Text mining
1.	Document Classification: How can text mining be used to classify documents into different categories?
Problem An organization wants to automatically classify emails into categories such as Work and Personal based on their content.
Sample Data: Email 1: Meeting scheduled with project team
Email 2: Family dinner this weekend
Email 3: Project deadline extended
Email 4: Birthday party invitation
Code:  

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


2.	Sentiment Analysis: How can text mining be applied to analyze customer sentiment from reviews?
Problem Statement An e-commerce company wants to determine whether customer reviews are positive or negative.
Sample Data: Review 1: The product is amazing
Review 2: Very bad quality
Review 3: I am happy with the purchase
Review 4: Worst experience ever
Code:  

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


3.	Search Engines: How does text mining help search engines retrieve relevant documents?
Problem Statement: A search system needs to find the most relevant document for a given user query.
Sample Data: Doc1: Data science and machine learning
Doc2: Introduction to text mining
Doc3: Python for data analysis
Code: 

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

4.	Spam Detection: How can text mining be used to detect spam messages?
Problem Statement: An email system wants to identify spam messages automatically.
Sample Data: Message 1: Win a free lottery now
Message 2: Meeting scheduled tomorrow
Message 3: Urgent offer claim now
Message 4: Project discussion today
Code: 

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


5.	Recommendation Systems: How is text mining used in recommendation systems?
Problem Statement: A movie platform wants to recommend movies based on movie descriptions.
Sample Data: Movie 1: Actio   n adventure and hero story
Movie 2: Romantic love story
Movie 3: Adventure and fantasy world
Code:  

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


6.	A news website wants to automatically classify articles into Sports and Politics based on content.

Article_ID	Content
A1	Football match ends with thrilling goal
A2	Government passes new education bill
A3	Cricket world cup semi-final today
A4	Election campaigns begin in several states
Code:

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


7.	A company wants to determine whether customer feedback is Positive or Negative.

Feedback_ID	Feedback
F1	Excellent service and friendly staff
F2	Very disappointed with the delivery
F3	Product quality is fantastic
F4	Customer support was terrible
Code:

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

8.	A library wants to retrieve the most relevant book description for a user search query.
Doc_ID	Description
D1	Introduction to Python programming
D2	Advanced machine learning concepts
D3	Basics of data science
Code:

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
print("Most relevant document index:",
      np.argmax(similarity[-1][:-1]))


9.	An email system wants to classify messages as Spam or Not Spam automatically.
Message_ID	Message
M1	Congratulations! You won a free gift
M2	Team meeting scheduled for tomorrow
M3	Claim your free prize now
M4	Project update report
Code:

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

10.	A movie streaming platform wants to suggest movies to users based on movie descriptions.
Movie_ID	Description
MV1	Romantic comedy with a fun storyline
MV2	Action thriller with suspense
MV3	Adventure in a magical world
Code:

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
