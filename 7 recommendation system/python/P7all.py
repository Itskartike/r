#AIM: Recommendation System
#Step 1: Import Required Libraries
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

#Step 2: Create the Dataset
data = {
    'User': ['U1','U1','U2','U2','U3','U3'],
    'Item': ['I1','I2','I1','I3','I2','I3'],
    'Rating': [5,4,4,5,3,4]
}
df = pd.DataFrame(data)

#Step 3: Create User–Item Rating Matrix
matrix = df.pivot_table(index='User', columns='Item', values='Rating').fillna(0)
print(matrix) 

#Step 4: Compute Item–Item Cosine Similarity
similarity = cosine_similarity(matrix.T)
similarity_df = pd.DataFrame(similarity, index=matrix.columns, columns=matrix.columns)
print(similarity_df)

#Step 5: Define Recommendation Function
def recommend(item):
    return similarity_df[item].sort_values(ascending=False)

#Step 6: Generate Recommendation
print(recommend('I1'))

'''Real-Life Scenario
An e-commerce website wants to recommend products based on user purchase ratings.
Given Data
User	Item	Rating
U1	Mobile	5
U1	Earphones	4
U2	Mobile	4
U2	PowerBank	5
U3	Earphones	3
U3	PowerBank	4
Practical Questions
•	Create a user–item rating matrix.
•	Compute item–item cosine similarity.
•	Recommend items similar to Mobile.
•	Identify which product should be cross-sold with Mobile.
Code:'''
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
# Step 1: Create Dataset
data = {
    'User': ['U1','U1','U2','U2','U3','U3'],
    'Item': ['Mobile','Earphones','Mobile','PowerBank','Earphones','PowerBank'],
    'Rating': [5,4,4,5,3,4]
}
df = pd.DataFrame(data)

# Step 2: Create User-Item Matrix
matrix = df.pivot_table(index='User', columns='Item', values='Rating').fillna(0)
print(matrix)

# Step 3: Compute Cosine Similarity
similarity = cosine_similarity(matrix.T)
similarity_df = pd.DataFrame(similarity, index=matrix.columns, columns=matrix.columns)
print(similarity_df)

# Step 4: Recommendation Function
def recommend(item):
    return similarity_df[item].sort_values(ascending=False)
print(recommend('Mobile'))


###2. Movie Recommendation System
'''Real-Life Scenario
A movie streaming platform wants to suggest movies based on user ratings.
Given Data
User	Movie	Rating
U1	Avengers	5
U1	Titanic	4
U2	Avengers	4
U2	Inception	5
U3	Titanic	3
U3	Inception	4
Practical Questions
•	Construct the user–movie matrix.
•	Calculate cosine similarity between movies.
•	Recommend movies similar to Avengers.
•	Explain how similarity affects recommendation quality.
Code:'''
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
data = {
    'User': ['U1','U1','U2','U2','U3','U3'],
    'Movie': ['Avengers','Titanic','Avengers','Inception','Titanic','Inception'],
    'Rating': [5,4,4,5,3,4]
}

df = pd.DataFrame(data)

matrix = df.pivot_table(index='User', columns='Movie', values='Rating').fillna(0)
print(matrix)

similarity = cosine_similarity(matrix.T)
similarity_df = pd.DataFrame(similarity, index=matrix.columns, columns=matrix.columns)
print(similarity_df)

def recommend(movie):
    return similarity_df[movie].sort_values(ascending=False)
print(recommend('Avengers'))


###3. Music Recommendation (Playlist Suggestion)
'''Real-Life Scenario
A music app wants to suggest songs based on listening preferences.
Given Data
User	Song	Rating
U1	SongA	5
U1	SongB	4
U2	SongA	4
U2	SongC	5
U3	SongB	3
U3	SongC	4
Practical Questions
•	Prepare the user–song matrix.
•	Compute cosine similarity between songs.
•	Recommend songs similar to SongA.
•	Explain how this improves playlist personalization.
Code:'''
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
data = {
    'User': ['U1','U1','U2','U2','U3','U3'],
    'Song': ['SongA','SongB','SongA','SongC','SongB','SongC'],
    'Rating': [5,4,4,5,3,4]
}

df = pd.DataFrame(data)

matrix = df.pivot_table(index='User', columns='Song', values='Rating').fillna(0)
print(matrix)

similarity = cosine_similarity(matrix.T)
similarity_df = pd.DataFrame(similarity, index=matrix.columns, columns=matrix.columns)
print(similarity_df)

def recommend(song):
    return similarity_df[song].sort_values(ascending=False)
print(recommend('SongA'))



###4. Online Course Recommendation
'''Real-Life Scenario
An e-learning platform wants to recommend courses based on learner ratings.
Given Data
Learner	Course	Rating
L1	Python	5
L1	SQL	4
L2	Python	4
L2	ML	5
L3	SQL	3
L3	ML	4
Practical Questions
•	Create learner–course matrix.
•	Apply cosine similarity on courses.
•	Recommend courses related to Python.
•	Identify the most popular course.
Code:'''
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
data = {
    'Learner': ['L1','L1','L2','L2','L3','L3'],
    'Course': ['Python','SQL','Python','ML','SQL','ML'],
    'Rating': [5,4,4,5,3,4]
}

df = pd.DataFrame(data)

matrix = df.pivot_table(index='Learner', columns='Course', values='Rating').fillna(0)
print(matrix)

similarity = cosine_similarity(matrix.T)
similarity_df = pd.DataFrame(similarity, index=matrix.columns, columns=matrix.columns)
print(similarity_df)

def recommend(course):
    return similarity_df[course].sort_values(ascending=False)
print(recommend('Python'))


###5. Restaurant Recommendation (Food App)
'''Real-Life Scenario
A food delivery app recommends restaurants based on ratings.
Given Data
User	Restaurant	Rating
U1	R1	5
U1	R2	4
U2	R1	4
U2	R3	5
U3	R2	3
U3	R3	4
Practical Questions
•	Build user–restaurant matrix.
•	Find restaurant similarity using cosine similarity.
•	Recommend restaurants similar to R1.
•	Discuss how rating sparsity affects recommendation.
Code:'''
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
data = {
    'User': ['U1','U1','U2','U2','U3','U3'],
    'Restaurant': ['R1','R2','R1','R3','R2','R3'],
    'Rating': [5,4,4,5,3,4]
}

df = pd.DataFrame(data)

matrix = df.pivot_table(index='User', columns='Restaurant', values='Rating').fillna(0)
print(matrix)

similarity = cosine_similarity(matrix.T)
similarity_df = pd.DataFrame(similarity, index=matrix.columns, columns=matrix.columns)
print(similarity_df)

def recommend(restaurant):
    return similarity_df[restaurant].sort_values(ascending=False)
print(recommend('R1'))


###6. Book Recommendation System
'''Real-Life Scenario
An online library wants to suggest books based on reading preferences.
Given Data
User	Book	Rating
U1	BookA	5
U1	BookB	4
U2	BookA	4
U2	BookC	5
U3	BookB	3
U3	BookC	4
Practical Questions
•	Generate the user–book matrix.
•	Compute similarity among books.
•	Recommend books related to BookA.
•	Explain cold start problem using this data.
Code:'''
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
data = {
    'User': ['U1','U1','U2','U2','U3','U3'],
    'Book': ['BookA','BookB','BookA','BookC','BookB','BookC'],
    'Rating': [5,4,4,5,3,4]
}

df = pd.DataFrame(data)

matrix = df.pivot_table(index='User', columns='Book', values='Rating').fillna(0)
print(matrix)

similarity = cosine_similarity(matrix.T)
similarity_df = pd.DataFrame(similarity, index=matrix.columns, columns=matrix.columns)
print(similarity_df)

def recommend(book):
    return similarity_df[book].sort_values(ascending=False)

print(recommend('BookA'))

