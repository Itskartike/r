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
