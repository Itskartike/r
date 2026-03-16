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
