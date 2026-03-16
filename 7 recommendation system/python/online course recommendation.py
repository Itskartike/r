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
