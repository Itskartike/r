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
