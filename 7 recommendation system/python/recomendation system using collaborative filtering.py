
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
