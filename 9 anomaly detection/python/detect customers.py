from sklearn.neighbors import LocalOutlierFactor
import pandas as pd
df = pd.DataFrame({
    'Spending_Score': [45, 50, 48, 52, 49, 98, 5],
    'Purchase_Frequency': [18, 20, 22, 19, 21, 65, 2]
})
lof = LocalOutlierFactor(n_neighbors=3)
df['Outlier'] = lof.fit_predict(df)
outliers = df[df['Outlier'] == -1] 
print(outliers) 




'''
An e-commerce company wants to detect customers whose purchasing behavior significantly deviates from that of their local neighborhood. Use Local Outlier Factor (LOF)
Customer_ID	Spending_Score	Purchase_Frequency
CU01	45	18
CU02	50	20
CU03	48	22
CU04	52	19
CU05	49	21
CU06	98	65
CU07	5	2
''' 
