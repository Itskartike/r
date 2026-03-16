
from sklearn.ensemble import IsolationForest
import pandas as pd
df = pd.DataFrame({
    'Amount': [2800, 3200, 3500, 2900, 3100, 18000, 50]
})
model = IsolationForest(contamination=0.2, random_state=42)
df['Anomaly'] = model.fit_predict(df)
print(df)



'''
A bank wants to detect unusual transaction amounts that may indicate fraudulent activity without using labeled fraud data. Use IsolationForest method
Transaction_ID	Transaction_Amount (₹)
T01	2,800
T02	3,200
T03	3,500
T04	2,900
T05	3,100
T06	18,000
T07	50
'''

