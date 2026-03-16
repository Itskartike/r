# IsolationForest method
from sklearn.ensemble import IsolationForest
import pandas as pd
df = pd.DataFrame({
 'Amount': [1200, 1450, 1300, 1500, 1400, 12000, 80]
})
model = IsolationForest(contamination=0.2, random_state=42)
df['Anomaly'] = model.fit_predict(df)
print(df)


'''
An online shopping platform wants to detect unusual order amounts that may indicate suspicious purchasing behavior without using labeled fraud data. Apply Isolation Forest method for the same
Order_ID	Order_Amount (₹)
O101	1,200
O102	1,450
O103	1,300
O104	1,500
O105	1,400
O106	12,000
O107	80
'''

