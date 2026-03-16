# Aim: Time Series Forecasting using ARIMA

# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Step 2: Load the Dataset
data = pd.read_csv(r'D:\Sujal\bin\dsf\8 time series forecasting\AirPassengers.csv')

data['Month'] = pd.to_datetime(data['Month'])
data.set_index('Month', inplace=True)

print(data.head())

# Step 3: Visualize the Time Series
plt.figure()
plt.plot(data['Passengers'])
plt.title('Monthly Air Passengers')
plt.xlabel('Year')
plt.ylabel('Number of Passengers')
plt.show()

# Step 4: Check Stationarity (ADF Test)
result = adfuller(data['Passengers'])
print("ADF Statistic:", result[0])
print("p-value:", result[1])

# Step 5: Make Series Stationary (First Order Differencing)
data_diff = data['Passengers'].diff().dropna()

plt.figure()
plt.plot(data_diff)
plt.title('First Differenced Series')
plt.show()

# ADF Test Again
result_diff = adfuller(data_diff)
print("ADF Statistic after Differencing:", result_diff[0])
print("p-value after Differencing:", result_diff[1])

# Step 6: Identify ARIMA Parameters (p,d,q)
# d = 1 (from differencing)

plot_pacf(data_diff, lags=20)
plt.show()

plot_acf(data_diff, lags=20)
plt.show()

# Step 7: Build ARIMA Model (Example: 1,1,1)
model = ARIMA(data['Passengers'], order=(1,1,1))
model_fit = model.fit()

print(model_fit.summary())

# Step 8: Forecast Future Values (Next 12 Months)
forecast = model_fit.forecast(steps=12)

print("Forecasted Values:")
print(forecast)

# Step 9: Plot Actual vs Forecast
plt.figure()
plt.plot(data['Passengers'], label='Actual')
plt.plot(forecast, label='Forecast', color='red')
plt.legend()
plt.title("Actual vs Forecast")
plt.show()
