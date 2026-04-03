import pandas as pd
import sqlite3
from statsmodels.tsa.arima.model import ARIMA

# Load data from SQL
conn = sqlite3.connect("supply_chain.db")
df = pd.read_sql("SELECT * FROM sales_data", conn)

# Convert Date column
df['Date'] = pd.to_datetime(df['Date'])

# Aggregate daily sales
df = df.groupby('Date')['sales'].sum().reset_index()

# Set index
df.set_index('Date', inplace=True)

print(df.head())

# Build ARIMA model
model = ARIMA(df['sales'], order=(5,1,0))
model_fit = model.fit()

# Forecast next 30 days
forecast = model_fit.forecast(steps=30)

print("\nForecast for next 30 days:")
print(forecast)

import matplotlib.pyplot as plt

# Plot historical data
plt.figure(figsize=(12,6))
plt.plot(df.index, df['sales'], label='Actual Sales')

# Create future dates
forecast_index = pd.date_range(start=df.index[-1], periods=30, freq='D')

# Plot forecast
plt.plot(forecast_index, forecast, label='Forecast')

# Labels
plt.legend()
plt.title("Sales Forecast vs Actual")
plt.xlabel("Date")
plt.ylabel("Sales")

plt.show()

# Save historical data
df.to_csv("historical_data.csv")

# Save forecast
forecast_df = forecast.reset_index()
forecast_df.columns = ['Date', 'Forecast']
forecast_df.to_csv("forecast_data.csv", index=False)

# Save historical data
df.to_csv("historical_data.csv")

# Save forecast data
forecast_df = forecast.reset_index()
forecast_df.columns = ['Date', 'Forecast']
forecast_df.to_csv("forecast_data.csv", index=False)

print("Files exported successfully!")