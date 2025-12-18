import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

symbol = "AAPL"
history_period = "1y"
forecast_days = 30

stock = yf.Ticker(symbol)
hist = stock.history(period=history_period)

print(f"Got {len(hist)} rows")

# Daten vorbereiten - nur numpy arrays verwenden
close_prices = hist["Close"].values
print(f"Close prices type: {type(close_prices)}")

# Daten als Python datetime
dates_clean = []
for d in hist.index:
    dt = d.to_pydatetime()
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    dates_clean.append(dt)

print(f"Dates clean type: {type(dates_clean[0])}")

# DataFrame ohne Timestamp-Index
df = pd.DataFrame({
    "Close": close_prices,
    "Date": dates_clean
})

# ARIMA nur mit numpy array
series = df["Close"].values
print(f"Series type: {type(series)}")

# Test ARIMA
model = ARIMA(series, order=(1, 1, 1))
fitted = model.fit()
print("Model fitted successfully")

# Forecast
forecast_mean = fitted.forecast(steps=forecast_days)
print(f"Forecast type: {type(forecast_mean)}")
print(f"Forecast values: {forecast_mean[:5]}")

# Konfidenzintervall
residuals = fitted.resid
std_err = np.std(residuals)
ci_margin = 1.96 * std_err * np.sqrt(np.arange(1, forecast_days + 1))
forecast_ci_lower = forecast_mean - ci_margin
forecast_ci_upper = forecast_mean + ci_margin
forecast_ci = np.column_stack([forecast_ci_lower, forecast_ci_upper])

print("CI calculated successfully")

# Prognose-Daten
last_date = dates_clean[-1]
print(f"Last date: {last_date}, type: {type(last_date)}")

forecast_date_list = []
current_date = last_date + timedelta(days=1)
while len(forecast_date_list) < forecast_days:
    if current_date.weekday() < 5:
        forecast_date_list.append(current_date)
    current_date = current_date + timedelta(days=1)

print(f"Forecast dates: {len(forecast_date_list)}")
print("All done!")
