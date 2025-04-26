import streamlit as st

import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso
import seaborn as sns
y_tickers = []


st.markdown("# Stock Price Prediction")
st.sidebar.markdown('## Simulation Date')
end_date = st.sidebar.date_input("", value="today")
st.sidebar.markdown('## Dependant Stock')
st.sidebar.markdown('300750.SZ')
st.sidebar.markdown('## Number of Days to Predict')
iterations_to_predict = st.sidebar.number_input("0", min_value=0, max_value=30, value=30, step=1, label_visibility ="hidden")
st.sidebar.divider()

st.sidebar.markdown('## Variation of the Features')
st.sidebar.markdown("- 1211.HK")
y_tickers.append({'ticker':"1211.HK",'measure':'Adjusted Close Price','change':st.sidebar.number_input("1", min_value=-100.00, max_value=100.00, value=0.00, format="%0.2f", step=0.10, label_visibility ="hidden")})

st.sidebar.markdown("- 300207.SZ")
y_tickers.append({'ticker':"300207.SZ",'measure':'Adjusted Close Price','change':st.sidebar.number_input("2", min_value=-100.00, max_value=100.00, value=0.00, format="%0.2f", step=0.10, label_visibility ="hidden")})

st.sidebar.markdown("- 002074.SZ")
y_tickers.append({'ticker':"002074.SZ",'measure':'Adjusted Close Price','change':st.sidebar.number_input("3", min_value=-100.00, max_value=100.00, value=0.00, format="%0.2f", step=0.10, label_visibility ="hidden")})

st.sidebar.markdown("- 300014.SZ")
y_tickers.append({'ticker':"300014.SZ",'measure':'Adjusted Close Price','change':st.sidebar.number_input("4", min_value=-100.00, max_value=100.00, value=0.00, format="%0.2f", step=0.10, label_visibility ="hidden")})

st.sidebar.markdown("- 000100.SZ")
y_tickers.append({'ticker':"000100.SZ",'measure':'Adjusted Close Price','change':st.sidebar.number_input("5", min_value=-100.00, max_value=100.00, value=0.00, format="%0.2f", step=0.10, label_visibility ="hidden")})
st.sidebar.divider()
st.sidebar.markdown('## Stock Internal Features')

internal_features = st.sidebar.multiselect("", ["RSI","Bollinger Bands","5 Day Lagged Price"])

# Simple function to retrieve data from FMP
def get_stock_data(_ticker,_start_date,_end_date, _metric):
    end_point='https://financialmodelingprep.com/stable/historical-price-eod/dividend-adjusted'
    api_key='4ZTUtST6urFlN83a1LDFy6U7plAHhegP'

    params = {}
    params['apikey'] = api_key
    params['symbol'] = _ticker
    params['from'] = _start_date.strftime('%Y-%m-%d')
    params['to'] = _end_date.strftime('%Y-%m-%d')

    fmp_response=requests.get(end_point, params=params)

    # Convert JSON to Dataframes
    df = pd.DataFrame(fmp_response.json())

    # I am setting the index to date, so as to be able to calcualte pct change and join
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')  #
    df.set_index('date', inplace=True, drop=True)

    df = df.drop(columns=['symbol'], errors='ignore')

    df.columns = [_ticker + '_' + col for col in df.columns.values]

    if 'Adjusted Close Price' == _metric:
        df = df[[_ticker + '_adjClose']]
    else:
        return df

    return df

# Feature Engineering for Predictive Power
def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(data, window=20, std_dev=2):
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * std_dev)
    lower_band = rolling_mean - (rolling_std * std_dev)
    return upper_band, lower_band

x_ticker = {'ticker': '300750.SZ', 'measure':'Adjusted Close Price'}

start_date = end_date - timedelta(days=90)

modeling_base_data = get_stock_data(x_ticker['ticker'], start_date, end_date, x_ticker['measure'])


for ticker in y_tickers:
    ticker_data = get_stock_data(ticker['ticker'], start_date, end_date, ticker['measure'])
    modeling_base_data = pd.merge(modeling_base_data, ticker_data, left_index=True, right_index=True, how='left')
    if ticker['measure'] == 'Adjusted Close Price':
        modeling_base_data[ticker['ticker'] + '_adjClose'] = modeling_base_data[ticker['ticker'] + '_adjClose'].ffill()

# Feature Scaling
scaler = StandardScaler()
scaled_modeling_base_data = scaler.fit_transform(modeling_base_data)

modeling_base_data = pd.DataFrame(scaled_modeling_base_data, columns=modeling_base_data.columns, index=modeling_base_data.index)

model_results_title = st.empty()
model_results = st.empty()
model_results_explanation_1 = st.empty()
model_results_explanation_2 = st.empty()

# Stock price trends
for i in [3, 10]:
  modeling_base_data[x_ticker['ticker'] + '_' + str(i) + 'MA'] = modeling_base_data[x_ticker['ticker'] + '_adjClose'].rolling(window=i).mean().bfill()

plt.figure(figsize=(12, 6))
plt.plot(modeling_base_data[x_ticker['ticker'] + '_adjClose'], label=x_ticker['ticker'] + '_adjClose', color='blue')

for i in [3, 10]:
  plt.plot(modeling_base_data[x_ticker['ticker'] + '_' + str(i) + 'MA'], label=x_ticker['ticker'] + '_' + str(i) + 'MA')

if 'RSI' in internal_features:
  modeling_base_data[f"{x_ticker['ticker']}_RSI"] = calculate_rsi(modeling_base_data[f"{x_ticker['ticker']}_adjClose"]).bfill()
elif 'Bollinger Bands' in internal_features:
  modeling_base_data[f"{x_ticker['ticker']}_BB_Upper"], modeling_base_data[f"{x_ticker['ticker']}_BB_Lower"] = calculate_bollinger_bands(modeling_base_data[f"{x_ticker['ticker']}_adjClose"])
  modeling_base_data[f"{x_ticker['ticker']}_BB_Upper"], modeling_base_data[f"{x_ticker['ticker']}_BB_Lower"] = modeling_base_data[f"{x_ticker['ticker']}_BB_Upper"], modeling_base_data[f"{x_ticker['ticker']}_BB_Lower"].bfill()
if '5 Day Lagged Price' in internal_features:
  for i in range(1, 6):  # Creating lagged features for up to 5 days
    modeling_base_data[f"{x_ticker['ticker']}_Lagged_adjClose_{i}_days"] = modeling_base_data[f"{x_ticker['ticker']}_adjClose"].shift(i).bfill()

# Lagged Features (Past Prices)
for ticker in y_tickers:
    if ticker['measure'] == 'Adjusted Close Price':
        if 'RSI' in internal_features:
          modeling_base_data[f"{ticker['ticker']}_RSI"] = calculate_rsi(modeling_base_data[f"{ticker['ticker']}_adjClose"]).bfill()
        elif 'Bollinger Bands' in internal_features:
          modeling_base_data[f"{ticker['ticker']}_BB_Upper"], modeling_base_data[f"{ticker['ticker']}_BB_Lower"] = calculate_bollinger_bands(modeling_base_data[f"{ticker['ticker']}_adjClose"])
          modeling_base_data[f"{ticker['ticker']}_BB_Upper"], modeling_base_data[f"{ticker['ticker']}_BB_Lower"] = modeling_base_data[f"{ticker['ticker']}_BB_Upper"], modeling_base_data[f"{ticker['ticker']}_BB_Lower"].bfill()
        if '5 Day Lagged Price' in internal_features:
          for i in range(1, 6):  # Creating lagged features for up to 5 days
            modeling_base_data[f"{ticker['ticker']}_Lagged_adjClose_{i}_days"] = modeling_base_data[f"{ticker['ticker']}_adjClose"].shift(i).bfill()

modeling_base_data['Day_of_Week'] = modeling_base_data.index.dayofweek
modeling_base_data['Month'] = modeling_base_data.index.month
modeling_base_data['Year'] = modeling_base_data.index.year


# some logic here
target = modeling_base_data[f"{x_ticker['ticker'] + '_adjClose'}"].shift(-1).ffill()

# Features (excluding the target stock's current day's data)
features = modeling_base_data.drop(columns=[f"{x_ticker['ticker'] + '_adjClose'}"], errors='ignore')


# Splitting the Data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Handling Missing Values in Features
X_train = X_train.ffill()
X_test = X_test.ffill()

# Creating and Training the Model
model = LinearRegression()
model.fit(X_train.values, y_train.values)


prediction_total_data = modeling_base_data.copy()
last_current_date = modeling_base_data.index[0]
st.write(f"Predicting From: {last_current_date}")

for eachDay in range(1, iterations_to_predict + 1):

    prediction_features = prediction_total_data.drop(columns=[f"{x_ticker['ticker'] + '_adjClose'}"], errors='ignore')

    latest_data = prediction_features.iloc[0]

    # Get the most recent data point
    latest_data_for_prediction = latest_data.values.reshape(1, -1)

    # Predict the next day's price
    pred_price = model.predict(latest_data_for_prediction)

    new_row = pd.DataFrame({f"{x_ticker['ticker'] + '_adjClose'}": pred_price}, index=[features.index[0] + timedelta(days=eachDay)])
    new_row['date'] = pd.to_datetime(new_row.index, format='%Y-%m-%d %H:%M:%S')  #
    new_row.set_index('date', inplace=True, drop=True)

    for ticker in y_tickers:
        if ticker['measure'] == 'Adjusted Close Price':
            daily_increase = ticker['change'] / iterations_to_predict
            new_row[ticker['ticker'] + '_adjClose'] = latest_data[ticker['ticker'] + '_adjClose'] * (1 + (daily_increase)/100)



    prediction_total_data = pd.concat([prediction_total_data, new_row], axis=0)
    prediction_total_data = prediction_total_data.sort_index(ascending=False)
    prediction_total_data = prediction_total_data.bfill()

    for i in [3, 10]:
      prediction_total_data[x_ticker['ticker'] + '_' + str(i) + 'MA'] = prediction_total_data[x_ticker['ticker'] + '_adjClose'].rolling(window=i).mean().bfill()

    if 'RSI' in internal_features:
      prediction_total_data[f"{x_ticker['ticker']}_RSI"] = calculate_rsi(prediction_total_data[f"{x_ticker['ticker']}_adjClose"]).bfill()
    elif 'Bollinger Bands' in internal_features:
      prediction_total_data[f"{x_ticker['ticker']}_BB_Upper"], prediction_total_data[f"{x_ticker['ticker']}_BB_Lower"] = calculate_bollinger_bands(prediction_total_data[f"{x_ticker['ticker']}_adjClose"])
      prediction_total_data[f"{x_ticker['ticker']}_BB_Upper"], prediction_total_data[f"{x_ticker['ticker']}_BB_Lower"] = prediction_total_data[f"{x_ticker['ticker']}_BB_Upper"], prediction_total_data[f"{x_ticker['ticker']}_BB_Lower"].bfill()
    if '5 Day Lagged Price' in internal_features:
      for i in range(1, 6):  # Creating lagged features for up to 5 days
        prediction_total_data[f"{x_ticker['ticker']}_Lagged_adjClose_{i}_days"] = prediction_total_data[f"{x_ticker['ticker']}_adjClose"].shift(i).bfill()

    for ticker in y_tickers:
      if ticker['measure'] == 'Adjusted Close Price':
        if 'RSI' in internal_features:
          prediction_total_data[f"{ticker['ticker']}_RSI"] = calculate_rsi(prediction_total_data[f"{ticker['ticker']}_adjClose"]).bfill()
        elif 'Bollinger Bands' in internal_features:
          prediction_total_data[f"{ticker['ticker']}_BB_Upper"], prediction_total_data[f"{ticker['ticker']}_BB_Lower"] = calculate_bollinger_bands(prediction_total_data[f"{ticker['ticker']}_adjClose"])
          prediction_total_data[f"{ticker['ticker']}_BB_Upper"], prediction_total_data[f"{ticker['ticker']}_BB_Lower"] = prediction_total_data[f"{ticker['ticker']}_BB_Upper"], prediction_total_data[f"{ticker['ticker']}_BB_Lower"].bfill()
          prediction_total_data.bfill()
        if '5 Day Lagged Price' in internal_features:
          for i in range(1, 6):  # Creating lagged features for up to 5 days
            prediction_total_data[f"{ticker['ticker']}_Lagged_adjClose_{i}_days"] = prediction_total_data[f"{ticker['ticker']}_adjClose"].shift(i).bfill()

    prediction_total_data['Day_of_Week'] = prediction_total_data.index.dayofweek
    prediction_total_data['Month'] = prediction_total_data.index.month
    prediction_total_data['Year'] = prediction_total_data.index.year

# Later, to inverse transform only those columns
final_chart_data = prediction_total_data[f"{x_ticker['ticker']}_adjClose"]
for ticker in y_tickers:
  final_chart_data = pd.merge(final_chart_data, prediction_total_data[f"{ticker['ticker']}_adjClose"], left_index=True, right_index=True, how='left')

unscaled_final_chart_data = scaler.inverse_transform(final_chart_data)
final_chart_data = pd.DataFrame(unscaled_final_chart_data, columns=final_chart_data.columns, index=final_chart_data.index)

# Separate past and predicted values
past_values = final_chart_data.loc[last_current_date:]
predicted_values = final_chart_data.loc[:last_current_date]

# Plot the chart
plt.figure(figsize=(12, 6))
plt.plot(past_values.index, past_values[f"{x_ticker['ticker']}_adjClose"], label="Past Values", color="blue")
plt.plot(predicted_values.index, predicted_values[f"{x_ticker['ticker']}_adjClose"], label="Predicted Values", color="orange", linestyle="--")

# Add labels, title, and legend
plt.title(f"{x_ticker['ticker']} Adjusted Close Price: Past vs Predicted")
plt.xlabel("Date")
plt.ylabel("Adjusted Close Price")
plt.legend()

# Display the chart in Streamlit
st.pyplot(plt)
