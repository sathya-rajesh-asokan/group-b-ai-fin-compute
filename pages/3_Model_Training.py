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

st.markdown("# Model Training and Results based on Linear Regression")
st.sidebar.markdown('## Dependant Stock')
st.sidebar.markdown('300750.SZ')
st.sidebar.write("This is the stock you want to predict")
st.sidebar.divider()
st.sidebar.markdown('## Features')
st.sidebar.markdown('- Adjusted Close Price of 1211.HK')


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

y_tickers = [{'ticker':'1211.HK','measure':'Adjusted Close Price'}]

end_date = datetime.now()
start_date = end_date - timedelta(days=90)

modeling_base_data = get_stock_data(x_ticker['ticker'], start_date, end_date, x_ticker['measure'])


for ticker in y_tickers:
    ticker_data = get_stock_data(ticker['ticker'], start_date, end_date, ticker['measure'])
    modeling_base_data = pd.merge(modeling_base_data, ticker_data, left_index=True, right_index=True, how='left')
    if ticker['measure'] == 'Adjusted Close Price':
        modeling_base_data[ticker['ticker'] + '_adjClose'] = modeling_base_data[ticker['ticker'] + '_adjClose'].ffill()
        modeling_base_data[ticker['ticker'] + '_adjClose'] = modeling_base_data[ticker['ticker'] + '_adjClose'].fillna(0)

# Display the DataFrame as a table in Streamlit
st.markdown("### Modeling Base Data")
st.dataframe(modeling_base_data.head())

# Feature Scaling
scaler = StandardScaler()
scaled_modeling_base_data = scaler.fit_transform(modeling_base_data)

modeling_base_data = pd.DataFrame(scaled_modeling_base_data, columns=modeling_base_data.columns, index=modeling_base_data.index)

# Visualizing Stock Trends
plt.figure(figsize=(12, 6))
for col in modeling_base_data.columns:
    plt.plot(modeling_base_data.index, modeling_base_data[col], label=col)
plt.title("Scaled Adjusted Closing Prices of Tech Stocks")
plt.xlabel("Date")
plt.ylabel("Scaled Adjusted Closing Price")
plt.legend()

st.pyplot(plt)

  # Correlation Matrix
correlation_matrix = modeling_base_data.corr()
plt.figure(figsize=(8, 6))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
tick_marks = np.arange(len(correlation_matrix.columns))
plt.xticks(tick_marks, correlation_matrix.columns, rotation=45)
plt.yticks(tick_marks, correlation_matrix.columns)
plt.title("Correlation Matrix of External features")
st.pyplot(plt)

# Stock price trends
for i in [3, 10]:
  modeling_base_data[x_ticker['ticker'] + '_' + str(i) + 'MA'] = modeling_base_data[x_ticker['ticker'] + '_adjClose'].rolling(window=i).mean().bfill()

plt.figure(figsize=(12, 6))
plt.plot(modeling_base_data[x_ticker['ticker'] + '_adjClose'], label=x_ticker['ticker'] + '_adjClose', color='blue')

for i in [3, 10]:
  plt.plot(modeling_base_data[x_ticker['ticker'] + '_' + str(i) + 'MA'], label=x_ticker['ticker'] + '_' + str(i) + 'MA')


#modeling_base_data[f"{x_ticker['ticker']}_RSI"] = calculate_rsi(modeling_base_data[f"{x_ticker['ticker']}_adjClose"]).bfill()
#modeling_base_data[f"{x_ticker['ticker']}_BB_Upper"], modeling_base_data[f"{x_ticker['ticker']}_BB_Lower"] = calculate_bollinger_bands(modeling_base_data[f"{x_ticker['ticker']}_adjClose"])
#modeling_base_data[f"{x_ticker['ticker']}_BB_Upper"], modeling_base_data[f"{x_ticker['ticker']}_BB_Lower"] = modeling_base_data[f"{x_ticker['ticker']}_BB_Upper"], modeling_base_data[f"{x_ticker['ticker']}_BB_Lower"].bfill()
for i in range(1, 6):  # Creating lagged features for up to 5 days
    modeling_base_data[f"{x_ticker['ticker']}_Lagged_adjClose_{i}_days"] = modeling_base_data[f"{x_ticker['ticker']}_adjClose"].shift(i).bfill()



# Lagged Features (Past Prices)
for ticker in y_tickers:
    if ticker['measure'] == 'Adjusted Close Price':
        #modeling_base_data[f"{ticker['ticker']}_RSI"] = calculate_rsi(modeling_base_data[f"{ticker['ticker']}_adjClose"]).bfill()
        #modeling_base_data[f"{ticker['ticker']}_BB_Upper"], modeling_base_data[f"{ticker['ticker']}_BB_Lower"] = calculate_bollinger_bands(modeling_base_data[f"{ticker['ticker']}_adjClose"])
        #modeling_base_data[f"{ticker['ticker']}_BB_Upper"], modeling_base_data[f"{ticker['ticker']}_BB_Lower"] = modeling_base_data[f"{ticker['ticker']}_BB_Upper"], modeling_base_data[f"{ticker['ticker']}_BB_Lower"].bfill()
        for i in range(1, 6):  # Creating lagged features for up to 5 days
            modeling_base_data[f"{ticker['ticker']}_Lagged_adjClose_{i}_days"] = modeling_base_data[f"{ticker['ticker']}_adjClose"].shift(i).bfill()

modeling_base_data['Day_of_Week'] = modeling_base_data.index.dayofweek
modeling_base_data['Month'] = modeling_base_data.index.month
modeling_base_data['Year'] = modeling_base_data.index.year



plt.title("300750.SZ Scaled Adjusted Closing Price with Moving Averages")
plt.xlabel("Date")
plt.ylabel("Scaled Adjusted Closing Price")
plt.legend()
st.pyplot(plt)

# Analyzing Relationships
for ticker in y_tickers:
    plt.figure(figsize=(10, 6))
    if ticker['measure'] == 'Adjusted Close Price':
        plt.scatter(modeling_base_data[x_ticker['ticker'] + '_adjClose'], modeling_base_data[ticker['ticker'] + '_adjClose'])
        plt.title(f"Scatter Plot: {x_ticker['ticker']} vs. {ticker['ticker']} Scaled Adjusted Closing Prices")
        plt.xlabel(x_ticker['ticker'])
        plt.ylabel(ticker['ticker'])
    st.pyplot(plt)



# Distributions
plt.figure(figsize=(10, 6))
plt.hist(modeling_base_data[x_ticker['ticker'] + '_adjClose'], bins=30)
plt.title(f"Distribution of {x_ticker['ticker']} Scaled Adjusted Closing Prices")
plt.xlabel("Scaled Adjusted Closing Price")
plt.ylabel("Frequency")
st.pyplot(plt)

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

# Making Predictions
y_pred = model.predict(X_test.values)

# Model Evaluation and Refinement

# Regression Metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

st.write(f"Mean Squared Error (MSE): {mse:.4f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
st.write(f"R-squared (R2): {r2:.4f}")

# Residual Analysis
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_test, residuals)
plt.xlabel("Actual Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
st.pyplot(plt)

# Regularization Techniques (Ridge and Lasso)
ridge_model = Ridge(alpha=1.0)  # Example alpha value
lasso_model = Lasso(alpha=0.1)  # Example alpha value

ridge_model.fit(X_train.values, y_train.values)
lasso_model.fit(X_train.values, y_train.values)

# Predictions using regularized models
y_pred_ridge = ridge_model.predict(X_test.values)
y_pred_lasso = lasso_model.predict(X_test.values)

# Evaluating regularized models (example with RMSE)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))

st.write(f"RMSE (Ridge): {rmse_ridge:.4f}")
st.write(f"RMSE (Lasso): {rmse_lasso:.4f}")

# Backtesting for Model Robustness
# Sliding Window Approach
window_size = 3  # Define the size of the rolling window
predictions = []
actual_prices = []

for i in range(window_size, len(modeling_base_data)):
    # Select data for the current window
    window_data = modeling_base_data.iloc[i - window_size:i]

    # Separate features and target for the window
    X_window = window_data.drop(columns=f"{x_ticker['ticker'] + '_adjClose'}")
    y_window = window_data[f"{x_ticker['ticker'] + '_adjClose'}"].shift(-1).ffill()

    # Split the window data into train and test sets
    X_train_window = X_window[:-1]
    X_test_window = X_window[-1:]
    y_train_window = y_window[:-1]

    # Handle potential NaNs introduced by shifting in features
    X_train_window = X_train_window.bfill()
    X_test_window = X_test_window.bfill()
    y_train_window = y_train_window.bfill()

    # Train the model on the window data
    model.fit(X_train_window.values, y_train_window.values)

    # Make a prediction for the next day
    next_day_prediction = model.predict(X_test_window.values)

    # Store the prediction and actual price
    predictions.append(next_day_prediction[0])
    actual_prices.append(modeling_base_data[f"{x_ticker['ticker'] + '_adjClose'}"].iloc[i])

# Create a DataFrame to store the backtest results
backtest_df = pd.DataFrame({'Actual': actual_prices, 'Predicted': predictions}, index=modeling_base_data.index[window_size:])

# Visualizing Backtest Results
plt.figure(figsize=(12, 6))
plt.plot(backtest_df['Actual'], label='Actual Prices')
plt.plot(backtest_df['Predicted'], label='Predicted Prices')
plt.title(
    f"Backtesting Results for {f"{x_ticker['ticker'] + '_adjClose'}"} - Sliding Window Approach")
plt.xlabel("Date")
plt.ylabel("Scaled Adjusted Closing Price")
plt.legend()
st.pyplot(plt)


# Feature importance (coefficients)
importance = pd.DataFrame({'Feature': features.columns,'Coefficient': model.coef_}).sort_values('Coefficient', ascending=False)

#st.write("\nFeature Importance:")
#st.write(importance)

# Plot feature importance
plt.figure(figsize=(10, 5))
sns.barplot(x='Coefficient', y='Feature', data=importance, palette='viridis')
plt.title("Linear Regression Feature Importance")
st.pyplot(plt)

iterations_to_predict = 30

prediction_total_data = modeling_base_data.copy()
last_current_date = modeling_base_data.index[0]
st.write(f"Predicting From: {last_current_date}")

for eachDay in range(1, iterations_to_predict + 1):
    prediction_features = prediction_total_data.drop(columns=[f"{x_ticker['ticker'] + '_adjClose'}"], errors='ignore')

    # Get the most recent data point
    latest_data = prediction_features.iloc[-1].values.reshape(1, -1)

    # Predict the next day's price
    pred_price = model.predict(latest_data)

    new_row = pd.DataFrame({f"{x_ticker['ticker'] + '_adjClose'}": pred_price}, index=[features.index[0] + timedelta(days=eachDay)])
    new_row['date'] = pd.to_datetime(new_row.index, format='%Y-%m-%d %H:%M:%S')  #
    new_row.set_index('date', inplace=True, drop=True)


    prediction_total_data = pd.concat([prediction_total_data, new_row], axis=0)
    prediction_total_data = prediction_total_data.sort_index(ascending=False)

    for i in [3, 10]:
      prediction_total_data[x_ticker['ticker'] + '_' + str(i) + 'MA'] = prediction_total_data[x_ticker['ticker'] + '_adjClose'].rolling(window=i).mean().bfill()


    #prediction_total_data[f"{x_ticker['ticker']}_RSI"] = calculate_rsi(prediction_total_data[f"{x_ticker['ticker']}_adjClose"]).bfill()
    #prediction_total_data[f"{x_ticker['ticker']}_BB_Upper"], prediction_total_data[f"{x_ticker['ticker']}_BB_Lower"] = calculate_bollinger_bands(prediction_total_data[f"{x_ticker['ticker']}_adjClose"])
    #prediction_total_data[f"{x_ticker['ticker']}_BB_Upper"], prediction_total_data[f"{x_ticker['ticker']}_BB_Lower"] = prediction_total_data[f"{x_ticker['ticker']}_BB_Upper"], prediction_total_data[f"{x_ticker['ticker']}_BB_Lower"].bfill()
    for i in range(1, 6):  # Creating lagged features for up to 5 days
      prediction_total_data[f"{x_ticker['ticker']}_Lagged_adjClose_{i}_days"] = prediction_total_data[f"{x_ticker['ticker']}_adjClose"].shift(i).bfill()

    for ticker in y_tickers:
      if ticker['measure'] == 'Adjusted Close Price':
        #prediction_total_data[f"{ticker['ticker']}_RSI"] = calculate_rsi(prediction_total_data[f"{ticker['ticker']}_adjClose"]).bfill()
        #prediction_total_data[f"{ticker['ticker']}_BB_Upper"], prediction_total_data[f"{ticker['ticker']}_BB_Lower"] = calculate_bollinger_bands(prediction_total_data[f"{ticker['ticker']}_adjClose"])
        #prediction_total_data[f"{ticker['ticker']}_BB_Upper"], prediction_total_data[f"{ticker['ticker']}_BB_Lower"] = prediction_total_data[f"{ticker['ticker']}_BB_Upper"], prediction_total_data[f"{ticker['ticker']}_BB_Lower"].bfill()
        prediction_total_data.bfill()
        for i in range(1, 6):  # Creating lagged features for up to 5 days
            prediction_total_data[f"{ticker['ticker']}_Lagged_adjClose_{i}_days"] = prediction_total_data[f"{ticker['ticker']}_adjClose"].shift(i).bfill()

    prediction_total_data['Day_of_Week'] = prediction_total_data.index.dayofweek
    prediction_total_data['Month'] = prediction_total_data.index.month
    prediction_total_data['Year'] = prediction_total_data.index.year


# Later, to inverse transform only those columns
final_chart_data = prediction_total_data[f"{x_ticker['ticker']}_adjClose"]
for ticker in y_tickers:
  final_chart_data = pd.merge(final_chart_data, prediction_total_data[f"{ticker['ticker']}_adjClose"], left_index=True, right_index=True, how='left')
st.dataframe(final_chart_data)
unscaled_final_chart_data = scaler.inverse_transform(final_chart_data)
final_chart_data = pd.DataFrame(unscaled_final_chart_data, columns=final_chart_data.columns, index=final_chart_data.index)
st.dataframe(final_chart_data)
# Define the split index (last index of past values)

# Separate past and predicted values
past_values = final_chart_data.loc[last_current_date:]
predicted_values = final_chart_data.loc[:last_current_date]

# Plot the chart
plt.figure(figsize=(12, 6))
plt.plot(past_values.index, past_values[f"{x_ticker['ticker']}_adjClose"], label="Predicted Values", color="blue")
plt.plot(predicted_values.index, predicted_values[f"{x_ticker['ticker']}_adjClose"], label="Past Values", color="orange", linestyle="--")

# Add labels, title, and legend
plt.title(f"{x_ticker['ticker']} Adjusted Close Price: Past vs Predicted")
plt.xlabel("Date")
plt.ylabel("Adjusted Close Price")
plt.legend()

# Display the chart in Streamlit
st.pyplot(plt)
