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

st.markdown("# Model Training and Analysis")
st.sidebar.markdown('## Simulation Date')
end_date = st.sidebar.date_input("", value="today")
st.sidebar.markdown('## Dependant Stock')
st.sidebar.markdown('300750.SZ')
st.sidebar.write("This is the stock you want to predict")
st.sidebar.divider()
st.sidebar.markdown('## Features')
y_features = st.sidebar.multiselect("", ["1211.HK","300207.SZ","002074.SZ","300014.SZ","000100.SZ"], ["1211.HK"])
st.sidebar.divider()
st.sidebar.markdown('## Stock Internal Features')
internal_features = st.sidebar.multiselect("", ["RSI","5 Day Lagged Price","Bollinger Bands"])

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

def calculate_bollinger_bands(data, window=3, std_dev=2):
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * std_dev)
    lower_band = rolling_mean - (rolling_std * std_dev)
    return upper_band, lower_band

x_ticker = {'ticker': '300750.SZ', 'measure':'Adjusted Close Price'}
y_tickers = []
for y_feature in y_features:
  y_tickers.append({'ticker':y_feature,'measure':'Adjusted Close Price'})

start_date = end_date - timedelta(days=90)

modeling_base_data = get_stock_data(x_ticker['ticker'], start_date, end_date, x_ticker['measure'])


for ticker in y_tickers:
    ticker_data = get_stock_data(ticker['ticker'], start_date, end_date, ticker['measure'])
    modeling_base_data = pd.merge(modeling_base_data, ticker_data, left_index=True, right_index=True, how='left')
    if ticker['measure'] == 'Adjusted Close Price':
        modeling_base_data[ticker['ticker'] + '_adjClose'] = modeling_base_data[ticker['ticker'] + '_adjClose'].ffill()
        modeling_base_data[ticker['ticker'] + '_adjClose'] = modeling_base_data[ticker['ticker'] + '_adjClose'].fillna(0)

# Feature Scaling
scaler = StandardScaler()
scaled_modeling_base_data = scaler.fit_transform(modeling_base_data)

modeling_base_data = pd.DataFrame(scaled_modeling_base_data, columns=modeling_base_data.columns, index=modeling_base_data.index)

model_results_title = st.empty()
model_results = st.empty()
model_results_explanation_1 = st.empty()
model_results_explanation_2 = st.empty()
# Visualizing Stock Trends
plt.figure(figsize=(12, 6))
for col in modeling_base_data.columns:
    plt.plot(modeling_base_data.index, modeling_base_data[col], label=col)
plt.title("Scaled Adjusted Closing Prices of the Stocks")
plt.xlabel("Date")
plt.ylabel("Scaled Adjusted Closing Price")
plt.legend()
st.pyplot(plt)
st.write("#### Why use Scaled Adjusted Closing Prices?")
st.write("The adjusted closing price is a stock's closing price after accounting for all applicable splits and dividend distributions. It provides a more accurate reflection of the stock's value over time, especially for long-term analysis. By scaling these prices, we can better visualize and compare trends across different stocks, making it easier to identify patterns and correlations.")

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
st.write("#### Why use Correlation Matrix?")
st.write("A correlation matrix is a table that shows the correlation coefficients between many variables. Each cell in the table displays the correlation between two variables. The value ranges from -1 to 1, where -1 indicates a strong negative correlation, 0 indicates no correlation, and 1 indicates a strong positive correlation. This helps in understanding the relationships between different stocks and can guide investment decisions.")

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
  modeling_base_data[f"{x_ticker['ticker']}_BB_Upper"], modeling_base_data[f"{x_ticker['ticker']}_BB_Lower"] = modeling_base_data[f"{x_ticker['ticker']}_BB_Upper"], modeling_base_data[f"{x_ticker['ticker']}_BB_Lower"].ffill()
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
          modeling_base_data[f"{ticker['ticker']}_BB_Upper"], modeling_base_data[f"{ticker['ticker']}_BB_Lower"] = modeling_base_data[f"{ticker['ticker']}_BB_Upper"], modeling_base_data[f"{ticker['ticker']}_BB_Lower"].ffill()
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

# Making Predictions
y_pred = model.predict(X_test.values)

# Model Evaluation and Refinement

# Regression Metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

model_results_title.markdown(
    f"<h2 style='text-align: center; color: #4CAF50;'>Model Performance Metrics</h2>",
    unsafe_allow_html=True
)
model_results.markdown(
    f"<div style='display: flex; justify-content: space-between; font-size: 18px;'>"
    f"<span><strong>Mean Squared Error (MSE):</strong> {mse:.4f}</span>"
    f"<span><strong>R-squared (R²):</strong> {r2:.4f}</span>"
    f"</div>",
    unsafe_allow_html=True
)
model_results_explanation_1.markdown("#### What does the value of R² mean?")
model_results_explanation_2.markdown("R² (R-squared) is a statistical measure that represents the proportion of the variance for a dependent variable that's explained by an independent variable or variables in a regression model. In simpler terms, it indicates how well the independent variables explain the variability of the dependent variable. A higher R² value (closer to 1) suggests a better fit of the model to the data, while a lower R² value (closer to 0) indicates that the model does not explain much of the variability.")


#st.write(f"Mean Squared Error (MSE): {mse:.4f}")
#st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
#st.write(f"R-squared (R2): {r2:.4f}")

# Residual Analysis
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_test, residuals)
plt.xlabel("Actual Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
st.pyplot(plt)
st.markdown("#### Why use Residual Analysis?")
st.markdown("Residual analysis helps in diagnosing the fit of a regression model. By examining the residuals (the differences between observed and predicted values), we can identify patterns that suggest whether the model is appropriate or if there are issues such as non-linearity, heteroscedasticity, or outliers. A good model should have residuals that are randomly scattered around zero, indicating that the model has captured all systematic information in the data.")


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

#st.write(f"RMSE (Ridge): {rmse_ridge:.4f}")
#st.write(f"RMSE (Lasso): {rmse_lasso:.4f}")

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
