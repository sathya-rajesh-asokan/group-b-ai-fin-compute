import streamlit as st

import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Correlation Calculation", page_icon="📈")

st.markdown("# Correlation Calculation")
st.sidebar.markdown('## Choose Dependant Stock')
ticker1 = st.sidebar.selectbox("", ["300750.SZ"])
st.sidebar.write("This is the stock you want to predict")
st.sidebar.divider()
st.sidebar.markdown('## Choose Independant Stock')
ticker2 = st.sidebar.selectbox("", ["1211.HK"])
st.sidebar.write("This is the stock/index you want to use as a feature in your linear regression")
st.sidebar.divider()
st.sidebar.markdown('## Find Corrlation based on')
findCorrelationFor = st.sidebar.selectbox("", ["Adjusted Close Price", "Price Change"])
st.sidebar.divider()

# Simple function to retrieve data from FMP
def get_stock_data(_ticker,_start_date,_end_date):
    end_point='https://financialmodelingprep.com/stable/historical-price-eod/dividend-adjusted'
    api_key='4ZTUtST6urFlN83a1LDFy6U7plAHhegP'

    params = {}
    params['apikey'] = api_key
    params['symbol'] = _ticker
    params['from'] = _start_date
    params['to'] = _end_date

    fmp_response=requests.get(end_point, params=params)

    # Convert JSON to Dataframes
    df = pd.DataFrame(fmp_response.json())

    # I am setting the index to date, so as to be able to calcualte pct change and join
    df.set_index('date', inplace=True, drop=True)

    if 'Adjusted Close Price' == findCorrelationFor or 'Price Change' == findCorrelationFor:
        df = df[['adjClose']].rename(columns={'adjClose': 'measure'})

    return df

# Set up the time period (last 1 year by default)
end_date = datetime.now()
start_date = end_date - timedelta(days=30)


stock_data1 = get_stock_data(ticker1, start_date, end_date)
stock_data2 = get_stock_data(ticker2, start_date, end_date)

#Have to merge these as both stock exchange have dates when the other was not working. For now just making it work. For finals need to extrapolate data
# Keep in mind that
merged_data1 = pd.merge(stock_data1, stock_data2, left_index=True, right_index=True, how='left').drop(columns=['measure_y'])
merged_data2 = pd.merge(stock_data1, stock_data2, left_index=True, right_index=True, how='left').drop(columns=['measure_x'])
# Extract the adjusted close prices and assign column names
merged_data1 = merged_data1.rename(columns={'measure_x': 'measure'})
merged_data2 = merged_data2.rename(columns={'measure_y': 'measure'})

# Fill forward to handle missing values
merged_data1 = merged_data1.ffill()
merged_data2 = merged_data2.ffill()

# Calculate daily returns (percent change)
# Note: This is a simple way to calculate returns, but you might want to consider using log returns for more accuracy in financial applications.
# Reshape the data to 2D arrays before scaling
if 'Adjusted Close Price' == findCorrelationFor:
    scaler = StandardScaler()
    returns1 = scaler.fit_transform(merged_data1.values.reshape(-1, 1)).flatten()
    returns1 = pd.DataFrame(returns1, columns=merged_data1.columns, index=merged_data1.index)
    returns2 = scaler.fit_transform(merged_data2.values.reshape(-1, 1)).flatten()
    returns2 = pd.DataFrame(returns2, columns=merged_data2.columns, index=merged_data2.index)
elif 'Price Change' == findCorrelationFor:
    returns1 = np.log(merged_data1 / merged_data1.shift(1)).dropna()
    returns2 = np.log(merged_data2 / merged_data2.shift(1)).dropna()


# Calculate correlation coefficient
r_value = np.corrcoef(returns1['measure'], returns2['measure'])[0, 1]

# Create the scatter plot (AI Generated Code)
plt.figure(figsize=(10, 6))
plt.scatter(returns1, returns2, alpha=0.5)
plt.title(f'{ticker1} vs {ticker2} Daily Returns Correlation\n'
          f'Time Period: {start_date.date()} to {end_date.date()}\n'
          f'Correlation Coefficient (r): {r_value:.3f}')
plt.xlabel(f'{ticker1} Daily Returns')
plt.ylabel(f'{ticker2} Daily Returns')

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
# Add regression line
m, b = np.polyfit(returns1['measure'], returns2['measure'], 1)
plt.plot(returns1, m*returns1 + b, color='red')

# Add grid and show plot
plt.grid(True)
plt.tight_layout()

st.pyplot(plt)


# Print the correlation coefficient
st.write(f"<p style='font-size:20px'><span style='font-weight:bold'>Correlation coefficient (r-value) between {ticker1} and {ticker2}:</span> <span style='color:blue;'>{r_value:.4f}</span></p>", unsafe_allow_html=True)

if r_value > 0.7:
    st.write(f"<h4>Strong positive correlation</h4> 👍🏽", unsafe_allow_html=True)
elif r_value > 0.3:
    st.write(f"<h4>Moderate positive correlation</h4> 👍🏽", unsafe_allow_html=True)
elif r_value > -0.3:
    st.write(f"<h4>Weak or no correlation</h4> ", unsafe_allow_html=True)
elif r_value > -0.7:
    st.write(f"<h4>Moderate negative correlation</h4>", unsafe_allow_html=True)
else:
    st.write(f"<h4>Strong negative correlation</h4>", unsafe_allow_html=True)
