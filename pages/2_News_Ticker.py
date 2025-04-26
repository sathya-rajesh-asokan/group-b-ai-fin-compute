import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests # collect HTML
import time
import torch
from bs4 import BeautifulSoup as bs
from datetime import datetime, timedelta

from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
#from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import shutil

#open browser
options = Options()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-gpu")
options.add_argument("--disable-features=NetworkService")
options.add_argument("--window-size=1920x1080")
options.add_argument("--disable-features=VizDisplayCompositor")
options.add_argument('--ignore-certificate-errors')

options = Options()

HEADLESS_OPTIONS = [ "--headless=new","--disable-gpu", "--disable-dev-shm-usage","--window-size=1920,1080","--disable-search-engine-choice-screen"]
for option in HEADLESS_OPTIONS:
  options.add_argument(option)

service = Service(shutil.which('chromedriver'))

    # Initialize the WebDriver
driver = webdriver.Chrome(service=service, options=options)


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

end_date = datetime.now()
start_date = end_date - timedelta(days=365)

st.title("Latest News")

empty_block = st.empty()
empty_block.markdown("# Your news is being scraped...")
modeling_base_data = get_stock_data('300750.SZ', start_date, end_date, 'Adjusted Close Price')


base_url = "https://wallstreetcn.com/search"
params = "?q=宁德时代"
try:
    # First load without parameters to get past initial checks
    driver.get(base_url)
    time.sleep(2)
    # Then use JavaScript to modify the URL
    driver.execute_script(f"window.history.replaceState(null, null, '{base_url}{params}');")

    # Force a page reload with the new URL
    driver.execute_script("location.reload();")

    # Wait for content to load
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, ".content")))

    # Verify the current URL contains our key parameter
    current_url = driver.current_url
    last_height = driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    #The way this website structured, it keeps add new items non-stop if using keep scrolling to bottom.
    #So here we limit just 10 pages which has more than 1 year of news articles
    for i in range (0,10):
        #  scroll down the page to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)   #wait for loading the page


except Exception as e:
    print("Error:", str(e))

fullPage = bs(driver.page_source, "html.parser")

#insert def get_product_data(i)
News_section = fullPage.find("div",{"class":"article"})

def get_top_news(i):

    #get thumbnail data including name and prices
    item = News_section.findAll("div",{"class":"container"})[i]
    news = item.find("span").text # name
    date_str = item.find("time").text
    cleaned_str = date_str.strip().split()[0]
    try:
      date_obj = datetime.strptime(cleaned_str, "%Y-%m-%d").date()
    except Exception as e:
      date_obj = datetime.strptime('2025/04/25', "%Y/%m/%d").date()

    return ["CATL", date_obj, news]

# to check number of products included

numbers_items = len(News_section.findAll("div",{"class":"container"}))

collection=[]

for i in range (0,numbers_items):
    product_item = get_top_news(i) #get data for html
    collection.append(product_item)

df_collection = pd.DataFrame(data=collection, columns=['Ticker','Date','News'])

df_collection['date'] = pd.to_datetime(df_collection['Date'], format='%Y-%m-%d')  #
df_collection.set_index('date', inplace=True, drop=True)
df_collection.drop(columns=['Date'], inplace=True, errors='ignore')

def analyze_sentiment(news_df):
    """Perform sentiment analysis with offline fallback"""
    print("\nAnalyzing sentiment...")

    # Mock sentiment data for demo (in case model fails to load)
   # mock_sentiment = np.random.uniform(-0.5, 0.5, len(news_df))
    results = []

    try:

        tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

        for idx, row in tqdm(news_df.iterrows(), total=len(news_df)):
            inputs = tokenizer(row["News"], return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                outputs = model(**inputs)
            #probs = torch.softmax(outputs.logits, dim=1)[0].numpy()
            predictions = outputs.logits.softmax(dim=1)[0].numpy()
            #print(predictions) #positive/negative/neutral
            sentiment = predictions[0]-predictions[2]
            results.append(sentiment)

    except Exception as e:
        print(f"Using mock sentiment data (model load failed: {str(e)})")
       # results = mock_sentiment.tolist()

    news_df['sentiment'] = results
    return news_df

sentiment_df = analyze_sentiment(df_collection)

empty_block.dataframe(sentiment_df)

correlation_sentiment = sentiment_df.groupby(['date']).mean(numeric_only=True)

merged_data1 = pd.merge(sentiment_df, modeling_base_data, left_index=True, right_index=True, how='inner')

merged_data1.drop(columns=['Ticker'], inplace=True, errors='ignore')
merged_data1.drop(columns=['News'], inplace=True, errors='ignore')


# Calculate correlation coefficient
r_value = np.corrcoef(merged_data1['sentiment'], merged_data1['300750.SZ_adjClose'])[0, 1]

# Create the scatter plot (AI Generated Code)
plt.figure(figsize=(10, 6))
plt.scatter(merged_data1, merged_data1, alpha=0.5)
plt.title(f'Correlation between news and stock price\n')
plt.xlabel(f'Sentiment Data')
plt.ylabel(f'CATl Daily Close Price')

# Add regression line
m, b = np.polyfit(merged_data1['sentiment'], merged_data1['300750.SZ_adjClose'], 1)
plt.plot(merged_data1, m*merged_data1 + b, color='red')

# Add grid and show plot
plt.grid(True)
plt.tight_layout()

st.pyplot(plt)

# Print the correlation coefficient
st.write(f"<p style='font-size:20px'><span style='font-weight:bold'>Correlation coefficient (r-value) between News and Stock Price:</span> <span style='color:blue;'>{r_value:.4f}</span></p>", unsafe_allow_html=True)

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
