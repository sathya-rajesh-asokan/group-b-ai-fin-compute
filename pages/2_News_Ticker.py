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

import pathlib
st.write(pathlib.Path('.'))
#open browser

options = Options()

HEADLESS_OPTIONS = [ "--headless=new","--disable-gpu", "--disable-dev-shm-usage","--window-size=1920,1080","--disable-search-engine-choice-screen"]
for option in HEADLESS_OPTIONS:
  options.add_argument(option)

service = Service(r"/Users/asaresh/Documents/Code/GitHub/asaresh/ai-and-financial-computing/stock-prediction-app/chromedriver_local")

    # Initialize the WebDriver
driver = webdriver.Chrome(service=service, options=options)


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
st.title("News Ticker")
st.subheader("Latest News")

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
st.dataframe(sentiment_df)
