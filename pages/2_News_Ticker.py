import streamlit
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
#from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

#open browser
driver = webdriver.Chrome()

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
