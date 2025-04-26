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

import os

def list_files_recursive(path='.'):
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            list_files_recursive(full_path)
        else:
            st.write(full_path)

# Specify the directory path you want to start from
directory_path = '.'
list_files_recursive(directory_path)

