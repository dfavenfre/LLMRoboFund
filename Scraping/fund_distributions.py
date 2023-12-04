from typing import List
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
import sqlite3
from selenium import webdriver
import pandas as pd
from datetime import datetime, timedelta
from Helpers.helper_functions import (
    create_table,
    add_data,
    get_data,
    scrape_fund_details,
    create_tablev2,
)
import numpy as np
import os

if __name__ == "__main__":
    combined_df = scrape_fund_details()
    combined_df = pd.read_csv(r"file_path")
    conn = sqlite3.connect("funddetails.db")
    curs = conn.cursor()
    create_tablev2(connection=conn, cursor=curs, tablename="detailtable")
    add_data(table_name="detailtable", connection=conn, data=combined_df)
    db_data = get_data(cursor=curs, table_name="detailtable")
    print(db_data)
