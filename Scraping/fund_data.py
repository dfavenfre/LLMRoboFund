from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from Helpers.helper_functions import (
    create_table,
    add_data,
    scroll_down,
    scrape_return_based_data,
    scrape_fund_management_fee_data,
    scrape_asset_size_data,
    process_columns,
)
from selenium import webdriver
import pandas as pd
import time
import sqlite3


def scrape_tefas_fund_data():
    url = "https://www.tefas.gov.tr/FonKarsilastirma.aspx"
    service = Service(ChromeDriverManager(driver_version="119.0.6045.160").install())
    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(service=service, options=options)
    driver.get(url)
    driver.maximize_window()

    fund_return_data = scrape_return_based_data(driver)
    if fund_return_data is not None:
        scroll_down(driver)
        driver.find_element(
            By.XPATH,
            '//body//div[@id="MainContent_tabs"]//li[@onclick="CreateManagementFees();"]',
        ).click()
        time.sleep(2)
        management_fee_data = scrape_fund_management_fee_data(driver)

        if management_fee_data is not None:
            scroll_down(driver)
            driver.find_element(
                By.XPATH,
                '//body//div[@id="MainContent_tabs"]//li[@onclick="CreateFundSizes();"]',
            ).click()
            time.sleep(2)
            asset_size_data = scrape_asset_size_data(driver)
            if asset_size_data is not None:
                driver.quit()

    place_holder_df = pd.concat(
        [fund_return_data, management_fee_data, asset_size_data], axis=1
    )

    columns = [
        "monthly_return",
        "monthly_3_return",
        "monthly_6_return",
        "since_jan",
        "annual_1_return",
        "annual_3_return",
        "annual_5_return",
        "applied_management_fee",
        "bylaw_management_fee",
        "applied_management_fee",
        "annual_realized_return_rate",
        "max_total_expense_ratio",
        "init_fund_size",
        "current_fund_size",
        "portfolio_size_change",
        "init_out_shares",
        "current_out_shares",
        "change_in_nshares",
        "realized_return_rate",
    ]

    new_df = process_columns(df=place_holder_df, column_list=columns)
    combined_df = pd.concat([place_holder_df.iloc[:, :3], new_df], axis=1)

    return combined_df


if __name__ == "__main__":
    conn = sqlite3.connect("tefas.db")
    curr = conn.cursor()
    create_table(connection=conn, cursor=curr)
    processed_tefas_data = scrape_tefas_fund_data()
    add_data(conn, data=processed_tefas_data)
