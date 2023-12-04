from selenium.webdriver.common.by import By
from selenium import webdriver
import time
import os


download_folder = os.environ.get("folder_path")
chrome_options = webdriver.ChromeOptions()
chrome_options.add_experimental_option(
    "prefs",
    {
        "download.default_directory": download_folder,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True,
    },
)

driver = webdriver.Chrome(options=chrome_options)
driver.get("https://www.kap.org.tr/tr/fonlarTumKalemler/kpy81_acc1_fon_sem_unvan")

source_element = driver.find_element(By.XPATH, '//div[@id="printAreaDiv"]')
iterable_rows = source_element.find_elements(
    By.XPATH,
    '//div[@class="comp-cell-row-div vtable infoColumnBorder evenRowBackground even infoColumnTopBorder infoColumnBottomBorder"]',
)

condition_template = (
    '//a[@class="w-inline-block subpage-button vtable alignTextToLeft"]'
)

for rows in iterable_rows:
    rows.click()
    time.sleep(1)

    kap_source_element = driver.find_element(
        By.XPATH, '//div[@class="w-col w-col-4 w-clearfix sub-col"]'
    )

    # Condition_ytf True: if -> 'yatırımcı bilgi formu' exists
    condition_ytf = kap_source_element.find_elements(
        By.XPATH, f'{condition_template}//*[contains(text(), "Yatırımcı Bilgi Formu")]'
    )
    if condition_ytf is not None:
        for ytf_link in condition_ytf:
            ytf_link.click()
            time.sleep(1)
            driver.back()
