import hashlib
import sys
import time

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver

from chat import job_list
print(job_list)
print(sys.path)

# Initialize global counter
id_counter = 0

def initialize_driver():
    options = webdriver.FirefoxOptions()
    options.add_argument("--headless")
    options.binary_location = r"C:\geckodriver\geckodriver.exe"

    driver = webdriver.Firefox(options=options)
    return driver


category = []
link = []
resume_links = pd.DataFrame()
job='software engineer'
# Iterate over job categories and pages to collect resume links
print("Scraping resume links...")
job = job.lower()
for i in range(1, 3):   # INCREASE THE RANGE TO GET MORE RESUME DATA
  PAGE = str(i)
  URL = "https://www.livecareer.com/resume-search/search?jt=" + job + "&bg=85&eg=100&comp=&mod=&pg=" + PAGE
  driver = initialize_driver()
  print(URL)
  driver.get(URL)
  time.sleep(0.5)
  page_source = driver.page_source
  soup = BeautifulSoup(page_source, 'html.parser')
  a_tags = soup.find_all('a', class_='sc-1dzblrg-0 caJIKu sc-1os65za-2 jhoVRR')
  for a in a_tags:
      category.append(job)
      link.append("https://www.livecareer.com" + a['href'])
      print(link)
      driver.quit()

print("Resume links scraped successfully.")
print("Total number of resume links found:", len(link))

resume_links["Category"] = category
resume_links["link"] = link

def id(x):
    global id_counter
    id_counter += 1
    return id_counter

resume_links["id"] = resume_links["link"].apply(id)

# Define the column names for the final DataFrame
column_names = ['ID', 'Resume_str', 'Resume_html', 'Category']

# Initialize an empty DataFrame with the specified column names
df = pd.DataFrame(columns=column_names)

dfs = []

for i in range(resume_links.shape[0]):
    url = resume_links['link'][i]
    driver = initialize_driver()
    driver.get(url)
    time.sleep(0.5)
    x = driver.page_source
    soup = BeautifulSoup(x, 'html.parser')
    x = x.replace(">", "> ")
    div = soup.find("div", class_="document")
    resume_str = div.get_text(separator=' ') if div else None

    # Appending data to the DataFrame
    temp_df = pd.DataFrame({'ID': [resume_links['id'][i]],
                            'Resume_str': [resume_str],
                            'Resume_html': [str(div)],
                            'Category': [resume_links['Category'][i]]})
    dfs.append(temp_df)
    driver.quit()

# Concatenate all temporary DataFrames into a single DataFrame
df = pd.concat(dfs, ignore_index=True)

# Save the DataFrame to CSV
df.to_csv("Resume.csv", index=False)
