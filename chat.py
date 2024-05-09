import random
import json
import subprocess
import sys
import time
import hashlib

import pymongo
import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
import os
import pdfkit
from analyzer import main as best_matches_main  # Importing the main function from best_matches.py

job_description_chat = ""
DATABASE_URL = "mongodb+srv://hrms:I9lOIMQX3SEtTZ9g@cluster0.zkuhzyx.mongodb.net/hrms"
client = pymongo.MongoClient(DATABASE_URL)
db = client.get_database("hrms")
responses_collection = db["responses"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
  intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE, map_location=torch.device('cpu'))
print(data.keys())
input_size = data["input_size"]
print(input_size)

hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "PI-DEV Chat Bot"

job_title_input = ""
job_skills_input = ""
job_experience_input = ""
additional_requirements_input = ""
preferred_sports_input = ""
job_list = []


def get_response(msg):
  global job_title_input, job_skills_input, job_experience_input, additional_requirements_input, preferred_sports_input, job_description_chat

  # Check if the user input corresponds to a job title
  sentence = tokenize(msg)
  X = bag_of_words(sentence, all_words)
  X = X.reshape(1, X.shape[0])
  X = torch.from_numpy(X).to(device)

  output = model(X)
  _, predicted = torch.max(output, dim=1)

  tag = tags[predicted.item()]

  probs = torch.softmax(output, dim=1)
  prob = probs[0][predicted.item()]
  response_generated = False

  if prob.item() > 0.75:
    for intent in intents['intents']:
      if tag == intent["tag"]:
        response = random.choice(intent['responses'])

        if tag == "job_inquiry":
          responses_collection.insert_one({"tag": tag, "input": msg, "response": response})
        elif tag == "job_title":
          job_title_input = msg
          job_description_chat += job_title_input
          job_list = [job_title_input]
          print("Job Title Input:", job_title_input)
          responses_collection.insert_one({"tag": tag, "input": msg, "response": response})
          scrape_resumes(job_title_input)
        elif tag == "job_skills":
          job_skills_input = msg
          job_description_chat += job_skills_input
          print("Job Skills Input:", job_skills_input)
          responses_collection.insert_one({"tag": tag, "input": msg, "response": response})
        elif tag == "job_experience":
          job_experience_input = msg
          job_description_chat += job_experience_input
          print("Job Experience Input:", job_experience_input)
          responses_collection.insert_one({"tag": tag, "input": msg, "response": response})
        elif tag == "additional_requirements":
          additional_requirements_input = msg
          job_description_chat += additional_requirements_input
          print("Additional Requirements Input:", additional_requirements_input)
          responses_collection.insert_one({"tag": tag, "input": msg, "response": response})
        elif tag == "preferred_sports":
          preferred_sports_input = msg
          job_description_chat += preferred_sports_input
          print("Preferred Sports Input:", preferred_sports_input)
          responses_collection.insert_one({"tag": tag, "input": msg, "response": response})
          if response == "Noted. Candidates with experience in will be considered.":
            update_training_data()
        elif tag == "best_matches":
          best_matches_result = best_matches_main()  # Call the function for generating best matches
          if best_matches_result is not None:
            response += "in list below : \n " + best_matches_result
          response_generated = True



        return response
  else:
    return "I'm not sure how to respond to that."



def update_training_data():
  global existing_df, job_description_chat

  # New data
  company_name = 'Talent Nest'
  position_title = job_title_input
  description_length = len(job_description_chat)
  required_skills = job_skills_input
  educational_requirements = additional_requirements_input

  experience_level = job_experience_input

  preferred_qualifications = preferred_sports_input

  model_response_content = {
    "Core Responsibilities": '',
    "Required Skills": required_skills,
    "Educational Requirements": educational_requirements,
    "Experience Level": experience_level,
    "Preferred Qualifications": preferred_qualifications,
    "Compensation and Benefits": ''
  }
  model_response = json.dumps(model_response_content)

  # Append new data to existing DataFrame
  if not existing_df.empty:
    existing_df = existing_df.iloc[1:]

    # Append new data to existing DataFrame
  new_data = {
    'company_name': company_name,
    'job_description': job_description_chat,
    'position_title': position_title,
    'description_length': description_length,
    'model_response': model_response
  }
  existing_df = pd.concat([existing_df, pd.DataFrame([new_data])], ignore_index=True)

  # Save DataFrame with updated data to training_data.csv
  existing_df.to_csv('training_data.csv', index=False)


def scrape_resumes(job):
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

  print("Scraping resume links...")
  job = job.lower()
  for i in range(1, 2):
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
    nonlocal id_counter
    id_counter += 1
    return id_counter

  resume_links["id"] = resume_links["link"].apply(id)

  column_names = ['ID', 'Resume_str', 'Resume_html', 'Category']

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

  df = pd.concat(dfs, ignore_index=True)

  df.to_csv("Resume.csv", index=False)

  labels = df['Category'].unique()

  parent_dir = os.path.join(os.getcwd(), "data")
  for label in labels:
    temp = df[df['Category'] == label].copy()
    temp.dropna(inplace=True)
    temp.reset_index(inplace=True, drop=True)
    path = os.path.join(parent_dir, label)
    os.makedirs(path, mode=0o666, exist_ok=True)
    for i in range(temp.shape[0]):
      resume_id = temp['ID'].iloc[i]
      print(resume_id)
      output_file = os.path.join(path, f"{resume_id}.pdf")
      with open(output_file, "wb") as f:
        f.write(pdfkit.from_string(temp['Resume_html'].iloc[i], False, configuration=config))


# Connect to MongoDB


# Specify the path to the wkhtmltopdf executable
config = pdfkit.configuration(wkhtmltopdf=r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe')
existing_df = pd.read_csv('training_data.csv')

if __name__ == "__main__":
  print("Let's chat! (type 'quit' to exit)")
  while True:
    sentence = input("You: ")
    if sentence == "quit":
      break

    resp = get_response(sentence)
    print(resp)
