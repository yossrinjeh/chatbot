import pandas as pd
from PyPDF2 import PdfReader
from nltk import pos_tag, sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string
import re
import torch
from tabulate import tabulate
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os


# Function to extract text, name, and email from a PDF file
def extract_text_and_info_from_pdf(file_path):
  try:
    reader = PdfReader(file_path)
    text = "".join(page.extract_text() for page in reader.pages)
    # Extract name using regex
    name_match = re.search(r'([A-Z][a-z]+(?: [A-Z][a-z]+)*)', text)
    name = name_match.group(1) if name_match else ""
    # Extract email using regex
    email_match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    email = email_match.group() if email_match else ""
    return text, name, email
  except Exception as e:
    print(f"Error extracting text and info from PDF: {e}")
    return "", "", ""


# Function to preprocess text
def preprocess_text(text):
  text = text.lower()
  text = re.sub('[^a-zA-Z]', ' ', text)
  sentences = sent_tokenize(text)
  features = {'feature': ""}
  stop_words = set(stopwords.words("english"))
  for sent in sentences:
    if any(criteria in sent for criteria in ['skills', 'education']):
      words = word_tokenize(sent)
      words = [word for word in words if word not in stop_words]
      tagged_words = pos_tag(words)
      filtered_words = [word for word, tag in tagged_words if tag not in ['DT', 'IN', 'TO', 'PRP', 'WP']]
      features['feature'] += " ".join(filtered_words)
  return features


# Function to load resume data from a CSV file
def load_resume_data(file_path):
  try:
    resume_data = pd.read_csv(file_path)
    resume_data = resume_data.drop(["Resume_html"], axis=1)
    return resume_data
  except Exception as e:
    print(f"Error loading resume data: {e}")
    return None


# Function to process each row of resume data
def process_resume_data(row):
  try:
    if pd.notna(row['ID']):
      id = int(row['ID'])
      category = row['Category']
      text, name, email = extract_text_and_info_from_pdf(f"data/{category}/{id}.pdf")
      features = preprocess_text(text)
      return features['feature'], name, email
  except Exception as e:
    print(f"Error processing resume data: {e}")
    return "", "", ""


# Function to get embeddings of text using BERT model
def get_embeddings(text, model, tokenizer, device):
  try:
    inputs = tokenizer(str(text), return_tensors="pt", truncation=True, padding=True).to(device)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().to("cpu").numpy()
    return embeddings.squeeze()
  except Exception as e:
    print(f"Error getting embeddings: {e}")
    return None


# Function to print top matching resumes for each job description
def print_top_matching_resumes(result_group):
  try:
    for job_id, group_data in result_group:
      print("\nJob ID:", job_id)
      print("Cosine Similarity | Resume ID | Name                      | Email                          | Domain Resume                  | Domain Description")
      for _, row in group_data.iterrows():
        print(f"{row['similarity']:17.4f} | {row['resumeId']:9} | {row['name']:25} | {row['email']:30} | {row['domainResume']:30} | {row['domainDesc']}")
  except Exception as e:
    print(f"Error printing top matching resumes: {e}")


# Main function
def main():
  result = None
  response = ""
  try:
    # Load data
    resume_data = load_resume_data("Resume.csv")
    job_description = pd.read_csv("training_data.csv")[["job_description", "position_title"]][:15]

    # Process resume data
    resume_data['Feature'], resume_data['Name'], resume_data['Email'] = zip(*resume_data.apply(process_resume_data, axis=1))
    resume_data = resume_data.dropna(subset=['Feature'])

    # Load BERT model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    # Get embeddings for job descriptions and resumes
    job_desc_embeddings = np.array(
      [get_embeddings(desc, model, tokenizer, device) for desc in job_description['job_description']])
    resume_embeddings = np.array(
      [get_embeddings(text, model, tokenizer, device) for text in resume_data['Feature']])

    # Calculate cosine similarity
    similarities = cosine_similarity(job_desc_embeddings, resume_embeddings)

    # Prepare result DataFrame
    result_df = pd.DataFrame(columns=['jobId', 'resumeId', 'similarity', 'domainResume', 'domainDesc', 'name', 'email'])
    for i, similarity_row in enumerate(similarities):
      top_k_indices = np.argsort(similarity_row)[::-1][:5]
      for j in top_k_indices:
        result_df.loc[len(result_df)] = [i, resume_data.iloc[j]['ID'], similarity_row[j],
                                         resume_data.iloc[j]['Category'],
                                         job_description.iloc[i]['position_title'],
                                         resume_data.iloc[j]['Name'],
                                         resume_data.iloc[j]['Email']]

    result_df = result_df.sort_values(by='similarity', ascending=False)
    result_group = result_df.groupby("jobId")

    for job_id, group_data in result_group:
      for _, row in group_data.iterrows():
        similarity = f"{row['similarity']:17.4f}"
        resume_id = f"{row['resumeId']:9}"
        name = f"{row['name'][:25]:25}"
        email = f"{row['email'][:30]:30}"
        domain_resume = f"{row['domainResume'][:30]:30}"
        domain_desc = f"{row['domainDesc']}"

        response += f"│ {similarity}  {resume_id}  {name}  {email}  {domain_resume}  {domain_desc} │\n"
        response +=f"││\n"




  except Exception as e:
    print(f"An error occurred: {e}")
    response = f"An error occurred: {e}"

  return response




if __name__ == "__main__":
  main()

