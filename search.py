from serpapi import GoogleSearch
import os
import json
import pandas as pd
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOllama
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
import os

query = st.text_input("Enter the Job Title to search like Junior data engineer Jobs Toronto")

params = {
  "engine": "google_jobs",
  "q": query,
  "hl": "en",
  "api_key": os.getenv("SERPER_API_KEY")
}

search = GoogleSearch(params)
results = search.get_dict()
jobs_results = results["jobs_results"]


# Serializing json
json_object = json.dumps(jobs_results)
  

# Writing to sample.json
with open("sample.json", "w") as outfile:
    outfile.write(json_object)

# Load the JSON file into a pandas DataFrame
df = pd.read_json('sample.json')

#dropping unnecessary columns
df.drop(["thumbnail"], inplace=True, axis=1)   

# Convert the DataFrame to a CSV file
df.to_csv('sample.csv', index=True)



loader = CSVLoader(file_path="sample.csv")
documents  = loader.load()

#converting text into embeddings for similarity search 
embeddings = HuggingFaceEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
db = FAISS.from_documents(documents,embeddings)

def retrieve_info(query):
    similar_response = db.similarity_search(query,k=5)
    page_contents_array = [doc.page_content for doc in similar_response]
    return page_contents_array

# 3. Setup llm chain and prompts
llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125",temperature=0.3)

template = """
You are a talent source agent reviewing a list of job listings. You should respond to the user's request based on the information provided in the jobs list.
If the information is available in the list, provide a natural, human-like response based on the data. Be sure to use language a talent source agent would use, like "I've got some great opportunities that might be a good fit for you..."
If the information is not available, respond with "I don't have that information at hand, but I can look into it."
Below is request 
{message}

Here is a list of jobs data:
{best_responses}

return job title, id, location, and company name in json format 
"""
st.title("Job Search Chatbot")


prompt = PromptTemplate(
    input_variables=["message", "best_responses"],
    template=template
)

chain = LLMChain(llm=llm,prompt=prompt, verbose=True)


# 4. Retreival Augmented Generation
def generate_response(message):
    best_responses = retrieve_info(message)
    response = chain.run(message=message, best_responses=best_responses)
    return response

#  Updating resume
def generate_resume(skills):
    # reading Text file 
    file_path = 'resume.txt'
    file = open(file_path)
    resume = file.read()
    template2 = """
    You are an expert a like a HR Manager whose hiring. 
    Below are the  {skills} , make changes in the below {resume} to get an Interview for the Job.
    Give me the updated resume 
    """

    prompt = PromptTemplate(
        input_variables=["skills", "resume"],
        template=template2
    )

    chain = LLMChain(llm=llm,prompt=prompt, verbose=True)
    response = chain.run(skills=skills, resume=resume)
    resume =response
    return resume

def main():
    
    print("\n")
    message = ("Skills to enter in Resume to get an Interview for the Job postings with Job Id ")
    if message:
        jobs = generate_response(message)
        print("\n") 
        print(jobs)
        job_id = input("Enter the Job Id to update resume:")
        df_jobs = pd.DataFrame(json.loads(jobs))
        print(df_jobs)
        row = df_jobs.loc[df_jobs.apply(lambda x: x.to_dict(), axis=1).apply(lambda x: json.dumps(x, sort_keys=True)) == job_id]
        print(row)
    
if __name__ == '__main__':
    main()
