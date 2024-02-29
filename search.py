from serpapi import GoogleSearch
import os
import json
import pandas as pd
from langchain.document_loaders import CSVLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from apikey import key,key2

os.environ['OPENAI_API_KEY'] = key

query = " Data Analyst or Gen AI or Data Engineer or ML Engineer related Jobs in Canada  "


params = {
  "engine": "google_jobs",
  "q": query,
  "hl": "en",
  "api_key": key2
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
df = pd.read_json('resume_bot/sample.json')

#dropping unnecessary columns
df.drop(["thumbnail","job_id"], inplace=True, axis=1)   

# Convert the DataFrame to a CSV file
df.to_csv('resume_bot/sample.csv', index=True)


loader = CSVLoader(file_path="resume_bot/sample.csv")
documents  = loader.load()

#converting text into embeddings for similarity search 
embeddings = HuggingFaceEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
db = FAISS.from_documents(documents,embeddings)

def retrieve_info(query):
    similar_response = db.similarity_search(query,k=3)
    page_contents_array = [doc.page_content for doc in similar_response]
    return page_contents_array

# 3. Setup llm chain and prompts
llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125",temperature=0.3)

template = """
You should act like a person. A person who thinks like the response i gave you when he is asked something.follow below rules:

1 list is of Job Listings and give answer based on that

2 if don't have relevant info reply i don't know

Below is request 
{message}

Here is a list of jobs data:
{best_responses}
"""

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

# adding skills to existing resume
def generate_resume(skills):
    # reading Text file 
    file_path = 'resume_bot/resume.txt'
    file = open(file_path)
    resume = file.read()
    template2 = """
    You should act like a HR whose hiring. 
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
    message = ("Skills to enter in Resume to get an Interview for the Job postings")
    if message:
        skills = generate_response(message)
        print("\n")
        print(generate_resume(skills))

    
if __name__ == '__main__':
    main()
