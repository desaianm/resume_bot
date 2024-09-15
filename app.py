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
import PyPDF2




def get_jobs(query):
    
    params = {
    "engine": "google_jobs",
    "q": query,
    "hl": "en",
    "api_key": os.getenv("SERPER_API_KEY")
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    print("----------------------Test -----------------------------------")
    print(results)
    print("------------------------------------------------------------")
    jobs_results = results["jobs_results"]

    return jobs_results

def write_to_file(jobs_results):
# Serializing json
    json_object = json.dumps(jobs_results)
    print(json_object)
    # Writing to sample.json
    with open("sample.json") as outfile:
        json_file = json.load(outfile)

    json_file.append(json_object)
    # Load the JSON file into a pandas DataFrame
    df = pd.read_json(json_object)

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
llm = ChatOpenAI(model_name="gpt-4o-mini",temperature=0.3,api_key=os.getenv("OPENAI_API_KEY"))


# 4. Retreival Augmented Generation
def generate_response(message):
    template = """
    You are a talent source agent reviewing a list of job listings. You should respond to the user's request based on the information provided in the jobs list.
    If the information is available in the list, provide a natural, human-like response based on the data. Be sure to use language a talent source agent would use, like "I've got some great opportunities that might be a good fit for you..."
    If the information is not available, respond with "I don't have that information at hand, but I can look into it."
    Below is request 
    {message}

    Here is a list of jobs data:
    {best_responses}

    return job title, id, location and company name in json format 
    strictly only output in json format
    """


    prompt = PromptTemplate(
        input_variables=["message", "best_responses"],
        template=template
    )

    chain = LLMChain(llm=llm,prompt=prompt, verbose=True)

    best_responses = retrieve_info(message)
    response = chain.run(message=message, best_responses=best_responses)
    return response

#  Updating resume according to the job description
def generate_resume(skills):
    # reading Text file 
    file_path = 'resume.txt'
    file = open(file_path)
    resume = file.read()
    template2 = """
    You are an expert HR Manager responsible for hiring. Your task is to tailor the candidate's resume to the job description provided.
    
    Job Description: {skills}
    Current Resume: {resume}
    
    Please analyze the job description and the current resume. Then, modify the resume to highlight relevant skills, experiences, and qualifications that align with the job requirements. Your goal is to increase the candidate's chances of securing an interview.
    
    Provide an updated version of the resume that:
    1. Emphasizes relevant skills and experiences
    2. Uses industry-specific keywords from the job description
    3. Quantifies achievements where possible
    4. Maintains a professional and concise format
    5. Don't add any new information if not in the given resume and add informationin suggestions section
    6. Make Sure to be very specific in the suggestions section
    
    Return the updated resume and suggestions in a clear, well-structured format.
    """

    prompt = PromptTemplate(
        input_variables=["skills", "resume"],
        template=template2
    )

    chain = LLMChain(llm=llm,prompt=prompt, verbose=True)
    response = chain.run(skills=skills, resume=resume)
    resume =response
    return resume

def save_resume(resume):
    
    pdfreader = PyPDF2.PdfReader(resume)

    num_pages = len(pdfreader.pages)

    # Extract text from the last page (change x+1 to the desired page number)
    pageobj = pdfreader.pages[num_pages - 1]
    text = pageobj.extract_text
    with open('resume.txt', 'w') as f:
        f.write(text)

    if text: 
        return True    



def main():
    st.title("Resume Generator Bot")
    print("\n")
    query = st.text_area("Enter the Job Description")
      
    pdf = st.file_uploader("Upload your resume", type=["pdf"])
    if st.button("Generate Resume"):
        if query and pdf:
            with st.spinner("Generating updated resume..."):
                    resume = generate_resume(query)
                    st.subheader("Updated Resume")
                    st.text_area("", resume, height=400)
        elif not query:
            st.warning("Please enter a job description.")
        elif not pdf:
            st.warning("Please upload your resume.")

                
if __name__ == '__main__':
    main()
