

# from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

import jenkins
import os
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings



from langchain_community.document_loaders import PyPDFLoader
file_path="Load Prediction Whitepaper.pdf"

loader = PyPDFLoader(file_path)

whitepaper_docs = []
for page in loader.load():
    whitepaper_docs.append(page)


# Jenkins server credentials
JENKINS_URL = 'http://localhost:8080'
USERNAME = 'chandrakant'
PASSWORD = '11310e1eb836f096680b723f26ef75a055'

# Connect to Jenkins
server = jenkins.Jenkins(JENKINS_URL, username=USERNAME, password=PASSWORD)

# Get Jenkins info
user_info = server.get_whoami()

print(f"Connected to Jenkins as {user_info['fullName']}")


def get_all_jobs():
    job_name_list=[]
    job_desc_list=[]
    for i in server.get_all_jobs():
        job_name_list.append(i['name'])
        job_desc_list.append(server.get_job_info(i['name'])['description'])
    return job_name_list,job_desc_list


def create_document(job_name_list,job_desc_list):
    docs = [Document(page_content=text, metadata={"source": job_name_list[i]})  
            for i, text in enumerate(job_desc_list)]
    return docs


def create_jenking_vector_db(docs):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    jenking_vectorstore = "jenking_faiss_store/index.pkl"

    if os.path.exists(jenking_vectorstore):
        print(f"The file '{jenking_vectorstore}' exists.")
    else:
        jenking_vectorstore  =FAISS.from_documents(docs,embeddings)
        jenking_vectorstore.save_local("jenking_faiss_store")
        print("✅ Jenking FAISS vector store created.")

    jenking_vectorstore = FAISS.load_local("jenking_faiss_store", embeddings, allow_dangerous_deserialization=True)
    return jenking_vectorstore


def similarity_search(data,jenking_vectorstore):
    projected_job=[]
    res=jenking_vectorstore.similarity_search(data,k=3)
    # return [print(i.metadata['source'] +" \n "+ i.page_content+"\n") for i in res]
    [projected_job.append(i.metadata['source'] ) for i in res]
    return projected_job


def create_whitepaper_vector_db():    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    whitepaper_vectorstore = "whitepaper_vectorstore/index.pkl"

    if os.path.exists(whitepaper_vectorstore):
        print(f"The file '{whitepaper_vectorstore}' exists.")
    else:
        whitepaper_vectorstore  =FAISS.from_documents(whitepaper_docs,embeddings)
        whitepaper_vectorstore.save_local("whitepaper_vectorstore")
        print("✅ whitepaper FAISS vector store created.")

    whitepaper_vectorstore = FAISS.load_local("whitepaper_vectorstore", embeddings, allow_dangerous_deserialization=True)
    res=whitepaper_vectorstore.similarity_search(
    "Find the intoduction and summary part of the document",
    k=2,
    filter={"page_label": "1"},
    )
    return res[0].page_content


# output is list
# projected_job is list
def final():
    job_name_list,job_desc_list=get_all_jobs()
    docs=create_document(job_name_list,job_desc_list)
    jenking_vectorstore=create_jenking_vector_db(docs)
    data=create_whitepaper_vector_db()
    # print(data)
    projected_job=similarity_search(data,jenking_vectorstore)
    return projected_job

# print(final())

# after list populated , based on user selection one item from list will pass
# job_name is string
def job_build(job_name):
    server.build_job(job_name)
    return f'Job is build for this {job_name} '

# job_name=final()[0]
# print(job_build(job_name))


