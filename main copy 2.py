import os
import nbformat
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings

import streamlit as st
import fitz
import tempfile
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_api_key)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

chroma_client = chromadb.PersistentClient(path="./chroma_openai1")

def get_file_hash(uploaded_file):
    uploaded_file.seek(0)
    hash_val = hashlib.sha256(uploaded_file.read()).hexdigest()
    uploaded_file.seek(0)
    return hash_val

def collection_exists(collection_name):
    try:
        chroma_client.get_collection(collection_name)
        return True
    except Exception:
        return False

# Start: Get white paper from one drive  
import msal
import requests
import time
import fitz
import os 

load_dotenv() 
Permission_ID ="6a94cb3a-9869-4b54-ae0b-f4f523df2614"  # Mote into env variable
client_id = Permission_ID
authority = "https://login.microsoftonline.com/consumers"
scopes = ["Files.Read"]
source_folder = "Documents/GitHub/code-validator/"    
file_name = "Load Prediction Whitepaper.pdf"       # input 
version_id = int(7)                                # input 
file_path = source_folder+file_name

def access_token_key(client_id, authority):
    scopes = ["Files.Read"]
    app = msal.PublicClientApplication(client_id=client_id, authority=authority)
    result = None

    accounts = app.get_accounts()
    if accounts:
        result = app.acquire_token_silent(scopes, account=accounts[0])
    if not result:
        result = app.acquire_token_interactive(scopes=scopes)
    if not result or "access_token" not in result:
        print("MSAL Error:", result)
    access_token = result["access_token"]

    return access_token


def get_raw_data(client_id, authority ,file_path ):
    access_token= access_token_key(client_id=client_id, authority=authority)
    url = f"https://graph.microsoft.com/v1.0/me/drive/root:/{file_path}:/content"
    headers = {"Authorization": f"Bearer {access_token}"}
    time.sleep(2)
    response = requests.get(url, headers=headers)
    print(f"Response code: {response.status_code}")
    if response.status_code == 200:
        file_bytes = response.content
        print("File read into memory!")
        return file_bytes
    else:
        print("Failed:", response.status_code, response.text)
        return None


def get_onedrive_whitepaper(file_bytes):
    
    # file_bytes is from above
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page_num, page in enumerate(doc):
        text += f"\n\n--- Page {page_num + 1} ---\n{page.get_text()}"

    print("First 1000 chars of PDF text:", text)
    
    return text
   
def prev_version( client_id, authority, file_path, version_id):
    access_token= access_token_key(client_id=client_id, authority=authority)

    versions_url = f"https://graph.microsoft.com/v1.0/me/drive/root:/{file_path}:/versions"
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(versions_url, headers=headers)

    if response.status_code == 200:
        versions = response.json()["value"]
        if len(versions) >= int(version_id):
            # 3. Get the 2nd version (index 1)
            version_id = versions[1]['id']
            print(f"2nd Version ID: {version_id}, Last Modified: {versions[1]['lastModifiedDateTime']}")
            
            # 4. Fetch 2nd version's PDF bytes
            download_url = f"https://graph.microsoft.com/v1.0/me/drive/root:/{file_path}:/versions/{version_id}/content"
            version_response = requests.get(download_url, headers=headers)
            if version_response.status_code == 200:
                pdf_bytes = version_response.content  # This is your PDF in memory
                
                # 5. Extract text from the PDF (in memory, no save)
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                all_text = ""
                for page_num, page in enumerate(doc):
                    all_text += f"\n--- Page {page_num+1} ---\n{page.get_text()}"
                
                print("Extracted PDF text (first 1000 chars):")
                print(all_text[:1000])
                return all_text
                # You can use `all_text` as needed (search, LLM input, etc)
            else:
                print("Failed to download 2nd version:", version_response.status_code, version_response.text)
        else:
            print("Less than 2 versions available!")
    else:
        print("Failed to fetch versions:", response.status_code, response.text)

# End: Get white paper from one drive  


# Start: Get notebook file from Github

import json 
import pandas as pd 
import requests
import requests
import base64

push_event= pd.read_json('push_events.json', )
latest_push_id = push_event[0].tolist()[0]
latest_push_id
owner = "arunkenwal02"
repo = "code-validator"
push_id = latest_push_id
file_path = "loan-approval-prediction_v2.ipynb"


def get_sha_pair_from_push_id(owner, repo, push_id):
    """
    Returns (before_sha, head_sha) for the given push_id.
    If not found, returns (None, None).
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/events"
    resp = requests.get(url)
    events = resp.json()
    for event in events:
        if event["type"] == "PushEvent" and event["id"] == str(push_id):
            before_sha = event["payload"]["before"]
            head_sha = event["payload"]["head"]
            print(f"Push ID: {push_id}\nbefore: {before_sha}\nhead: {head_sha}")
            return before_sha, head_sha
    print(f"Push ID {push_id} not found in recent events.")
    return None, None

def fetch_latest_file_for_sha(owner, repo, file_path, sha_pairs):
    """
    For each (sha_old, sha_new) in sha_pairs, check if file_path was updated.
    If yes, download file from sha_new. Else, download most recently updated version.
    """
    for i, (sha_old, sha_new) in enumerate(sha_pairs):
        print(f"\nProcessing pair {i+1}: {sha_old} → {sha_new}")

        # 1. Compare the two SHAs
        compare_url = f"https://api.github.com/repos/{owner}/{repo}/compare/{sha_old}...{sha_new}"
        compare_resp = requests.get(compare_url)
        compare_data = compare_resp.json()

        file_changed = False
        for f in compare_data.get("files", []):
            if f["filename"] == file_path:
                file_changed = True
                print(f"File {file_path} was changed in this push.")
                break

        if file_changed:
            # Download updated file from sha_new
            content_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
            params = {"ref": sha_new}
            file_resp = requests.get(content_url, params=params)
            file_data = file_resp.json()
            
        # Check for 'content' key (base64-encoded)
            if "content" in file_data:
                nb_json = base64.b64decode(file_data["content"]).decode("utf-8")
                notebook_dict = json.loads(nb_json)
                return notebook_dict
            else:
                raise Exception("Notebook not found or could not fetch content. Details: " + str(file_data))

        else:
            print(f"File {file_path} was NOT changed between {sha_old} and {sha_new}.")
            # Get most recent commit where this file was updated
            commits_url = f"https://api.github.com/repos/{owner}/{repo}/commits"
            params = {"path": file_path, "per_page": 1}
            commits_resp = requests.get(commits_url, params=params)
            last_update_sha = commits_resp.json()[0]["sha"]
            print("Most recent commit where file was changed:", last_update_sha)
            # Download file at that SHA
            content_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
            params = {"ref": last_update_sha}
            file_resp = requests.get(content_url, params=params)
            file_data = file_resp.json()
            
            # Check for 'content' key (base64-encoded)
            if "content" in file_data:
                nb_json = base64.b64decode(file_data["content"]).decode("utf-8")
                notebook_dict = json.loads(nb_json)
                return notebook_dict
            else:
                raise Exception("Notebook not found or could not fetch content. Details: " + str(file_data))




# End Get notebook from github


def extract_from_pdf(client_id, authority, file_path, version_id):
    extracted_text = prev_version(client_id=client_id, authority=authority,  file_path= file_path, version_id = version_id)


    # doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    # extracted_text = ""
    # for page_num, page in enumerate(doc):
    #     text = page.get_text()
    #     extracted_text += f"\n\n--- Page {page_num + 1} ---\n{text}"
    return extracted_text


def create_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,     # faster, smaller chunk
        chunk_overlap=100   # reduced overlap
    )
    return text_splitter.split_text(text)


 
@st.cache_resource(show_spinner="Generating/retrieving embeddings...")
def get_or_create_embeddings(uploaded_file, text, _embedding_model, collection_name):
    chunks = create_chunks(text)

    if collection_exists(collection_name):
        collection = chroma_client.get_collection(collection_name)
    else:
        embeddings = _embedding_model.embed_documents(chunks)
        collection = store_in_chromaDB(chunks, embeddings, collection_name)

    return collection, chunks


def read_notebook_with_outputs(owner,repo ,push_id ,file_path):

    '''
    Get data from Github
    1. Connect to push id and get all shas
    2. check loan approival prediction models caanges in any of these sha ids if cahnges pick updated one in not get previous one 
    
    '''
    sha_pairs = get_sha_pair_from_push_id(owner = owner, repo = repo, push_id = push_id)
    sha_pairs1 = [sha_pairs]
    notebook_contents = fetch_latest_file_for_sha(owner = owner, repo = repo, file_path = file_path, sha_pairs1 = sha_pairs1)

    all_cells_text = ""

    for i, cell in enumerate(notebook_contents['cells']):
        if cell['cell_type'] == 'code' and cell.get('outputs'):
            # Add cell number and code
            all_cells_text += f"\nCell #{i+1}\n"
            all_cells_text += "Code:\n"
            all_cells_text += "".join(cell['source']).strip() + "\n"
            all_cells_text += "Output(s):\n"
            # Add outputs
            for output in cell['outputs']:
                output_text = ""
                if output.get('output_type') == 'stream':
                    text = output.get('text', '')
                    if isinstance(text, list):
                        text = "".join(text)
                    output_text += text.strip()
                elif output.get('output_type') in ['execute_result', 'display_data']:
                    data = output.get('data', {})
                    text = data.get('text/plain', '')
                    if isinstance(text, list):
                        text = "".join(text)
                    output_text += text.strip()
                # Skipping errors
                if output_text:
                    all_cells_text += output_text + "\n"
            all_cells_text += "-" * 30 + "\n"

    # Optional: remove leading/trailing whitespace
    all_cells_text = all_cells_text.strip()

    return all_cells_text

    '''
    nb = nbformat.read(file_path, as_version=4)
    cells_content = []
    for cell in nb.cells:
        if cell.cell_type == 'markdown':
            cells_content.append(f"## Markdown Cell:\n{cell.source}")
        elif cell.cell_type == 'code':
            code = f"## Code Cell:\n```python\n{cell.source}\n```"
            outputs = []
            for output in cell.get("outputs", []):
                if output.output_type == "stream":
                    outputs.append(f"Output (stream):\n{output.text}")
                elif output.output_type == "execute_result":
                    result = output.get("data", {}).get("text/plain", "")
                    outputs.append(f"Output (execute_result):\n{result}")
                elif output.output_type == "error":
                    outputs.append("Error:\n" + "\n".join(output.get("traceback", [])))
            full_output = "\n".join(outputs)
            if full_output:
                code += f"\n\n### Output:\n```\n{full_output}\n```"
            cells_content.append(code)
    return "\n\n".join(cells_content)
    '''

def queryFun_parallel(queries, embedding_model, collection):
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(queryFun, query, embedding_model, collection) for query in queries]
        return [future.result() for future in as_completed(futures)]


def queryFun(query, embedding_model, collection):
    query_embedding = embedding_model.embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=5)
    l_docs = [doc for doc in results["documents"][0]]
    return l_docs


def refine_extracted_elements_with_context(similar_elements, query_context):
    combined_elements = "\n\n".join(similar_elements)
    prompt = [
        SystemMessage(content="You are a product analyst."),
        HumanMessage(content=f"""
        The following are the top 5 similar elements retrieved from a vector database:

        {combined_elements}

        The original query context is:
        "{query_context}"

        - Identify and extract only the most relevant elements or functionalities.
        - Avoid verbose explanations; focus on clarity and precision.
        - Extract details from given context keep length short in summary format 
        - Do not recommend, only extract
        - Extract metrics score/values, model name and and hyperpapramter values if available in context 
        - Provide concise, bullet-pointed outputs or insights based on retrieved data.
        - Format the response using IPython Markdown style for readability

        """)
    ]
    return llm(prompt).content.strip()

def compare_functionalities(whitepaper_funcs, code_funcs):
    prompt = [
        SystemMessage(content="You are a software QA expert."),
        HumanMessage(content=f"""
        Whitepaper Functionalities:
        {whitepaper_funcs}

        Code Functionalities:
        {code_funcs}
        Compare each functionality described in the white paper with the corresponding code implementation. For every element, document whether it is implemented, partially implemented, or missing in the code.

        Then, provide a detailed summary covering the following:
        1. Model selection changes (if any)
        2. Validation metrics and performance score differences
        3. Modifications in hyperparameters
        4. Additions or removals in features used

        Finally, conclude with a clear summary. If the code implementation does not align with the white paper, explicitly state:
        “White paper needs to be updated” or “Code is not aligned with the current white paper.”
      
        """)
    ]
    return llm(prompt).content.strip()

def summarize(whitepaper_text):
    prompt = [
        SystemMessage(content="You are a product analyst."),
        HumanMessage(content=f"""
        Here is the whitepaper or product requirement document with code:

        {whitepaper_text}
         
        Give brief objective of white paper at top. Summarize and condense the content from both the updated version and the white paper into a structured report. Remove any duplicate information and present the findings in a clear, organized format.
        Follow sequence: Model Overview, train_test_split, Preprocessing step, Feature selection and Handle Imbalance data, Model Architecture, Base Line Metrics and Fallback mechanism
            Report Structure:
            1. Model Overview
                - Extract and compare model information from the notebook and white paper
                - Highlight differences or similarities in:
                - Data splitting strategy train_test_split 
                - Preprocessing steps
                - Name of Features and select features, and display all features. Feature engineering steps
                - Handling of imbalanced data (if discussed)
                - Model architecture or Main model used for prediction
                - Baseline metrics and respective scores 
                - Fallback mechanisms (if any)
                - High light any mismatches or changes for above pointers


            2. Validation Metrics
                - Compile validation metrics and scores from both the white paper and the notebook/updated version.
                - Note any mismatches or changes.

            3. Hyperparameter Configuration
                - List hyperparameters used in the model(s) from both sources.
                - Note any mismatches or changes.

            4. Final Summary
                - Clearly state whether the notebook and white paper are aligned.
                - If not, include the note: "White paper is not aligned with the code. Please update the white paper accordingly."
        """)
    ]
    return llm(prompt).content.strip()

queries = [
    "Summary/Objective of white paper ",
    "Select All seatures/ Features Name ",
    "Training and resting methodology",
    "Preprocessing steps and data transformation steps",
    "Model selected for classification",
    "List of Hyper parameters and respective values",
    "What are list of validation scores and the performance scores?",
    "Ethical considerations"
]


def main():
    st.set_page_config(page_title="Functionality Coverage Checker", layout="wide")
    st.title("🧠 AI Feature Mapping Validator")
    st.subheader("Compare functionalities between a Whitepaper and its Codebase")

    whitepaper_text = st.text_area("📄 Paste Whitepaper Content", height=300, key="whitepaper")
    version = st.text_area("💻 Paste Code or Notebook Content", height=300, key="code")

    # uploaded_whitepaper = st.file_uploader("📄 Upload Whitepaper File", type=["txt", "md", "pdf"]) # Not required  
    # uploaded_code = st.file_uploader("💻 Upload Code File", type=["py", "txt", "ipynb"])   # not required 

    if whitepaper_text and version:
        if st.button("Click to Process Files"):

            # whitepaper_hash = get_file_hash(uploaded_whitepaper)
            # code_hash = get_file_hash(uploaded_code)

           # --- Show spinner for feedback ---
            # with st.spinner("Extracting and embedding whitepaper..."):
            #     whitepaper = extract_from_pdf(uploaded_whitepaper)
            #     collection_wp, chunks_wp = get_or_create_embeddings(
            #         uploaded_file=uploaded_whitepaper,
            #         text=whitepaper,
            #         _embedding_model=embedding_model,
            #         collection_name=f"whitepaper_{whitepaper_hash}"
            #     )

            # st.write('Getting relevant chunks from vector database for white paper')
            # with st.spinner("Running vector search for whitepaper..."):
            #     # --- Parallel vector queries ---
            #     list_pdf_docs = queryFun_parallel(queries, embedding_model, collection_wp)

            # st.write('Refining extracted chunks from vector database for white paper')
            # with st.spinner("Refining chunks..."):
            #     with ThreadPoolExecutor() as executor:
            #         list_refine_context_from_extracted_element_from_pdf = list(
            #             executor.map(
            #                 refine_extracted_elements_with_context,
            #                 list_pdf_docs, queries
            #             )
            #         )

            # if uploaded_code.name.endswith(".ipynb"):
            #     with tempfile.NamedTemporaryFile(delete=False, suffix=".ipynb", mode='wb') as tmp_file:
            #         tmp_file.write(uploaded_code.read())
            #         temp_file_path = tmp_file.name
            #     notebook_contents = read_notebook_with_outputs(temp_file_path)
            #     with st.spinner("Extracting and embedding notebook..."):
            #         collection_nb, chunks_nb = get_or_create_embeddings(
            #             uploaded_file=uploaded_code,
            #             text=notebook_contents,
            #             _embedding_model=embedding_model,
            #             collection_name=f"notebook_{code_hash}"
            #         )
            #     st.write('Getting relevant chunks from vector database for model')
            #     with st.spinner("Running vector search for codebase..."):
            #         list_notebook_queries_item = queryFun_parallel(queries, embedding_model, collection_nb)
            #     st.write('Refining extracted chunks from vector database for model')
            #     with ThreadPoolExecutor() as executor:
            #         list_refine_context_from_extracted_element_from_markdown = list(
            #             executor.map(
            #                 refine_extracted_elements_with_context,
            #                 list_notebook_queries_item, queries
            #             )
            #         )
            # else:
            #     code = uploaded_code.read().decode("utf-8")
            #     code_funcs = extract_functionalities_from_code(code)
            #     list_refine_context_from_extracted_element_from_markdown = [code_funcs] * len(queries)

            # st.write('Comparing functionalities (this may take a moment)...')
            # def compare_all():
            #     with ThreadPoolExecutor() as executor:
            #         return list(executor.map(
            #             compare_functionalities,
            #             list_refine_context_from_extracted_element_from_pdf,
            #             list_refine_context_from_extracted_element_from_markdown
            #         ))
            
            # list_missing_funcs = compare_all()
            # st.write('Missing functionality', list_missing_funcs)
            # st.write("-------------------------------------------------------")
            # st.write('Summarizing report findings...')
            # output = summarize("\n\n".join(list_missing_funcs))
            # st.write(output)                                           # output


if __name__ == "__main__":
    main()
