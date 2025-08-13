
from fastapi import FastAPI, HTTPException, Query
from typing import Optional, Tuple, List
import os
import json
import base64
import time
import hashlib
from datetime import datetime
import requests
import fitz
import pandas as pd
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
# --- LLM / LangChain ---
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set.")

# Models
llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Chroma (persistent)
CHROMA_PATH = "./chroma_openai1"
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# OneDrive / MSAL config (you were using a public client)
import msal
MSAL_CLIENT_ID = "6a94cb3a-9869-4b54-ae0b-f4f523df2614"  # consider moving to env
MSAL_AUTHORITY = "https://login.microsoftonline.com/consumers"
MSAL_SCOPES = ["Files.Read"]

# GitHub config
GITHUB_OWNER = "arunkenwal02"
GITHUB_REPO = "code-validator"
NOTEBOOK_FILE_PATH = "loan-approval-prediction_v1.ipynb"

# Output file (as in your code)
OUTPUT_TXT = "white_paper_comparision.txt"

# -------------------------------- Utils --------------------------------

def get_file_hash(uploaded_file) -> str:
    uploaded_file.seek(0)
    hash_val = hashlib.sha256(uploaded_file.read()).hexdigest()
    uploaded_file.seek(0)
    return hash_val

def get_string_hash(text_data: str) -> str:
    return hashlib.sha256(text_data.encode("utf-8")).hexdigest()

def collection_exists(collection_name: str) -> bool:
    try:
        chroma_client.get_collection(collection_name)
        return True
    except Exception:
        return False

def store_in_chromaDB(chunks: List[str], embeddings: List[List[float]], path: str, collection_name: str):
    if not (isinstance(chunks, list) and isinstance(embeddings, list)):
        raise TypeError("Both chunks and embeddings must be lists.")
    if len(chunks) != len(embeddings):
        raise ValueError(f"Length mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings.")
    if not all(isinstance(e, list) for e in embeddings):
        raise TypeError("Each embedding should be a list of floats (i.e., a vector).")

    _client = chromadb.PersistentClient(path=path)
    collection = _client.get_or_create_collection(name=collection_name)
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk-{i}" for i in range(len(chunks))]
    )
    return collection

def get_or_create_embeddings(text: str, _embedding_model, collection_name: str):
    chunks = create_chunks(text)
    if collection_exists(collection_name):
        collection = chroma_client.get_collection(collection_name)
    else:
        embeddings = _embedding_model.embed_documents(chunks)
        collection = store_in_chromaDB(chunks, embeddings, CHROMA_PATH, collection_name)
    return collection, chunks

# -------------------- MSAL token (device flow fallback) --------------------

def access_token_key(client_id: str, authority: str) -> str:
    app_msal = msal.PublicClientApplication(client_id=client_id, authority=authority)
    result = None

    # Try silent
    accounts = app_msal.get_accounts()
    if accounts:
        result = app_msal.acquire_token_silent(MSAL_SCOPES, account=accounts[0])

    # Try interactive (only works if you actually have a UI)
    if not result:
        try:
            result = app_msal.acquire_token_interactive(scopes=MSAL_SCOPES)
        except Exception:
            # Fallback to device code (server-friendly)
            flow = app_msal.initiate_device_flow(scopes=MSAL_SCOPES)
            if "user_code" not in flow:
                raise RuntimeError("Failed to create device flow.")
            # Log the code so you can complete auth out-of-band
            print(f"[MSAL] To authorize, visit {flow['verification_uri']} and enter code: {flow['user_code']}")
            result = app_msal.acquire_token_by_device_flow(flow)

    if not result or "access_token" not in result:
        raise RuntimeError(f"MSAL auth error: {result}")
    return result["access_token"]

# -------------------- OneDrive PDF helpers --------------------

def prev_version(client_id: str, authority: str, file_path: str, version_number: int) -> str:
    access_token = access_token_key(client_id=client_id, authority=authority)
    headers = {"Authorization": f"Bearer {access_token}"}

    versions_url = f"https://graph.microsoft.com/v1.0/me/drive/root:/{file_path}:/versions"
    response = requests.get(versions_url, headers=headers)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch versions: {response.status_code} {response.text}")

    versions = response.json().get("value", [])
    if not versions:
        raise RuntimeError("No versions found for this file.")

    def _parse_dt(v):
        ts = v.get("lastModifiedDateTime")
        return datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else datetime.min

    versions.sort(key=_parse_dt, reverse=True)
    total = len(versions)
    if not (1 <= int(version_number) <= total):
        raise ValueError(f"Invalid version_number {version_number}. Only {total} versions exist.")

    selected = versions[int(version_number) - 1]
    internal_id = selected["id"]

    download_url = f"https://graph.microsoft.com/v1.0/me/drive/root:/{file_path}:/versions/{internal_id}/content"
    version_response = requests.get(download_url, headers=headers)
    if version_response.status_code != 200:
        raise RuntimeError(f"Failed to download version #{version_number}: {version_response.status_code} {version_response.text}")

    pdf_bytes = version_response.content

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    all_text = ""
    for page_num, page in enumerate(doc):
        all_text += f"\n--- Page {page_num+1} ---\n{page.get_text()}"
    return all_text


# pip install pymupdf requests

import fitz  # pymupdf
import requests
from io import BytesIO
from urllib.parse import quote



def read_white_paper_from_gcp(filename, base_url, version_number):
    # Safely encode filename for a URL

    filename = "Load Prediction Whitepaper.pdf"
    filename.split(".")
    encoded_filename = quote(filename.split(".")[0])
    file_type = filename.split(".")[1]
    url = f"{base_url}{encoded_filename}_v{version_number}.{file_type}"

    # Download the PDF into memory
    response = requests.get(url)
    response.raise_for_status()

    # Open PDF from bytes
    pdf_stream = BytesIO(response.content)
    doc = fitz.open(stream=pdf_stream, filetype="pdf")
    text_block = ""
    # Iterate through pages
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")  # Extract as plain text
        text_block += f"--- Page {page_num + 1} ---\n{text}\n\n"
        print(f"--- Page {page_num + 1} ---")
        print(text)
        # print()
    return text_block 

 

def extract_from_pdf(whitepaper_name : str, base_url : str,version_number : int) -> str:
    return read_white_paper_from_gcp(whitepaper_name, base_url, version_number)

# def extract_from_pdf(client_id: str, authority: str, file_path: str, version_number: int) -> str:
#     return prev_version(client_id=client_id, authority=authority, file_path=file_path, version_number=version_number)

# -------------------- GitHub helpers --------------------

def get_sha_pair_from_push_id(owner: str, repo: str, push_id: str) -> Tuple[Optional[str], Optional[str]]:
    url = f"https://api.github.com/repos/{owner}/{repo}/events"
    resp = requests.get(url)
    if resp.status_code != 200:
        raise RuntimeError(f"GitHub events fetch failed: {resp.status_code} {resp.text}")
    events = resp.json()
    for event in events:
        if event.get("type") == "PushEvent" and event.get("id") == str(push_id):
            before_sha = event["payload"]["before"]
            head_sha = event["payload"]["head"]
            return before_sha, head_sha
    return None, None

def fetch_latest_file_for_sha(owner: str, repo: str, notebook_file_path: str, sha_pairs: List[Tuple[str, str]]) -> dict:
    for (sha_old, sha_new) in sha_pairs:
        compare_url = f"https://api.github.com/repos/{owner}/{repo}/compare/{sha_old}...{sha_new}"
        compare_resp = requests.get(compare_url)
        if compare_resp.status_code != 200:
            raise RuntimeError(f"GitHub compare failed: {compare_resp.status_code} {compare_resp.text}")
        compare_data = compare_resp.json()

        file_changed = any(f.get("filename") == notebook_file_path for f in compare_data.get("files", []))

        if file_changed:
            content_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{notebook_file_path}"
            params = {"ref": sha_new}
        else:
            commits_url = f"https://api.github.com/repos/{owner}/{repo}/commits"
            params = {"path": notebook_file_path, "per_page": 1}
            commits_resp = requests.get(commits_url, params=params)
            if commits_resp.status_code != 200:
                raise RuntimeError(f"GitHub commits fetch failed: {commits_resp.status_code} {commits_resp.text}")
            last_update_sha = commits_resp.json()[0]["sha"]
            content_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{notebook_file_path}"
            params = {"ref": last_update_sha}

        file_resp = requests.get(content_url, params=params)
        if file_resp.status_code != 200:
            raise RuntimeError(f"GitHub content fetch failed: {file_resp.status_code} {file_resp.text}")

        file_data = file_resp.json()
        if "content" not in file_data:
            raise RuntimeError("Notebook not found or could not fetch content. Details: " + str(file_data))
        nb_json = base64.b64decode(file_data["content"]).decode("utf-8")
        return json.loads(nb_json)

    raise RuntimeError("No SHA pairs yielded a notebook.")

def read_notebook_with_outputs(owner: str, repo: str, push_id: str, notebook_file_path: str) -> str:
    sha_pair = get_sha_pair_from_push_id(owner=owner, repo=repo, push_id=push_id)
    if not sha_pair or not sha_pair[0] or not sha_pair[1]:
        raise RuntimeError(f"Push ID {push_id} not found in recent events.")
    notebook_contents = fetch_latest_file_for_sha(owner=owner, repo=repo, notebook_file_path=notebook_file_path, sha_pairs=[sha_pair])

    all_cells_text = ""
    for i, cell in enumerate(notebook_contents.get('cells', [])):
        if cell.get('cell_type') == 'code' and cell.get('outputs'):
            all_cells_text += f"\nCell #{i+1}\n"
            all_cells_text += "Code:\n"
            all_cells_text += "".join(cell.get('source', [])).strip() + "\n"
            all_cells_text += "Output(s):\n"
            for output in cell['outputs']:
                output_text = ""
                if output.get('output_type') == 'stream':
                    text = output.get('text', '')
                    if isinstance(text, list):
                        text = "".join(text)
                    output_text += (text or "").strip()
                elif output.get('output_type') in ['execute_result', 'display_data']:
                    data = output.get('data', {})
                    text = data.get('text/plain', '')
                    if isinstance(text, list):
                        text = "".join(text)
                    output_text += (text or "").strip()
                if output_text:
                    all_cells_text += output_text + "\n"
            all_cells_text += "-" * 30 + "\n"

    return all_cells_text.strip()

# -------------------- Embedding / querying helpers --------------------

def create_chunks(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)
    return splitter.split_text(text or "")

def queryFun(query: str, _embedding_model, collection):
    query_embedding = _embedding_model.embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=5)
    return [doc for doc in results.get("documents", [[]])[0]]

def queryFun_parallel(queries: List[str], _embedding_model, collection):
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(queryFun, q, _embedding_model, collection) for q in queries]
        out = []
        for fut in as_completed(futures):
            out.append(fut.result())
        # preserve input order (optional); current code returns as completed
        return out

# -------------------- LLM prompts --------------------

context_list = [
    "From the provided HTML or text, extract summary overview in 2 lines only, excluding all other sections or details.",
    "Keep only features name. Do not include descriptions, preprocessing details, training methodology, target variable explanations, or any other text.",
    "Include only the train/test percentages and their purposes if mentioned. Do not include hyperparameter tuning, validation strategy, retraining details, evaluation metrics, or deployment strategy.",
    "Strictly Keep Model name only. exclude other details/information.",
    "Extract only the validation/performance metrics with their scores and the best hyperparameter scores, excluding all other details.",
    "Keep only ethical considerations. do not include other details."
]

def refine_extracted_elements_with_context(similar_elements, query_context, context_list_ele):
    combined_elements = "\n\n".join(similar_elements)
    prompt = [
        SystemMessage(content="You are a product analyst."),
        HumanMessage(content=f"""
The following are the top 5 similar elements retrieved from a vector database and create a structured report in HTML format with the following three sections, using dangerouslySetInnerHTML={{ __html: reportMarkdown }}; html should not affect other elements: 
{combined_elements}

The original query context is:
"{query_context}"
- {context_list_ele}
- Identify and extract only the most relevant elements or functionalities.
- Do not recommend, only extract.
- Avoid verbose explanations; focus on clarity and precision.
- Provide concise, bullet-pointed outputs or insights based on retrieved data.
""")
    ]
    
    resp = llm.invoke(prompt)
    return resp.content.strip()
    # return llm(prompt).content.strip()

def compare_functionalities(whitepaper_funcs, code_funcs):
    prompt = [
        SystemMessage(content="You are a AI report comparision tool."),
        HumanMessage(content=f"""
Whitepaper Functionalities:
{whitepaper_funcs}

Code Functionalities and create a structured report in HTML format with the following three sections, using dangerouslySetInnerHTML={{ __html: reportMarkdown }}; html should not affect other elements
{code_funcs}

Compare each functionality described in the white paper funcs with the corresponding code funcs implementation.
If there is any mismatch highlight that with mismatch.

eg: 
White Paper:
Code:
Mismatch: 
""")
    ]
    return llm(prompt).content.strip()

def summarize(whitepaper_text):
    prompt = [
        SystemMessage(content="You are a product analyst."),
        HumanMessage(content=f"""
The HTML will be rendered in React via dangerouslySetInnerHTML={{ __html: reportMarkdown }} — ensure the markup is self-contained and does not affect other page elements:

Create exactly three top-level sections with these subheadings, in this order: White paper, code, Mismatch.

Stick strictly to the input content; do not add, remove, or invent information.
{whitepaper_text}
         
Give brief objective of white paper at top. 
Follow sequence: Summart, Model Overview, train_test_split, Preprocessing step, Feature selection and Handle Imbalance data, Model Architecture, Base Line Metrics and Fallback mechanism
    Report Structure:
    1. Summary: overall summry

    2. Preprocesing steps: 
        - Data splitting strategy train_test_split 
        - Name of Features and select features, and display all features. Feature engineering steps

    3. Handle Imbalance data: 
        - Methods to hanlde imbalance data on both Code and white paper

    4. Feature Selection:
        - list down all featuress used in white paper and code.
        - show Mismatch if there is any mismatch in fetures

    5. Model Overview
        - Extract and compare model information from the notebook and white paper ,Model architecture or Main model used for prediction.
        - show Mismatch if there is any mismatch in Model Overview.

    6. Validation Metrics
        - Show all validation metrics and scores from both the white paper and the notebook/updated version.
        - show mismatch if there is any mismatch in validation metrics.

    7. Hyperparameter Configuration
        - List hyperparameters used in the model(s) and respective scores.
        - Mismatches id there is any otherwise there is no mismatch in code and white papaer.

    8. Critical metrics:
        - List critical metics if available.
        -show mismatch if there is any mismatch in critical metrics.

    9. Fallback mechanism
        - List fall back mechanish present in white paper and code.
        - Show Mismatch if there is any mismatch in Fallback mechanism.

    10. Final Summary
        - Clearly state whether the notebook and white paper are aligned.
        - If not, include the note: "White paper is not aligned with the code. Please update the white paper accordingly."
""")
    ]

    resp = llm.invoke(prompt)
    return resp.content.strip()
    # return llm(prompt).content.strip()

QUERIES = [
    "Summary/Objective of white paper ",
    "Select All features/ Features Name ",
    "Training and resting methodology",
    "Preprocessing steps and data transformation steps",
    "Model selected for classification",
    "List of Hyper parameters and respective values",
    "What are list of validation scores and the performance scores?",
    "Ethical considerations"
]

# -------------------- Main orchestration --------------------

def get_latest_push_id_from_file(path: str = "push_events.json") -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    df = pd.read_json(path)
    # your code did: push_event[0].tolist()[0]
    # replicate safely:
    first_col = df.columns[0]
    return str(df[first_col].tolist()[0])


def main(whitepaper_name: str, version_number: int) -> str:
    try:
        base_url = "https://storage.googleapis.com/whitepaper_test/"
        # 1) Whitepaper text via OneDrive version
        whitepaper_text = extract_from_pdf(whitepaper_name, base_url, version_number)
        whitepaper_hash = get_string_hash(whitepaper_text)
        collection_wp, _ = get_or_create_embeddings(
            text=whitepaper_text,
            _embedding_model=embedding_model,
            collection_name=f"whitepaper_{whitepaper_hash}"
        )


        # # 1) Whitepaper text via OneDrive version
        # file_path = f"Documents/GitHub/code-validator/{whitepaper_name}"
        # whitepaper_text = extract_from_pdf(file_path)
        # whitepaper_hash = get_string_hash(whitepaper_text)
        # collection_wp, _ = get_or_create_embeddings(
        #     text=whitepaper_text,
        #     _embedding_model=embedding_model,
        #     collection_name=f"whitepaper_{whitepaper_hash}"
        # )

        # 2) Query & refine for whitepaper
        list_pdf_docs = queryFun_parallel(QUERIES, embedding_model, collection_wp)
        with ThreadPoolExecutor() as executor:
            refined_pdf = list(
                executor.map(
                    refine_extracted_elements_with_context,
                    list_pdf_docs, QUERIES, context_list
                )
            )

        # 3) Notebook text via GitHub push
        push_id = get_latest_push_id_from_file()
        notebook_text = read_notebook_with_outputs(GITHUB_OWNER, GITHUB_REPO, push_id, NOTEBOOK_FILE_PATH)
        code_hash = get_string_hash(notebook_text)
        collection_nb, _ = get_or_create_embeddings(
            text=notebook_text,
            _embedding_model=embedding_model,
            collection_name=f"notebook_{code_hash}"
        )

        list_notebook_docs = queryFun_parallel(QUERIES, embedding_model, collection_nb)
        with ThreadPoolExecutor() as executor:
            refined_nb = list(
                executor.map(
                    refine_extracted_elements_with_context,
                    list_notebook_docs, QUERIES, context_list
                )
            )

        # 4) Compare
        def compare_all():
            with ThreadPoolExecutor() as executor:
                return list(executor.map(
                    compare_functionalities,
                    refined_pdf,
                    refined_nb
                ))

        comparisons = compare_all()

        # 5) Summarize and persist
        output = summarize("\n\n".join(comparisons))
        with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
            f.write(output)

        return output

    except Exception as e:
        # Bubble up so FastAPI can return proper error
        raise

