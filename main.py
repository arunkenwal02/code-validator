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

def store_in_chromaDB(chunks, embeddings, collection_name):
    if collection_exists(collection_name):
        return chroma_client.get_collection(collection_name)
    collection = chroma_client.get_or_create_collection(name=collection_name)
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk-{i}" for i in range(len(chunks))]
    )
    return collection


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


def extract_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    extracted_text = ""
    for page_num, page in enumerate(doc):
        text = page.get_text()
        extracted_text += f"\n\n--- Page {page_num + 1} ---\n{text}"
    return extracted_text


def read_notebook_with_outputs(file_path):
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
        - Extract relevant scores, metrics and model name if available in context and 
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
        Here is the whitepaper or product requirement document:

        {whitepaper_text}
         
        Give brief objective of white paper at top. Summarize and condense the content from both the notebook and the white paper into a structured report. Remove any duplicate information and present the findings in a clear, organized format.

            Report Structure:
            1. Model Overview
                - Extract and compare model information from the notebook and white paper.
                - Highlight differences or similarities in:
                - Feature selection and feature engineering
                - Model architecture or type
                - Data splitting strategy
                - Baseline metrics
                - Fallback mechanisms (if any)
                - Data import and preprocessing steps
                - Handling of imbalanced data (if discussed)

            2. Validation Metrics
                - Compile validation metrics and scores from both the white paper and the notebook.
                - Highlight any discrepancies.

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
    "Features mentioned",
    "Preprocessing steps and data transformation steps",
    "Model selected for classification",
    "Training and resting methodology",
    "List of Hyper parameters and respective values",
    "What are list of validation scores and the performance scores?",
    "Ethical considerations"
]

def main():
    st.set_page_config(page_title="Functionality Coverage Checker", layout="wide")
    st.title("🧠 AI Feature Mapping Validator")
    st.subheader("Compare functionalities between a Whitepaper and its Codebase")

    uploaded_whitepaper = st.file_uploader("📄 Upload Whitepaper File", type=["txt", "md", "pdf"])
    uploaded_code = st.file_uploader("💻 Upload Code File", type=["py", "txt", "ipynb"])

    if uploaded_whitepaper and uploaded_code:
        if st.button("Click to Process Files"):

            whitepaper_hash = get_file_hash(uploaded_whitepaper)
            code_hash = get_file_hash(uploaded_code)

            # --- Show spinner for feedback ---
            with st.spinner("Extracting and embedding whitepaper..."):
                whitepaper = extract_from_pdf(uploaded_whitepaper)
                collection_wp, chunks_wp = get_or_create_embeddings(
                    uploaded_file=uploaded_whitepaper,
                    text=whitepaper,
                    _embedding_model=embedding_model,
                    collection_name=f"whitepaper_{whitepaper_hash}"
                )

            st.write('Getting relevant chunks from vector database for white paper')
            with st.spinner("Running vector search for whitepaper..."):
                # --- Parallel vector queries ---
                list_pdf_docs = queryFun_parallel(queries, embedding_model, collection_wp)

            st.write('Refining extracted chunks from vector database for white paper')
            with st.spinner("Refining chunks..."):
                with ThreadPoolExecutor() as executor:
                    list_refine_context_from_extracted_element_from_pdf = list(
                        executor.map(
                            refine_extracted_elements_with_context,
                            list_pdf_docs, queries
                        )
                    )

            if uploaded_code.name.endswith(".ipynb"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".ipynb", mode='wb') as tmp_file:
                    tmp_file.write(uploaded_code.read())
                    temp_file_path = tmp_file.name
                notebook_contents = read_notebook_with_outputs(temp_file_path)
                with st.spinner("Extracting and embedding notebook..."):
                    collection_nb, chunks_nb = get_or_create_embeddings(
                        uploaded_file=uploaded_code,
                        text=notebook_contents,
                        _embedding_model=embedding_model,
                        collection_name=f"notebook_{code_hash}"
                    )
                st.write('Getting relevant chunks from vector database for model')
                with st.spinner("Running vector search for codebase..."):
                    list_notebook_queries_item = queryFun_parallel(queries, embedding_model, collection_nb)
                st.write('Refining extracted chunks from vector database for model')
                with ThreadPoolExecutor() as executor:
                    list_refine_context_from_extracted_element_from_markdown = list(
                        executor.map(
                            refine_extracted_elements_with_context,
                            list_notebook_queries_item, queries
                        )
                    )
            else:
                code = uploaded_code.read().decode("utf-8")
                code_funcs = extract_functionalities_from_code(code)
                list_refine_context_from_extracted_element_from_markdown = [code_funcs] * len(queries)

            st.write('Comparing functionalities (this may take a moment)...')
            def compare_all():
                with ThreadPoolExecutor() as executor:
                    return list(executor.map(
                        compare_functionalities,
                        list_refine_context_from_extracted_element_from_pdf,
                        list_refine_context_from_extracted_element_from_markdown
                    ))
            list_missing_funcs = compare_all()
            st.write('Summarizing report findings...')
            output = summarize("\n\n".join(list_missing_funcs))
            st.write(output)

if __name__ == "__main__":
    main()
