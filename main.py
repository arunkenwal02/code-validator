import os
import nbformat
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings

import tempfile

client = chromadb.Client(Settings())

load_dotenv()
import streamlit as st
import fitz
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o",  
                 temperature=0,
                 openai_api_key= openai_api_key)

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small") 



def read_notebook(file_path):
    """Read .ipynb notebook and extract content."""
    nb = nbformat.read(file_path, as_version=4)
    content = []
    for cell in nb.cells:
        if cell.cell_type == 'markdown':
            content.append("## Markdown Cell:\n" + cell.source)
        elif cell.cell_type == 'code':
            content.append("## Code Cell:\n```python\n" + cell.source + "\n```")
    return "\n\n".join(content)


def read_file(file_path):
    """Reads the content of a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def extract_functionalities_from_code(notebook_content):

    """Uses LLM to extract functionalities from Python code."""
    prompt = f"""
    You are an expert Python code reviewer. Here is a Jupyter notebook:
    {notebook_content}

    The following is a Jupyter notebook content (code and markdown). 
    Please extract the following:
    Analyze the notebook and answer:

    1. List of features used in the model.
    2. Name/type of ML model used, only name of model
    3. Accuracy metrics (e.g., accuracy, F1, precision, recall, AUC, etc.), only metrics name. 
    4. What is the purpose of this notebook?
    5. What are the main operations and their results?
    6. Are there any errors or anomalies in outputs?
    7. What conclusions can be drawn from the outputs?

    """

    response = llm.invoke([SystemMessage(content="You are a helpful assistant."), HumanMessage(content=prompt)])

    return response.content.strip()


def extract_functionalities_from_whitepaper(whitepaper_text):
    """Uses LLM to extract functionalities from whitepaper."""
    prompt = [
        SystemMessage(content="You are a product analyst."),
        HumanMessage(content=f"""
        Here is the whitepaper or product requirement document:

        {whitepaper_text}

        List all functionalities or features the whitepaper mentions. Use bullet points.
        """)
            ]
    return llm(prompt).content.strip()


def refine_extracted_elements_with_context(similar_elements, query_context):
    """Uses LLM to refine extracted elements based on query context."""
    combined_elements = "\n\n".join(similar_elements)
    prompt = [
        SystemMessage(content="You are a product analyst."),
        HumanMessage(content=f"""
        The following are the top 5 similar elements retrieved from a vector database:

        {combined_elements}

        The original query context is:
        "{query_context}"

        
        - **Objective**: Refine query context and extract only the most relevant functionalities and insights.
        - **Instructions**:
        - Analyze the query context to determine the user's intent.
        - Identify and extract only the most relevant elements or functionalities.
        - Provide concise, bullet-pointed outputs or insights based on retrieved data.
        - Avoid verbose explanations; focus on clarity and precision.
        - Format the response using IPython Markdown style for readability
        
        """)
    ]
    return llm(prompt).content.strip()


def compare_functionalities(whitepaper_funcs, code_funcs):
    """Compares two sets of functionalities using the LLM."""
    prompt = [
        SystemMessage(content="You are a software QA expert."),
        HumanMessage(content=f"""
        Whitepaper Functionalities:
        {whitepaper_funcs}

        Code Functionalities:
        {code_funcs}
        
        compare elements of white paper with code funcs and note it down with each respect. 
        1.Track and summarize each change in feature selection, model selection, validation metrics 

      
        """)
            ]
    return llm(prompt).content.strip()


def read_notebook_with_outputs(file_path):
    """Read .ipynb notebook and include both code and output."""
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
                    # Display the result of the cell (e.g., print(2+2))
                    result = output.get("data", {}).get("text/plain", "")
                    outputs.append(f"Output (execute_result):\n{result}")
                elif output.output_type == "error":
                    outputs.append("Error:\n" + "\n".join(output.get("traceback", [])))

            full_output = "\n".join(outputs)
            if full_output:
                code += f"\n\n### Output:\n```\n{full_output}\n```"
            cells_content.append(code)

    return "\n\n".join(cells_content)


def read_notebook(path):
    with open(path, "r", encoding="utf-8") as f:
        return nbformat.read(f, as_version=4)
    

def extract_from_pdf(uploaded_file):
    
    if uploaded_file is not None:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")

    extracted_text = ""
    for page_num, page in enumerate(doc):
        text = page.get_text()
        extracted_text += f"\n\n--- Page {page_num + 1} ---\n{text}"

    return extracted_text

    
    '''
    # doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    doc = fitz.open(stream=uploaded_file.encode('utf-8'), filetype="pdf")

    extracted_text = ""
    for page_num, page in enumerate(doc):
            text = page.get_text()
            extracted_text += f"\n\n--- Page {page_num + 1} ---\n{text}"

    return extracted_text
    '''


def create_chunks(pdf_text):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,     # tokens (approx. 750–1000 words)
    chunk_overlap=200,   # overlap to preserve context
    )
    chunks = text_splitter.split_text(pdf_text)

    return chunks


def create_embeddings(embedding_model, chunks):
    
    embeddings = embedding_model.embed_documents(chunks)
    return embeddings


def store_in_chromaDB(chunks,embeddings, path , collection_name):
    # Validate input
    if not (isinstance(chunks, list) and isinstance(embeddings, list)):
        raise TypeError("Both chunks and embeddings must be lists.")
    if len(chunks) != len(embeddings):
        raise ValueError(f"Length mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings.")
    if not all(isinstance(e, list) for e in embeddings):
        raise TypeError("Each embedding should be a list of floats (i.e., a vector).")

    chroma_client = chromadb.PersistentClient(path= path)
    collection = chroma_client.get_or_create_collection(name= collection_name)


    # Add data to collection
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk-{i}" for i in range(len(chunks))]
    )
    return collection


def summarize(whitepaper_text):
    """Uses LLM to extract functionalities from whitepaper."""
    prompt = [
        SystemMessage(content="You are a product analyst."),
        HumanMessage(content=f"""
        Here is the whitepaper or product requirement document:

        {whitepaper_text}
         
        summarise the context from notebook and white paper, remove duplicate details and condense in form of report structure 
        1. Keep model used in notebook and white paper
        2. Keep validation metrics and scores from white paper and notebook
        3. Keep hyperparameters 
        If white paper and code are not align, add in summary, white paper is not aligh with code, please update white paper.
  
        """)
            ]
    return llm(prompt).content.strip()


def queryFun(query, embedding_model,collection):
    query_embedding = embedding_model.embed_query(query)
    l_docs = []
    results = collection.query(query_embeddings=[query_embedding], n_results=5)
    for doc in results["documents"][0]:
        l_docs.append(doc)
        # print("🔎 Match:", l_docs.append(doc))
    return l_docs

queries = [
            "Summary/Objective of white paper ",
            "Features mentioned",
            "Preprocessing steps and data transformation steps",
            "Model selected for classification",
            "Training and resting methodology",
            "List of Hyper parameters and respective values",
            "What are list of validation scores and the performance scores?",
            "Ethical considerations" ]


def main():
    st.set_page_config(page_title="Functionality Coverage Checker", layout="wide")
    
    st.title("🧠 AI Feature Mapping Validator")
    st.subheader("Compare functionalities between a Whitepaper and its Codebase")

    uploaded_whitepaper = st.file_uploader("📄 Upload Whitepaper File", type=["txt", "md", "pdf"])
    uploaded_code = st.file_uploader("💻 Upload Code File", type=["py", "txt", "ipynb"])

    
    if uploaded_whitepaper and uploaded_code:
        if st.button("Click to Process Files"):
            # Read whitepaper content
            whitepaper = extract_from_pdf(uploaded_whitepaper)
            chunks =create_chunks(whitepaper)
            embeddings = create_embeddings(embedding_model = embedding_model,chunks=chunks)
            collection = store_in_chromaDB(chunks=chunks, embeddings=embeddings, path= './chroma_openai1', collection_name= 'whitepaper_embeddings')
            
            list_pdf_docs = []
            st.write('Getting relevant chunks from vector database for white paper')
            for query in queries:
                l_docs = queryFun(query=query,embedding_model=embedding_model,collection =collection)
                list_pdf_docs.append(l_docs)
                
            list_refine_context_from_extracted_element_from_pdf =[]
            st.write('Refining extracted chunks from vector database for white paper')            
            for i in range(len(queries)):
                refine_context_from_extracted_element_from_pdf = refine_extracted_elements_with_context(similar_elements = list_pdf_docs[i], query_context = queries[i]) 
                list_refine_context_from_extracted_element_from_pdf.append(refine_context_from_extracted_element_from_pdf)
        

            # for ele in list_refine_context_from_extracted_element_from_pdf:
            #display(Markdown(ele))
            #print("-------------------------------------")
            
            # Handle .ipynb or other code files
            if uploaded_code.name.endswith(".ipynb"):
                # Write the raw content to a temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".ipynb", mode='wb') as tmp_file:
                    tmp_file.write(uploaded_code.read())
                    temp_file_path = tmp_file.name

                # notebook_contents = read_notebook(temp_file_path)
                notebook_contents = read_notebook_with_outputs(temp_file_path)
                
                # Generating embeddings
                chunks_notebook =create_chunks(notebook_contents)
                embeddings_notebook = create_embeddings(embedding_model = embedding_model,chunks=chunks_notebook)
                # store in vector database
                collection_notebook = store_in_chromaDB(chunks = chunks_notebook,embeddings =embeddings_notebook , path =  "./chroma_openai1", collection_name = "notebook_embeddings")


                list_notebook_queries_item = []
                st.write('Getting relevant chunks from vector database for model')
                for query in queries:
                    l_docs = queryFun(query=query,embedding_model=embedding_model,collection =collection_notebook)
                    list_notebook_queries_item.append(l_docs)

                list_refine_context_from_extracted_element_from_markdown =[]
                st.write('Refining extracted chunks from vector database for model')
                for i in range(len(queries)):
                    refine_context_from_extracted_element_from_notebook = refine_extracted_elements_with_context(similar_elements = list_notebook_queries_item[i], query_context = queries[i]) 
                    list_refine_context_from_extracted_element_from_markdown.append(refine_context_from_extracted_element_from_notebook)
                    
                # for ele in list_refine_context_from_extracted_element_from_markdown:
                    # display(Markdown(ele))
                    # print("-----------------")
                    
            else:
                code = uploaded_code.read().decode("utf-8")
                code_funcs = extract_functionalities_from_code(code)

            list_missing_funcs = []
            for i in range(len(list_refine_context_from_extracted_element_from_markdown)):
                list_missing_funcs.append(compare_functionalities(list_refine_context_from_extracted_element_from_pdf[i], list_refine_context_from_extracted_element_from_markdown[i]))

            # for ele in list_missing_funcs:
           
                # Markdown(ele)
                # print("------------------------------------------------")
                
            list_missing_funcs = "\n\n".join(list_missing_funcs)
            st.write('Summarizing report findings')
            output = summarize(list_missing_funcs)

            st.write(output)
            
            
            # whitepaper_funcs = extract_functionalities_from_whitepaper(whitepaper)

            # st.markdown("### ⚖️ Comparing Functionalities")
            # missing_funcs = compare_functionalities(whitepaper_funcs, code_funcs)
            # st.markdown(missing_funcs)
            
            

if __name__ == "__main__":
    main()