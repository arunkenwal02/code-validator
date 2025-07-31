from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate
from IPython.display import Markdown, display
from dotenv import load_dotenv
load_dotenv()
import os
import nbformat
import streamlit as st
import tempfile

openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o",  
                 temperature=0,
                 openai_api_key= openai_api_key)




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


def compare_functionalities(whitepaper_funcs, code_funcs):
    """Compares two sets of functionalities using the LLM."""
    prompt = [
        SystemMessage(content="You are a software QA expert."),
        HumanMessage(content=f"""
        Whitepaper Functionalities:
        {whitepaper_funcs}

        Code Functionalities:
        {code_funcs}
        Extract validation metrics from code funcs eg, precision, recall and other validation are in output cell.
        Compare the two lists and identify which functionalities from the whitepape, if functionality is implemented in code but not available in white paper, print: white paper is not updated please update the document. and show details of each section 
        listmissing sections like feature and if model varies according to white paper and same for validation metrics.
        compare validation scores : Compare scores of code function with white paper.
        Also compare critical validation metrics: make sure critical metrics of code should be grater then white paper critical metrics
        if thereis no changhe in metrics of docuemt and code_funcs: keep output 'white paper is updated please proceed to next steps. no other information is required'  
        
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


def main():
    st.set_page_config(page_title="Functionality Coverage Checker", layout="wide")
    
    st.title("🧠 AI Feature Mapping Validator")
    st.subheader("Compare functionalities between a Whitepaper and its Codebase")

    uploaded_whitepaper = st.file_uploader("📄 Upload Whitepaper File", type=["txt", "md", "pdf"])
    uploaded_code = st.file_uploader("💻 Upload Code File", type=["py", "txt", "ipynb"])

    if uploaded_whitepaper and uploaded_code:
        if st.button("Click to Process Files"):
            # Read whitepaper content
            whitepaper = uploaded_whitepaper.read().decode("utf-8")

            # Handle .ipynb or other code files
            if uploaded_code.name.endswith(".ipynb"):
                # Write the raw content to a temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".ipynb", mode='wb') as tmp_file:
                    tmp_file.write(uploaded_code.read())
                    temp_file_path = tmp_file.name

                # notebook_contents = read_notebook(temp_file_path)
                notebook_contents = read_notebook_with_outputs(temp_file_path)
                code_funcs = extract_functionalities_from_code(notebook_contents)
            else:
                code = uploaded_code.read().decode("utf-8")
                code_funcs = extract_functionalities_from_code(code)

            whitepaper_funcs = extract_functionalities_from_whitepaper(whitepaper)

            st.markdown("### ⚖️ Comparing Functionalities")
            missing_funcs = compare_functionalities(whitepaper_funcs, code_funcs)
            st.markdown(missing_funcs)

if __name__ == "__main__":
    main()

