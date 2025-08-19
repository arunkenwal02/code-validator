from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv
from IPython.display import Markdown, display
import os
from pathlib import Path
from langchain_openai import ChatOpenAI


# Load API key
load_dotenv()

MODEL = "gpt-4o"
OUTPUT_FILE = "FinalReport.txt"
openai_api_key = os.getenv("OPENAI_API_KEY")

# Setup LLM
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    openai_api_key=openai_api_key
)

def read_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def generate_summary(Push_Commit_summary: str, White_paper_comparision: str) -> str:
    """Takes two text inputs + instruction, returns LLM output."""
    
    # 2) Define instruction
    instruction = (

    '''
        You are an AI report generator. Based on the inputs provided

        Render the following sections into markdown format with clear headings and subheadings. 
        - Use bullet points for all lists. 
        - Format any tabular data as markdown tables.
        - Ensure readability and structure.
        
        Keep document titles name only, remove document A, b 
        1. Overall summary of the white paper and code comparison, in 3-4 pointers
        2. Extract and include only the textual summaries from push commit summaries in pointers -
        3. Retain the following sections from the white paper comparison:
            - White paper summary
            - Evaluation metrics
            - Model architecture
            - Model monitoring and drift
            - Critical metrics comparison
            - Highlights

        4. Provide Recommendations that:
            - Suggest improvements or next steps based on validation results and code comparison.
            - Include actionable changes to improve accuracy, consistency, or system robustness.

    '''
    )

    user_content = (
        f"{instruction}\n"
        f"## Document A:{Push_Commit_summary}\n"
        f"## Document B:{White_paper_comparision}\n"
    )
    
    return llm.invoke([
        SystemMessage(content="You are a concise, structured assistant. Use Markdown."),
        HumanMessage(content=user_content)
    ]).content



# Push_Commit_summary = read_file("event_summary.txt.txt")
# White_paper_comparision = read_file("white_paper_comparision.txt")





# 3) Get output
# output = generate_summary(Push_Commit_summary, White_paper_comparision)

# 4) Print + save
#print(output)
# Path(OUTPUT_FILE).write_text(output, encoding="utf-8")
# print(f"\nSaved to {OUTPUT_FILE}")