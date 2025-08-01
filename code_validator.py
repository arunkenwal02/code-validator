'''
Generate structured report, summarize code comparision validation and inferences with visual differences and recommendations
'''


'''
Input from Block 1 -> Format ?
Input from Block 2 -> Format ?
Input from Block 3 -> Format ?

Generate report:
Components of report based on given input
1. Code comparision inferences  
2. Validation metrics
3. Recommendations
'''

'''
Table to white paper
Grapg
Font size 12
Font type: Arxiv
17. Performance : 
Features and definitions: 
'''
# ✅ Updated imports based on LangChain v0.2+
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv

import os

# Load API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")




# 3. Setup LLM (GPT Model)
llm = ChatOpenAI(model="gpt-4o",  
                 temperature=0,
                 openai_api_key= openai_api_key)

# 4. Build summarization chain
def CreateReport(input1,input2,input3):

    Prompt= """
    You are an AI report generator. Based on the input below, create a comprehensive and structured report with the following three sections:

    1. **Validation Metrics**  
    - Summarize all model evaluation metrics such as accuracy, precision, recall, F1 score, AUC, or any other relevant performance indicators.  
    - Highlight strengths or weaknesses reflected in these metrics.

    2. **Code Comparison Inferences**  
    - Compare the functionalities or logic of the implemented code with the described features, goals, or documentation (such as a whitepaper).  
    - Mention any discrepancies, missing elements, or deviations from the expected implementation.  
    - Identify parts of the code that successfully align with the intended functionality.

    3. **Recommendations**  
    - Suggest actionable improvements or next steps based on the validation metrics and code comparison.  
    - Include suggestions for improving accuracy, fixing code gaps, optimizing performance, or ensuring alignment with specifications.

    **Input:**  
    {Imput1} , {input2}, {input3}

    <insert evaluation results / code summary / whitepaper / any relevant documentation or content here>

    Generate the output in a clear and concise format, using bullet points where appropriate.

    """



    response = llm.invoke([SystemMessage(content="You are a helpful assistant to generate governanc fopcument for Loan approval system."), HumanMessage(content=Prompt)])

    return response.content.strip()


if __name__ == "__main__":
    input1 ='''
    New Features / Enhancements:
    Added feature engineering pipeline for income, credit history, and loan term normalization.
    Integrated missing value imputation using median/mode strategies.
    Introduced XGBoost and Random Forest classifiers alongside logistic regression for improved model performance.
    Implemented model selection and hyperparameter tuning using GridSearchCV.
    Added streamlit-based frontend for interactive loan approval predictions.
    Included model versioning with MLflow for tracking experiments.
    🐛 Bug Fixes / Code Refactoring:
    Fixed issue with incorrect encoding of categorical variables (replaced LabelEncoder with OneHotEncoder).
    Refactored data loading and preprocessing into modular functions (data_utils.py).
    Improved error handling and logging across preprocessing and inference scripts.
    📁 Repository Structure Updates:
    Created notebooks/, src/, and models/ directories for cleaner project organization.
    Added requirements.txt and README.md with setup instructions.
    Updated .gitignore to exclude model artifacts and environment files.
    📈 Performance Changes:
    Validation accuracy improved from ~78% to 84% with model tuning and feature engineering.
    Reduced training time by 20% after optimizing preprocessing and model pipeline.
    📌 Commit Comparison Highlights:
    Compared commits: a1c2b3d (old baseline model) → d4e5f6g (latest tuned system).
    Major differences:
    Introduction of new ML models and evaluation metrics.
    UI integration for real-time prediction.
    Codebase modularization and documentation improvements.
    '''
    input2 = '''
    Structure & Organization
    V1: Single Jupyter notebook; all logic inline.
    V2: Modular scripts (data_utils.py, model.py, app.py); clean folder structure.
    Inference: Shift from exploratory to production-grade code.
    🧮 Data Preprocessing
    V1: Basic null handling and LabelEncoder.
    V2: SimpleImputer, OneHotEncoder, scaling, ColumnTransformer.
    Inference: More robust and reusable preprocessing pipeline.
    🧠 Feature Engineering
    V1: Used raw features.
    V2: Added domain-driven features (e.g., debt-to-income ratio, loan amount bins).
    Inference: Better input representation, likely improved model performance.
    🤖 Modeling
    V1: Logistic Regression, no tuning.
    V2: Added Random Forest, XGBoost, and GridSearchCV.
    Inference: More powerful models with hyperparameter optimization.
    📊 Evaluation
    V1: Accuracy only.
    V2: Precision, Recall, F1, AUC, confusion matrix.
    Inference: Deeper insight into performance, especially for imbalanced classes.
    🌐 Deployment/UI
    V1: No interface
    V2: Streamlit app for real-time predictions.
    Inference: User-friendly and deployable.

    📈 Experiment Tracking
    V1: None.
    V2: MLflow used for tracking metrics and versions.
    Inference: Enables reproducibility and team collaboration.

    ✅ Final Verdict:
    V2 demonstrates a professional-grade ML system — modular, explainable, user-facing, and maintainable — a significant upgrade over the initial proof-of-concept in V1.

    '''
    
    input3 = '''

Connecting
🧠 AI Feature Mapping Validator
Compare functionalities between a Whitepaper and its Codebase
📄 Upload Whitepaper File

white paper.txt
Drag and drop file here
Limit 200MB per file • TXT, MD, PDF
white paper.txt
329.0B
💻 Upload Code File

model.ipynb
Drag and drop file here
Limit 200MB per file • PY, TXT, IPYNB
model.ipynb
8.9KB

⚖️ Comparing Functionalities
To perform a comprehensive comparison between the whitepaper and the code functionalities, let's break down the information provided and identify any discrepancies or updates needed:

Comparison of Features:
Whitepaper Features:

Sepal width (cm)
Petal length (cm)
Petal width (cm)
Code Features:

Sepal length (cm)
Sepal width (cm)
Petal length (cm)
Petal width (cm)
Missing Feature in Whitepaper:

Sepal length (cm) is used in the code but not mentioned in the whitepaper.
Comparison of Model:
Both the whitepaper and the code use Logistic Regression. There is no discrepancy here.
Comparison of Validation Metrics:
Whitepaper Metrics:

Accuracy: 90%
Precision: 90%
Recall: 85%
F1 Score: 88%
Code Metrics:

Accuracy: 96.67%
Precision: 96.67%
Recall: 96.67%
F1 Score: 96.67%
Comparison of Scores:

The code metrics are higher than those specified in the whitepaper. This indicates that the model performs better than the expectations set in the whitepaper.
Critical Validation Metrics:
The critical metrics (Accuracy, Precision, Recall, F1 Score) in the code are all greater than those in the whitepaper.
Conclusion:
Feature Discrepancy: The whitepaper is missing the feature "Sepal length (cm)" which is used in the code.
Validation Metrics Discrepancy: The code achieves higher validation metrics than those specified in the whitepaper.
Action Required:

White Paper Update Needed: The whitepaper is not updated. Please update the document to include the missing feature "Sepal length (cm)" and revise the validation metrics to reflect the improved performance of the model as demonstrated in the code.
This analysis ensures that the documentation accurately reflects the implementation and performance of the model.
            '''
    result = CreateReport(input1,input2,input3)
    print("=== Summary ===")
    print(result)
