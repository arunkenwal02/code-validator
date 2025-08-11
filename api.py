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
import whitepaper_validation  as wv
import report_generation as rg
# ---------------- App init ----------------
app = FastAPI()

# -------------------- Routes --------------------

@app.post("/whitepaper_comparision")
def input_file_name(
    filename: str = Query(..., description="Whitepaper file name"),
    version: int = Query(..., description="Version number")
):
    try:
        result = wv.main(filename, version)
        return result  # keep same behavior (plain string)
    except Exception as e:
        # Return a clear 500 with details
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.post("/report_summary")
def input_file_name():
    try:
        Push_Commit_summary = rg.read_file("event_summary.txt")
        White_paper_comparision = rg.read_file("white_paper_comparision.txt")
        result = rg.generate_summary(Push_Commit_summary, White_paper_comparision)

        return result  # keep same behavior (plain string)
    except Exception as e:
        # Return a clear 500 with details
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/check")
def get_test():
    return "Hello World"
