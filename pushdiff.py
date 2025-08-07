import re
import json
import requests
# from fastapi import FastAPI, Query, HTTPException
from langchain_core.documents import Document
import os
from os import environ as env
from dotenv import load_dotenv
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain.text_splitter import RecursiveCharacterTextSplitter



def get_push_history(repo_url: str = "https://github.com/arunkenwal02/code-validator",push_id1: str = 52764657352,push_id2: str = 52699423231):    # Parse GitHub repo URL
    match = re.match(r"https://github.com/([^/]+)/([^/]+)(?:\.git)?", repo_url)
    if not match:
        raise HTTPException(status_code=400, detail="❌ Invalid GitHub repository URL format.")

    owner, repo_name_candidate = match.groups()
    repo_name = repo_name_candidate[:-4] if repo_name_candidate.endswith('.git') else repo_name_candidate

    api_url = f"https://api.github.com/repos/{owner}/{repo_name}/events"
    response = requests.get(api_url)
    events = response.json()
    req_history = [e for e in events if e['id'] in (push_id1, push_id2)]
    grouped_push_events = []
    commits_list = []
    for event in req_history:
        push_id = event['id']
        created_at = event['created_at']
        repo = event['repo']['name']
        commits_list = []

        for commit in event["payload"]["commits"]:
            sha = commit['sha']
            author = commit['author']['name']
            message = commit['message']

            commit_detail_url = f"https://api.github.com/repos/{owner}/{repo_name}/commits/{sha}"
            commit_detail_response = requests.get(commit_detail_url)

            if commit_detail_response.status_code != 200:
                diff = "❌ Failed to fetch diff"
            else:
                commit_detail = commit_detail_response.json()
                diffs = []
                for file in commit_detail.get('files', []):
                    patch = file.get('patch')
                    if patch:
                        diffs.append(f"File: {file['filename']}\n{patch}")
                diff = "\n\n".join(diffs) if diffs else "No diff available"

            commits_list.append({
                "sha": sha,
                "author": author,
                "commit_message": message,
                "code_diff": diff
            })

        grouped_push_events.append({
            "push_id": push_id,
            "repo": repo,
            "created_at": created_at,
            "commits": commits_list
        })

    with open("push_events.json", "w") as f:
        json.dump([push_id1,push_id2, owner, repo_name], f, indent=2)
    return grouped_push_events

get_push_history()