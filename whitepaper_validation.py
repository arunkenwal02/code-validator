
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import os
import time
from datetime import datetime
import fitz
import pandas as pd
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests, json, base64, hashlib, difflib
import re
from io import BytesIO
from urllib.parse import quote
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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

# GitHub config
GITHUB_OWNER = "arunkenwal02"
GITHUB_REPO = "code-validator"
NOTEBOOK_FILE_PATH = "loan-approval-prediction.ipynb"

# Output file (as in your code)
# OUTPUT_TXT = "white_paper_comparision.txt"

# WHITEPAPER_NAME = "Load Prediction Whitepaper.pdf"
# VERSION_NUMBER =1 


def read_white_paper_from_gcp(filename, base_url, version_number):
    # Safely encode filename for a URL
    #filename = "Load Prediction Whitepaper.pdf"
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
    print(f"\n Extracting text from {len(doc)} pages of {filename}... \n")
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")  # Extract as plain text
        text_block += f"--- Page {page_num + 1} ---\n{text}\n\n"
        # print(f"--- Page {page_num + 1} ---")
        # print(text)
        # print()
    return text_block 

 
def extract_from_pdf(whitepaper_name : str, base_url : str,version_number : int) -> str:
    return read_white_paper_from_gcp(whitepaper_name, base_url, version_number)


# ------------------Start Get 2 psuh ids from JSON file------------------

def get_first_two_push_ids(path: str = "push_events.json") -> list:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    results = []
    
    # If the JSON is a list of dicts
    if isinstance(data, list):
        for item in data[:2]:  # first two items
            if isinstance(item, dict):
                first_key = list(item.keys())[0]
                results.append(str(item[first_key]))
            else:
                results.append(str(item))
    
    # If the JSON is a dict of lists
    elif isinstance(data, dict):
        first_key = next(iter(data))
        results = [str(val) for val in data[first_key][:2]]
    
    else:
        raise ValueError("Unsupported JSON structure.")
    
    return results

#. -----------------End Get 2 Push Ids from Json File

# --------------------Start Code Diff and base version from GitHub--------------------

def _gh_get(url: str, params: Optional[dict] = None) -> dict:
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"GitHub API error {r.status_code}: {r.text}")
    return r.json()

def get_push_event(owner: str, repo: str, push_id: str) -> Optional[dict]:
    url = f"https://api.github.com/repos/{owner}/{repo}/events"
    events = _gh_get(url)
    for e in events:
        if e.get("type") == "PushEvent" and str(e.get("id")) == str(push_id):
            return e
    return None

def get_sha_pair_from_push_id(owner: str, repo: str, push_id: str) -> Tuple[Optional[str], Optional[str]]:
    e = get_push_event(owner, repo, push_id)
    if not e:
        return None, None
    return e["payload"]["before"], e["payload"]["head"]

def list_commits_between(owner: str, repo: str, base_sha: str, head_sha: str) -> List[dict]:
    """
    Return commits (oldest..newest) between base..head via compare API.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/compare/{base_sha}...{head_sha}"
    data = _gh_get(url)
    return data.get("commits", [])

def commit_touches_path(owner: str, repo: str, commit_sha: str, path: str) -> bool:
    """
    Inspect a single commit for files it touched.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{commit_sha}"
    data = _gh_get(url)
    for f in data.get("files", []) or []:
        if f.get("filename") == path:
            return True
    return False

def all_change_commits_in_push(owner: str, repo: str, push_before: str, push_head: str, path: str) -> List[dict]:
    """
    Within a push range, return the list of commit objects that changed 'path', oldest..newest.
    Each item is the full commit dict from compare API ('commits' array).
    """
    commits = list_commits_between(owner, repo, push_before, push_head)
    change_commits = []
    for c in commits:
        sha = c.get("sha")
        if not sha:
            continue
        if commit_touches_path(owner, repo, sha, path):
            change_commits.append(c)
    return change_commits

def latest_commit_for_path(owner: str, repo: str, path: str) -> Optional[str]:
    url = f"https://api.github.com/repos/{owner}/{repo}/commits"
    params = {"path": path, "per_page": 1}
    commits = _gh_get(url, params=params)
    if commits:
        return commits[0]["sha"]
    return None

def get_file_content_at(owner: str, repo: str, path: str, ref: str) -> Optional[str]:
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    try:
        data = _gh_get(url, params={"ref": ref})
    except RuntimeError:
        return None
    if "content" not in data:
        return None
    encoding = data.get("encoding", "base64")
    if encoding != "base64":
        raise RuntimeError(f"Unexpected encoding '{encoding}' for {path}@{ref}")
    return base64.b64decode(data["content"]).decode("utf-8", errors="replace")


def _clip(s: str, max_chars: int) -> str:
    s = s if isinstance(s, str) else str(s)
    return s if len(s) <= max_chars else s[:max_chars] + " …[truncated]"

def _as_lines(s: str) -> list[str]:
    return s.splitlines() if s else ["(empty)"]

def normalize_for_diff(
    text: str,
    url_hint: str,
    include_outputs: bool = False,
    output_mode: str = "summary",
    max_output_chars: int = 2000
) -> List[str]:
    # Try parse as .ipynb
    nb = None
    try:
        maybe = json.loads(text)
        if isinstance(maybe, dict) and "cells" in maybe:
            nb = maybe
    except Exception:
        pass

    if nb is None and not url_hint.endswith(".ipynb"):
        return text.splitlines()

    if nb is None:
        return text.splitlines()

    lines: list[str] = []
    for i, cell in enumerate(nb.get("cells", []), start=1):
        ctype = cell.get("cell_type", "unknown")
        src_field = cell.get("source", [])
        src = "".join(src_field) if isinstance(src_field, list) else (src_field or "")
        exec_count = cell.get("execution_count", None)
        header = f"### CELL {i} [{ctype}]" + (f" (exec_count={exec_count})" if exec_count is not None else "")
        lines.append(header)

        source_lines = src.splitlines() if src else ["(empty source)"]
        lines.extend(source_lines)
        lines.append("")

        if include_outputs and cell.get("outputs"):
            lines.append(">>> OUTPUTS")
            for j, out in enumerate(cell["outputs"], start=1):
                ot = out.get("output_type", "unknown")
                if ot == "stream":
                    s = _clip(out.get("text", ""), max_output_chars)
                    lines.append(f"[{j}] stream:")
                    lines.extend(_as_lines(s))
                elif ot in ("execute_result", "display_data"):
                    data = out.get("data", {})
                    if "text/plain" in data:
                        s = data["text/plain"]
                        if isinstance(s, list):
                            s = "".join(s)
                        s = _clip(s, max_output_chars)
                        lines.append(f"[{j}] {ot} text/plain:")
                        lines.extend(_as_lines(s))
                    else:
                        keys = list(data.keys())
                        summary_bits = []
                        if "image/png" in keys:
                            b64 = data["image/png"]
                            if isinstance(b64, list):
                                b64 = "".join(b64)
                            digest = hashlib.sha1(b64.encode("utf-8")).hexdigest()[:10]
                            summary_bits.append(f"image/png sha1={digest} len={len(b64)}")
                        for k in keys:
                            if k not in ("text/plain", "image/png"):
                                summary_bits.append(k)
                        lines.append(f"[{j}] {ot} ({', '.join(summary_bits) if summary_bits else 'non-text output'})")

                        if output_mode == "full":
                            try:
                                safe_dump = {k: _clip(json.dumps(v) if not isinstance(v, str) else v, max_output_chars)
                                             for k, v in data.items()}
                                dumped = json.dumps(safe_dump, indent=2)
                                lines.extend(dumped.splitlines())
                            except Exception as e:
                                lines.append(f"(could not dump rich output: {e})")
                elif ot == "error":
                    ename = out.get("ename", "")
                    evalue = out.get("evalue", "")
                    tb = "\n".join(out.get("traceback", []) or [])
                    tb = _clip(tb, max_output_chars)
                    lines.append(f"[{j}] error: {ename}: {evalue}")
                    lines.extend(_as_lines(tb))
                else:
                    lines.append(f"[{j}] {ot} (unrecognized output_type)")
            lines.append("<<< END OUTPUTS")
            lines.append("")
    return lines

def make_unified_diff(old_text: str, new_text: str, hint: str) -> str:
    old_lines = normalize_for_diff(old_text, hint, include_outputs=True, output_mode="full", max_output_chars=4000)
    new_lines = normalize_for_diff(new_text, hint, include_outputs=True, output_mode="full", max_output_chars=4000)
    diff = difflib.unified_diff(
        old_lines, new_lines,
        fromfile="BASELINE_VERSION",
        tofile="UPDATED_VERSION",
        lineterm=""
    )
    return "\n".join(diff) or "No differences found."

def cumulative_push_diff(
    owner: str,
    repo: str,
    notebook_file_path: str,
    push_event: dict
) -> Optional[Dict]:
    """
    Return baseline + updated versions (in-memory, no disk save).
    """
    before_sha = push_event["payload"]["before"]
    head_sha   = push_event["payload"]["head"]

    changed = all_change_commits_in_push(owner, repo, before_sha, head_sha, notebook_file_path)
    if not changed:
        return None

    first_change = changed[0]
    last_change  = changed[-1]

    base_parent_sha = first_change["parents"][0]["sha"] if first_change.get("parents") else before_sha
    updated_sha     = last_change["sha"]

    baseline_content = get_file_content_at(owner, repo, notebook_file_path, base_parent_sha)
    updated_content  = get_file_content_at(owner, repo, notebook_file_path, updated_sha)

    if baseline_content is None or updated_content is None:
        raise RuntimeError(f"Could not fetch contents at {base_parent_sha} or {updated_sha}")

    diff = make_unified_diff(baseline_content, updated_content, notebook_file_path)


    old_norn_code = normalize_for_diff(baseline_content, notebook_file_path, include_outputs=True, output_mode="full", max_output_chars=4000)
    new_norm_code = normalize_for_diff(updated_content, notebook_file_path, include_outputs=True, output_mode="full", max_output_chars=4000)
    

    old_norn_code = "\n".join(old_norn_code) or "No differences found."
    new_norm_code =  "\n".join(new_norm_code) or "No differences found."

    
    return {
        "changed": True,
        "baseline_sha": base_parent_sha,
        "updated_sha": updated_sha,
        "baseline_content": old_norn_code,  # ← previous version text
        "updated_content": new_norm_code,    # ← latest version text
        "diff": diff
    }


def analyze_two_pushes(
    owner: str,
    repo: str,
    notebook_file_path: str,
    push_id_1: str,
    push_id_2: str
) -> Dict:
    """
    Priority rule for multiple changes:
      1) Prefer cumulative changes within Push 1 (if any).
      2) Else, prefer cumulative changes within Push 2 (if any).
      3) Else, unchanged across both pushes -> return latest prior version.
    """
    e1 = get_push_event(owner, repo, push_id_1)
    e2 = get_push_event(owner, repo, push_id_2)
    if not e1 or not e2:
        raise RuntimeError("One or both push IDs were not found in recent events.")

    # 1) Try Push 1 first (PRIORITY)
    res1 = cumulative_push_diff(owner, repo, notebook_file_path, e1)
    if res1:
        return {
            "status": "changed",
            "priority_push_id": str(e1["id"]),
            "message": "Changes detected. Showing cumulative diff within Push 1 (baseline -> most updated change in Push 1).",
            **res1
        }

    # 2) If Push 1 had no change, try Push 2
    res2 = cumulative_push_diff(owner, repo, notebook_file_path, e2)
    if res2:
        return {
            "status": "changed",
            "priority_push_id": str(e2["id"]),
            "message": "No change in Push 1; showing cumulative diff within Push 2 (baseline -> most updated change in Push 2).",
            **res2
        }

    # 3) Neither push changed the file -> provide latest prior version
    last_touch_sha = latest_commit_for_path(owner, repo, notebook_file_path)
    if not last_touch_sha:
        return {
            "status": "unchanged_no_prior",
            "message": "No changes across the two pushes and no prior commit found that touched the file."
        }

    prior_text = get_file_content_at(owner, repo, notebook_file_path, last_touch_sha)
    if prior_text is None:
        return {
            "status": "unchanged_no_prior_content",
            "last_touch_sha": last_touch_sha,
            "message": "No changes across the two pushes; prior commit found but content could not be retrieved."
        }

    return {
        "status": "unchanged",
        "last_touch_sha": last_touch_sha,
        "message": "No changes between those two pushes; returning the latest prior version of the file.",
        "prior_version_excerpt": prior_text[:1500] + ("…[truncated]" if len(prior_text) > 1500 else "")
    }

# -------------------End Code Diff and base version from Github--------------------


# ------------------ Srart White paper comparision with code 

# -------------------- Config --------------------
SECTION_QUERIES: Dict[str, List[str]] = {
    # Used for Summary:
    "Executive Summary": ["executive summary", "overview", "objective", "goal"],

    # Kept sections:
    "Data Description": ["features", "dataset", "data description", "columns", "shape", "dimension"],
    "Preprocessing": [
        "preprocessing", "missing values", "imputation", "encoding",
        "scaling", "standardization", "normalization",
        "balancing", "smote", "undersampling", "oversampling",
        "one-hot", "label encoding", "feature engineering"
    ],
    "Model Architecture": [
        "final model", "algorithm", "logistic regression", "random forest", "xgboost", "svm",
        "model selection", "best params", "hyperparameters", "business justification"
    ],
    "Training Methodology": ["train test split", "validation", "k-fold", "grid search", "cross-validation", "threshold"],
    "Evaluation Metrics": [
        "metrics", "accuracy", "precision", "recall", "f1", "pr-auc", "roc-auc", "specificity",
        "best model", "evaluation", "validation score", "test score"
    ],
    "Model Monitoring & Drift": [
        "monitoring", "data drift", "concept drift", "alerts", "thresholds", "retraining",
        "performance monitoring", "population stability index", "PSI", "alerting", "dashboards"
    ],
    "Fallback / Human-in-the-Loop": [
        "fallback", "human-in-the-loop", "override", "escalation", "manual review",
        "confidence threshold", "confidence score", "policy exception"
    ],
    "Stress Conditions / Scenario Analysis": [
        "stress", "scenario", "crisis", "worst case", "edge case", "macroeconomic",
        "adverse", "shock", "sensitivity analysis"
    ],

    # Excluded but used for WP-only addenda:
    "Introduction": ["introduction", "background", "context"],
    "Related Work / Literature Review": ["related work", "literature", "prior work"],
    "Deployment Strategy": ["deployment", "serving", "api", "scalability", "latency", "throughput"],
    "Future Work": ["future work", "next steps", "roadmap"],
}

# Metrics to consider for Phase 2
METRIC_PATTERNS = {
    "Accuracy": r"\baccuracy\b",
    "Precision": r"\bprecision\b|\bppv\b",
    "Recall": r"\brecall\b|\bsensitivity\b|\btpr\b",
    "F1-Score": r"\bf1(?:[-\s]?score)?\b",
    "PR-AUC": r"\bpr-?auc\b|\bprecision[-\s]?recall\s*auc\b|\baverage\s*precision\b|\bAP\b",
    "ROC-AUC": r"\broc-?auc\b|\bauc\b(?!\s*pr)",
    "Specificity": r"\bspecificity\b|\btnr\b",
}
VALUE_PATTERN = r"(?:(?:~|≈|=)?\s*)(\d+(?:\.\d+)?)(\s*%)?"

# Phase 1 sections to show (order matters)
PHASE1_SECTIONS = [
    "Summary",  # maps to Executive Summary (WP-only in bullets)
    "Data Description",
    "Preprocessing",
    "Model Architecture",
    "Evaluation Details & Metrics",  # merges Training Methodology + Evaluation Metrics
    "Model Monitoring & Drift",
    "Fallback / Human-in-the-Loop",
    "Stress Conditions / Scenario Analysis",
    "White Paper — Additional Details Not Captured Above",  # WP addenda section
]

# Map Phase 1 synthetic section names to underlying retrieval sections
SECTION_ALIASES: Dict[str, List[str]] = {
    "Summary": ["Executive Summary"],
    "Evaluation Details & Metrics": ["Training Methodology", "Evaluation Metrics"],
    # others are 1:1 and use their own name
}

# Excluded sections that we still mine for WP addenda
EXCLUDED_SECTIONS_FOR_WP_ADDENDA = [
    "Introduction", "Related Work / Literature Review", "Deployment Strategy", "Future Work"
]


# -------------------- Data types --------------------
@dataclass
class FilteredItem:
    section: str
    source: str  # "White Paper" | "Code Diff" | "Code v1"
    text: str
    score: float


# -------------------- Chunkers (simple) --------------------
def _simple_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]

def chunk_whitepaper(text: str) -> List[str]:
    chunks: List[str] = []
    for m in re.finditer(r"```json\s*(.*?)```", text or "", flags=re.DOTALL | re.IGNORECASE):
        chunks.extend(_simple_paragraphs(m.group(1)))
    if not chunks:
        chunks = _simple_paragraphs(text)
    return chunks

def chunk_code_diff(text: str) -> List[str]:
    # split on list bullets and blank lines to capture atomic statements from diffs
    blocks = [b.strip() for b in re.split(r"(?m)^\s*[-*•]\s+|\n\s*\n", text or "") if b.strip()]
    return blocks or _simple_paragraphs(text)

def chunk_old_code(text: str) -> List[str]:
    try:
        nb = json.loads(text or "")
        if isinstance(nb, dict) and "cells" in nb:
            out = []
            for c in nb.get("cells", []):
                src = c.get("source", [])
                if isinstance(src, list):
                    src = "".join(src)
                if src and src.strip():
                    out.append(src.strip())
            return out or _simple_paragraphs(text)
    except Exception:
        pass
    return _simple_paragraphs(text)


# -------------------- Retrieval --------------------
def top_k_snippets_for_query(chunks: List[str], query: str, k: int = 3) -> List[Tuple[str, float]]:
    if not chunks:
        return []
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    X = vec.fit_transform(chunks)
    sims = cosine_similarity(vec.transform([query]), X).ravel()
    order = sims.argsort()[::-1]
    return [(chunks[i], float(sims[i])) for i in order[:k] if sims[i] > 0]

def retrieve_filtered_context(whitepaper: str, code_diff: str, old_code: str, k_per_source: int = 3) -> List[FilteredItem]:
    wp_chunks = chunk_whitepaper(whitepaper)
    v2_chunks = chunk_code_diff(code_diff)
    v1_chunks = chunk_old_code(old_code)

    items: List[FilteredItem] = []
    for section, queries in SECTION_QUERIES.items():
        query = " ".join([section] + queries)
        for txt, sc in top_k_snippets_for_query(wp_chunks, query, k=k_per_source):
            items.append(FilteredItem(section, "White Paper", txt, sc))
        for txt, sc in top_k_snippets_for_query(v2_chunks, query, k=k_per_source):
            # tag as Code Diff (not "Code v2")
            items.append(FilteredItem(section, "Code Diff", txt, sc))
        for txt, sc in top_k_snippets_for_query(v1_chunks, query, k=k_per_source):
            items.append(FilteredItem(section, "Code v1", txt, sc))

    items.sort(key=lambda x: (x.section, -x.score))
    return items


# -------------------- Metrics & WP extractors --------------------
def _as_pct(val: float) -> float:
    return val * 100.0 if val <= 1.0 else val

def extract_metrics(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    low = (text or "").lower()
    for name, patt in METRIC_PATTERNS.items():
        for m in re.finditer(patt, low):
            window = low[m.end(): m.end() + 200]
            vm = re.search(VALUE_PATTERN, window)
            if vm:
                try:
                    out.setdefault(name, _as_pct(float(vm.group(1))))
                    break
                except ValueError:
                    pass
    return out

def pct_str(x: Optional[float]) -> str:
    return f"{x:.1f}%" if isinstance(x, (int, float)) else "N/A"

# --- NEW: extract preprocessing steps explicitly mentioned in WP ---
_PREPROC_CANON = [
    ("Missing value imputation", r"missing|imput"),
    ("One-hot / label encoding", r"one[-\s]?hot|label\s+encod"),
    ("Scaling / standardization", r"scal|standardiz|normaliz"),
    ("Class balancing (SMOTE/over/under)", r"smote|over-?sampl|under-?sampl|class\s+imbalance|balanc"),
    ("Feature engineering", r"feature\s+engineer|derive|transform"),
    ("Data type consistency / cleaning", r"dtype|data\s+type|cleaning|sanitize"),
]

def extract_preprocessing_steps_wp(wp_text: str) -> List[str]:
    low = (wp_text or "").lower()
    steps = []
    for label, patt in _PREPROC_CANON:
        if re.search(patt, low):
            steps.append(label)
    return steps

# --- NEW: extract final model name + hyperparameters from WP ---
_MODEL_PATTERNS = [
    ("Logistic Regression", r"logistic\s+regress"),
    ("Random Forest", r"random\s+forest"),
    ("XGBoost", r"xgboost|xgb"),
    ("SVM", r"\bsvm\b|support\s+vector"),
    ("Decision Tree", r"decision\s+tree"),
]

_HYPER_PATTERNS = [
    ("C", r"\bC\s*[:=]\s*([0-9]*\.?[0-9]+)"),
    ("max_iter", r"\bmax[_\s-]?iter\s*[:=]\s*(\d+)"),
    ("penalty", r"\bpenalty\s*[:=]\s*['\"]?([a-z0-9_-]+)['\"]?"),
    ("solver", r"\bsolver\s*[:=]\s*['\"]?([a-z0-9_-]+)['\"]?"),
    ("n_estimators", r"\bn[_\s-]?estimators\s*[:=]\s*(\d+)"),
    ("max_depth", r"\bmax[_\s-]?depth\s*[:=]\s*(\d+|None)"),
    ("min_samples_split", r"\bmin[_\s-]?samples[_\s-]?split\s*[:=]\s*(\d+)"),
    ("min_samples_leaf", r"\bmin[_\s-]?samples[_\s-]?leaf\s*[:=]\s*(\d+)"),
    ("criterion", r"\bcriterion\s*[:=]\s*['\"]?([a-z0-9_-]+)['\"]?"),
    ("learning_rate", r"\blearning[_\s-]?rate\s*[:=]\s*([0-9]*\.?[0-9]+)"),
]

def extract_model_from_wp(wp_text: str) -> Tuple[Optional[str], Dict[str, str]]:
    low = (wp_text or "").lower()
    model_name = None
    for name, patt in _MODEL_PATTERNS:
        if re.search(patt, low):
            model_name = name
            break
    params: Dict[str, str] = {}
    # scan the full text once for each param
    for key, patt in _HYPER_PATTERNS:
        m = re.search(patt, wp_text)
        if m:
            params[key] = m.group(1)
    return model_name, params

# -------------------- Builders --------------------
def _group_by_section(items: List[FilteredItem]) -> Dict[str, List[FilteredItem]]:
    g = defaultdict(list)
    for it in items:
        g[it.section].append(it)
    return g

def _collect_items_for_phase_section(g: Dict[str, List[FilteredItem]], phase_section: str) -> List[FilteredItem]:
    """Maps synthetic Phase 1 section names to underlying retrieval sections."""
    bases = SECTION_ALIASES.get(phase_section, [phase_section])
    out: List[FilteredItem] = []
    for base in bases:
        out.extend(g.get(base, []))
    # sort by score desc
    return sorted(out, key=lambda x: -x.score)

def _bullets_grouped_by_source(section_items: List[FilteredItem], bullets_per_source: int, summary_wp_only: bool=False) -> List[str]:
    """
    Return bullets grouped under [WP], [Code Diff], [Code-v1].
    If summary_wp_only is True, only include [WP] bullets regardless of other sources.
    """
    lines: List[str] = []
    by_src = defaultdict(list)
    for it in section_items:
        by_src[it.source].append(it)

    # Order: WP, Code Diff, Code v1
    ordered_sources = ("White Paper", "Code Diff", "Code v1")
    for src in ordered_sources:
        if summary_wp_only and src != "White Paper":
            continue
        if src in by_src:
            tag = "[WP]" if src == "White Paper" else ("[Code Diff]" if src == "Code Diff" else "[Code-v1]")
            topn = sorted(by_src[src], key=lambda x: -x.score)[:bullets_per_source]
            for t in topn:
                snippet = " ".join(t.text.split())
                snippet = snippet[:600] + " …" if len(snippet) > 600 else snippet
                lines.append(f"- {tag} {snippet}")
    return lines

def _format_metrics_line(metrics: Dict[str, float]) -> Optional[str]:
    if not metrics:
        return None
    ordered = ["Accuracy", "Precision", "Recall", "F1-Score", "PR-AUC", "ROC-AUC", "Specificity"]
    parts = []
    for m in ordered:
        if m in metrics:
            parts.append(f"{m}={metrics[m]:.1f}%")
    return ", ".join(parts) if parts else None

def build_phase1_summary(
    items: List[FilteredItem],
    bullets_per_source: int = 2,
    wp_metrics: Optional[Dict[str, float]] = None,
    code_metrics: Optional[Dict[str, float]] = None,
    v1_metrics: Optional[Dict[str, float]] = None,
    wp_text_for_inference: str = "",
) -> str:
    """
    Phase 1 section builder with:
      - Summary: WP-only bullets.
      - Preprocessing: prepend explicit WP-derived checklist if present.
      - Model Architecture: prepend explicit WP-derived model + hyperparameters (and scores).
      - Evaluation Details & Metrics: inject explicit WP metrics line at top.
    """
    g = _group_by_section(items)
    lines = ["# PHASE 1 — Business-Focused Summary (Kept Sections Only)"]
    for section in PHASE1_SECTIONS:
        if section == "White Paper — Additional Details Not Captured Above":
            # WP addenda from excluded sections
            addenda_items: List[FilteredItem] = []
            for ex_sec in EXCLUDED_SECTIONS_FOR_WP_ADDENDA:
                addenda_items.extend(g.get(ex_sec, []))
            addenda_items = [it for it in addenda_items if it.source == "White Paper"]
            if not addenda_items:
                continue
            lines.append(f"\n## {section}")
            # Up to 8 extra WP-only points
            for it in sorted(addenda_items, key=lambda x: -x.score)[:8]:
                snip = " ".join(it.text.split())
                snip = snip[:700] + " …" if len(snip) > 700 else snip
                lines.append(f"- [WP] {snip}")
            continue

        section_items = _collect_items_for_phase_section(g, section)
        # If completely empty section, skip
        if not section_items and section not in ("Preprocessing", "Model Architecture", "Evaluation Details & Metrics", "Summary"):
            continue

        # Section header
        lines.append(f"\n## {section}")

        # Summary: WP-only bullets (as pointers)
        if section == "Summary":
            if section_items:
                lines.extend(_bullets_grouped_by_source(section_items, bullets_per_source, summary_wp_only=True))
            continue

        # Preprocessing: prepend inferred WP checklist if present
        if section == "Preprocessing":
            inferred_steps = extract_preprocessing_steps_wp(wp_text_for_inference)
            if inferred_steps:
                lines.append("- [WP] **Preprocessing steps (explicit in White Paper):** " + "; ".join(inferred_steps))
            # then normal grouped bullets
            if section_items:
                lines.extend(_bullets_grouped_by_source(section_items, bullets_per_source))
            continue

        # Model Architecture: prepend model name + hyperparameters from WP
        if section == "Model Architecture":
            model_name, params = extract_model_from_wp(wp_text_for_inference)
            wp_scores_line = _format_metrics_line(wp_metrics or {})
            if model_name or params or wp_scores_line:
                parts = []
                if model_name:
                    parts.append(f"Final model: **{model_name}**")
                if params:
                    hp = ", ".join([f"{k}={v}" for k, v in params.items()])
                    parts.append(f"Hyperparameters: {hp}")
                if wp_scores_line:
                    parts.append(f"WP Scores: {wp_scores_line}")
                lines.append("- [WP] " + " | ".join(parts))
            else:
                lines.append("- [WP] **Final model not explicitly named in WP.** Please confirm model choice and provide business justification for its selection.")
            # then normal grouped bullets
            if section_items:
                lines.extend(_bullets_grouped_by_source(section_items, bullets_per_source))
            continue

        # Evaluation Details & Metrics: inject WP metrics line first (if available)
        if section == "Evaluation Details & Metrics":
            wp_line = _format_metrics_line(wp_metrics or {})
            if wp_line:
                lines.append(f"- [WP] Scores — {wp_line}")
            if section_items:
                lines.extend(_bullets_grouped_by_source(section_items, bullets_per_source))
            continue

        # All other sections: grouped bullets (WP, Code Diff, Code-v1)
        if section_items:
            lines.extend(_bullets_grouped_by_source(section_items, bullets_per_source))

    return "\n".join(lines).strip()

def build_phase2(whitepaper: str, code_diff: str, old_code: str) -> Tuple[str, str, str]:
    """
    Build table + highlights + notes.
    RULE: Exclude any metric that is missing in BOTH WP and Code (Diff/v1).
    Column label: 'Code Diff Score'.
    """
    wp = extract_metrics(whitepaper)
    v2 = extract_metrics(code_diff)
    v1 = extract_metrics(old_code)
    code: Dict[str, Optional[float]] = {k: v2.get(k, v1.get(k)) for k in METRIC_PATTERNS.keys()}

    rows, improvements, regressions = [], [], []

    for m in METRIC_PATTERNS.keys():
        wv, cv = wp.get(m), code.get(m)

        # Skip if missing in BOTH
        if wv is None and cv is None:
            continue

        ws, cs = pct_str(wv), pct_str(cv)
        if wv is None or cv is None:
            delta, finding = "N/A", "Missing in one source"
        else:
            d = cv - wv
            delta = f"{'+' if d>=0 else ''}{d:.1f} pp"
            if abs(d) < 0.05:
                finding = "Aligned"
            elif d > 0:
                finding = "Improved"
                if d >= 1.0: improvements.append(f"{m}: {delta}")
            else:
                finding = "Regressed"
                if -d >= 1.0: regressions.append(f"{m}: {delta}")

        rows.append((m, ws, cs, delta, finding))

    # Build table
    table = [
        "# PHASE 2 — Critical Metrics Comparison (Reported Only)",
        "",
        "| Metric | White Paper Score | Code Diff Score | Delta (Code - WP) | Finding |",
        "|--------|-------------------|-----------------|-------------------|---------|",
    ] + [f"| {m} | {ws} | {cs} | {d} | {f} |" for m, ws, cs, d, f in rows]

    # Highlights
    highlights = ["\n## Highlights"]
    if improvements:
        highlights.append("- **Top Improvements**")
        highlights += [f"  - {x}" for x in improvements]
    if regressions:
        highlights.append("- **Regressions**")
        highlights += [f"  - {x}" for x in regressions]

    # Notes
    notes = [
        "\n## Notes",
        "- Metrics are shown only if present in at least one source (WP or Code Diff/Code v1).",
        "- Code metrics prefer Code Diff. If absent, they fall back to Code v1.",
        "- Metric parsing is regex-based; ensure metric names sit close to numeric values.",
        "- All scores shown as percentages with 1 decimal.",
        "- If an out-of-range value appears (e.g., >100%), treat it as a parsing artifact and verify manually.",
    ]
    return "\n".join(table), "\n".join(highlights), "\n".join(notes)

def build_prompt(
    items: List[FilteredItem],
    table: str,
    highlights: str,
    notes: str,
    prompt_context: str = "Loan Approval Model: Comprehensive Summary & Comparison for Business Stakeholders"
) -> str:
    """
    Final prompt with explicit business-facing instructions.

    Sections to include (in order):
      1) Summary — executive overview for stakeholders (WP-only).
      2) Data Description.
      3) Preprocessing — LIST ALL steps discovered (imputation, encoding, scaling, balancing, feature engineering, etc.).
      4) Model Architecture — STATE the FINAL model; KEEP hyperparameters & scores from the White Paper if present.
         Otherwise, explicitly ASK stakeholders for the business justification.
      5) Evaluation Details & Metrics — include BOTH the evaluation approach and the MAIN metrics/scores
         for the FINAL model, with an explicit White Paper scores line first when available.
      6) Model Monitoring & Drift — detailed guidance (what to track, thresholds, cadence, alerting, retraining triggers).
      7) Fallback / Human-in-the-Loop — criteria/thresholds for fallback and WHEN to route to human review (critical conditions).
      8) Stress Conditions / Scenario Analysis — scenarios and mitigation strategies.
      9) White Paper — Additional Details Not Captured Above — bullet remaining WP-only details from Intro/Related/Deployment/Future Work.

    Formatting rules:
      - For each kept section, present bullets grouped by source tag: [WP], [Code Diff], [Code-v1].
      - For Summary: show [WP] bullets ONLY.
      - Use short bullets; no raw code blocks or “### CELL” headers.
      - Keep a plain-language, outcome-oriented tone (business stakeholders).
    """
    g = _group_by_section(items)

    def _grouped(section_name: str, limit_per_src: int = 6, summary_wp_only: bool=False) -> List[str]:
        section_items = _collect_items_for_phase_section(g, section_name)
        return _bullets_grouped_by_source(section_items, bullets_per_source=limit_per_src, summary_wp_only=summary_wp_only)

    out: List[str] = [
        f"### Prompt Context: {prompt_context}",
        "Audience: business stakeholders and non-technical leaders.",
        "Style: concise, outcome-oriented, plain language; highlight business impact, risk, compliance, and operations.",
        "Sources: [WP]=White Paper, [Code Diff]=Code diff (latest changes), [Code-v1]=Older code/notebook.",
        "\n=== FILTERED CONTEXT BY SECTION (Kept Only) ===",
    ]

    # Ordered sections per PHASE1_SECTIONS
    for sec in PHASE1_SECTIONS:
        if sec == "White Paper — Additional Details Not Captured Above":
            addenda_items: List[FilteredItem] = []
            for ex_sec in EXCLUDED_SECTIONS_FOR_WP_ADDENDA:
                addenda_items.extend(g.get(ex_sec, []))
            addenda_items = [it for it in addenda_items if it.source == "White Paper"]
            if addenda_items:
                out.append(f"\n## {sec}")
                for it in sorted(addenda_items, key=lambda x: -x.score)[:8]:
                    snip = " ".join(it.text.split())
                    snip = snip[:700] + " …" if len(snip) > 700 else snip
                    out.append(f"- [WP] {snip}")
            continue

        if sec == "Summary":
            bullets = _grouped(sec, limit_per_src=6, summary_wp_only=True)
        else:
            bullets = _grouped(sec, limit_per_src=6, summary_wp_only=False)

        if bullets:
            out.append(f"\n## {sec}")
            out.extend(bullets)

    out += [
        "\n=== METRICS COMPARISON (Reported Only) ===",
        table,
        highlights,
        notes,
        "\nInstruction: Produce Phase 1 using ONLY the filtered context above and the formatting rules; "
        "then append Phase 2 exactly as provided (table + highlights + notes).",
    ]
    return "\n".join(out).strip()

# -------------------- Public runner (use your variables here) --------------------
def run_pipeline_from_vars(
    wp_text: str,
    diff_text: str,
    v1_text: str,
    k_per_source: int = 3,
    bullets_per_source: int = 2,
    prompt_context: str = "Loan Approval Model: Comprehensive Summary & Comparison for Business Stakeholders",
):
    # Retrieve items
    items = retrieve_filtered_context(wp_text, diff_text, v1_text, k_per_source=k_per_source)

    # Extract metrics (for injecting WP scores into Phase 1 evaluation & model arch sections)
    wp_metrics = extract_metrics(wp_text)
    code_metrics = extract_metrics(diff_text)
    v1_metrics = extract_metrics(v1_text)

    # Build Phase 1 with injections + WP inference (preprocessing/model)
    phase1 = build_phase1_summary(
        items,
        bullets_per_source=bullets_per_source,
        wp_metrics=wp_metrics,
        code_metrics=code_metrics,
        v1_metrics=v1_metrics,
        wp_text_for_inference=wp_text,
    )

    # Build Phase 2 (filtered)
    table, highlights, notes = build_phase2(wp_text, diff_text, v1_text)

    # Build prompt
    prompt = build_prompt(items, table, highlights, notes, prompt_context=prompt_context)

    return {
        "phase1_summary": phase1,
        "phase2_table": table,
        "phase2_highlights": highlights,
        "phase2_notes": notes,
        "final_prompt": prompt,
        "filtered_items": items,
    }


def build_html_report_via_llm(output: dict, llm) -> str:
    sys_msg = (
    "You are a markdown formatter. "
    "Your output must be in clean markdown only (no code fences or backticks). "
    "Use proper headings (##, ###). "
    "Always use bullet points for lists (never plain text lists). "
    "For tabular data, render xas markdown tables. "
    "Ensure the output is clean, readable, and consistent."
)

    human_msg = f"""
    Render the following sections into markdown format with clear headings and subheadings. 
    - Use bullet points for all lists. 
    - Format any tabular data as markdown tables.
    - Ensure readability and structure.

    ## { "PHASE 1 — Business-Focused Summary (Kept Sections Only)" }
    {output.get("phase1_summary","")}

    ## { "PHASE 2 — Critical Metrics Comparison (Reported Only)" }
    {output.get("phase2_table","")}

    ## { "Highlights" }
    {output.get("phase2_highlights","")}

    ## { "Notes" }
    {output.get("phase2_notes","")}

    ## { "Final Prompt (for LLM)" }
    {output.get("final_prompt","")}
    """

    resp = llm.invoke([
        SystemMessage(content=sys_msg),
        HumanMessage(content=human_msg),
    ])
    return resp.content






# -----------------End of white paper comparision with code --------------------

def main(whitepaper_name: str, version_number: int) -> str:
    try:
        base_url = "https://storage.googleapis.com/whitepaper_test/"
        # 1) Whitepaper text via OneDrive version
        whitepaper_text = extract_from_pdf(whitepaper_name, base_url, version_number)
        first_two = get_first_two_push_ids("push_events.json")
        print("First two push IDs:", first_two)
        first_push_id = first_two[1]
        second_push_id = first_two[0]

        push_id_1 = first_push_id  
        push_id_2 = second_push_id  

        result = analyze_two_pushes(
            owner=GITHUB_OWNER,
            repo=GITHUB_REPO,
            notebook_file_path= NOTEBOOK_FILE_PATH,
            push_id_1=push_id_1,
            push_id_2=push_id_2
         )     

        print("Result from analyze_two_pushes:") 
        baseline_text = result["baseline_content"]  # previous version
        # updated_text  = result["updated_content"]   # new version
        diff_text = result["diff"]  # unified diff output
        # print ("Baseline Text: =============== \n", baseline_text )
        # print ("Diff Text: ===============\n", diff_text )

        try:
            # Expect you already defined these in your session:
            #   WHITEPAPER_, code_diff, old_code = ...
            print("Inside Try block ======================")
            output = run_pipeline_from_vars(wp_text= whitepaper_text,  diff_text= diff_text, v1_text= baseline_text)
            # print("Run pipeline fron vars runs successfully")
            # print(output["phase1_summary"])
            # print("\n" + output["phase2_table"])
            # print("\n" + output["phase2_highlights"])
            # print("\n" + output["phase2_notes"])
            # print("\n" + output["final_prompt"])

            # ---------------- Example usage ----------------
            print("Run pipeline from vars runs successfully")

            output_summary = build_html_report_via_llm(output, llm)

            # now `output_summary` holds the full HTML string
            # you can pass it directly to React: dangerouslySetInnerHTML={{ __html: output_summary }}

            print("\n=== OUTPUT SUMMARY ===\n")
            # print(output_summary)
            return output_summary

        except NameError:
            print("Define WHITEPAPER_, code_diff, old_code in your environment, or import run_pipeline_from_vars() and call it directly.")

        

    except Exception as e:
        # Bubble up so FastAPI can return proper error
        print("Exception in main block")
        return "Exception in main block"


# if __name__ == "__main__":
#     # Example usage
#     try:
#         result = main(WHITEPAPER_NAME, VERSION_NUMBER)
#         # print(result)
#     except Exception as e:
#         print(f"Error: {e}")

