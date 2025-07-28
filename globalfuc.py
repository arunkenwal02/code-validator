import requests
import re 

def get_repo_info(url):
    """
    Parses GitHub URL to extract owner and repo name.
    Returns (owner, repo_name) or None if invalid URL.
    """
    match = re.match(r"https://github.com/([^/]+)/([^/]+)(?:\.git)?", url)
    if match:
        owner, repo_name_candidate = match.groups()
        if repo_name_candidate.endswith('.git'):
            repo_name = repo_name_candidate[:-4]
        else:
            repo_name = repo_name_candidate
        return owner, repo_name
    return None

def fetch_commits(owner, repo_name, token):
    """Fetches commit history for a given repository using a token."""
    headers = {"Authorization": f"token {token}"}
    
    api_url = f"https://api.github.com/repos/{owner}/{repo_name}/commits"
    
    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status() # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching commits: {e}. Check URL or GitHub Token permissions.")
        return None

def fetch_commit_diff(owner, repo_name, sha, token):
    """Fetches the diff for a specific commit SHA using a token."""
    headers = {
        "Accept": "application/vnd.github.v3.diff",
        "Authorization": f"token {token}"
    }

    api_url = f"https://api.github.com/repos/{owner}/{repo_name}/commits/{sha}"
    
    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        return response.text 
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching diff for commit {sha}: {e}. Check GitHub Token permissions.")
        return None

def format_diff_for_streamlit(diff_text):
    """
    Formats raw Git diff text with HTML-based coloring (red for deletions, green for additions),
    showing only added/deleted lines, with improved filtering for .ipynb JSON.
    """
    formatted_lines = []
    in_hunk = False 
    is_ipynb = False

    for line in diff_text.splitlines():
        if line.startswith('diff --git'):
            formatted_lines.append(f"<pre style='color: white; background-color: #333; padding: 5px; border-radius: 5px; font-weight: bold;'>{line}</pre>")
            in_hunk = False 
            is_ipynb = '.ipynb' in line 
        elif line.startswith('--- a/') or line.startswith('+++ b/'):
            continue
        elif line.startswith('@@'):
            in_hunk = True
            continue 
        elif in_hunk:
            if line.startswith('+'):
                html_color = "#008000" 
                if is_ipynb:
                    match = re.match(r'^\+\s*"((?:[^"\\]|\\.)*)\\n",?$', line)
                    if match:
                        content = match.group(1).replace('\\n', '\n').replace('\\"', '"')
                        formatted_lines.append(f"<span style='color:{html_color};'>+{content}</span>")
                    else:
                        formatted_lines.append(f"<span style='color:{html_color};'>{line}</span>")
                else:
                    formatted_lines.append(f"<span style='color:{html_color};'>{line}</span>")
            elif line.startswith('-'):
                html_color = "#FF0000" # Red
                if is_ipynb:
                    match = re.match(r'^\-\s*"((?:[^"\\]|\\.)*)\\n",?$', line)
                    if match:
                        content = match.group(1).replace('\\n', '\n').replace('\\"', '"')
                        formatted_lines.append(f"<span style='color:{html_color};'>-{content}</span>")
                    else:
                        formatted_lines.append(f"<span style='color:{html_color};'>{line}</span>")
                else:
                    formatted_lines.append(f"<span style='color:{html_color};'>{line}</span>")
            # Context lines (starting with space) are still filtered out
        else:
            # Lines before the first '@@' or other non-code diff lines are skipped
            continue

    if not formatted_lines:
        return "No significant code changes detected (only metadata or context lines filtered out)."
        
  
    return f"<pre style='background-color: #262626; padding: 10px; border-radius: 8px; overflow-x: auto;'>{'<br>'.join(formatted_lines)}</pre>"

