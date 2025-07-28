import streamlit as st
from globalfuc import *

st.set_page_config(layout="wide", page_title="GitHub Repo Code Validator & Viewer")

st.title("Code-Validator-and-Deployer-Agent")

github_repo_url = st.text_input(
    "Enter GitHub Repo URL",
    "" 
)


github_token = st.secrets.get("GITHUB_TOKEN")

if not github_token:
    st.error("GitHub Token not found in Streamlit secrets. "
             "API rate limits will be very low, and access to private repositories will fail. "
             "Please add `GITHUB_TOKEN = \"your_pat_here\"` to your `.streamlit/secrets.toml` file.")
    st.stop() 


submit_button = st.button("Fetch Repo Details")


if submit_button:
    # Ensure the URL is provided before proceeding
    if not github_repo_url:
        st.warning("Please enter a GitHub repository URL.")
        st.stop()

    repo_info = get_repo_info(github_repo_url)

    if not repo_info:
        st.error("Invalid GitHub repository URL. Please enter a URL like `https://github.com/owner/repo_name`.")
    else:
        owner, repo_name = repo_info
        st.subheader(f"Repository: {owner}/{repo_name}")
        # st.write(f"Project Name: {project_name}") # Project name is commented out in your provided code

        st.markdown("---")
        st.subheader("Commit History")

        commits = fetch_commits(owner, repo_name, github_token) 
        
        if commits:
            for commit in commits:
                sha = commit['sha']
                author = commit['commit']['author']['name']
                date = commit['commit']['author']['date']
                message = commit['commit']['message']

                st.markdown(f"**Commit:** `{sha[:7]}`")
                st.markdown(f"**Author:** {author}")
                st.markdown(f"**Date:** {date}")
                st.markdown(f"**Message:** {message}")
                
                # --- Display Code Difference Directly (No Expander) ---
                # st.markdown(f"**Code Difference for Commit `{sha[:7]}`:**")
                diff_text = fetch_commit_diff(owner, repo_name, sha, github_token)
                if diff_text:
                    # Use st.markdown with unsafe_allow_html=True for HTML rendering
                    st.markdown(format_diff_for_streamlit(diff_text), unsafe_allow_html=True) 
                else:
                    st.warning("Could not retrieve diff for this commit. Check token permissions or API rate limits.")
                st.markdown("---") # Separator between commits
        else:
            st.warning("No commits found or unable to fetch commit history. Check URL or GitHub API rate limits.")

