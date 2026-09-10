#!/usr/bin/env python3
"""
Push script - Run this with your GitHub token to push the repo.

Usage:
    export GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
    python3 push_repo.py djohnson334/neurosymbolic-game-ai
"""

import subprocess
import sys
import os

def push_repo(repo_name):
    """Push the project to GitHub."""
    
    # Get token from environment
    token = os.environ.get('GITHUB_TOKEN')
    if not token:
        print("ERROR: GITHUB_TOKEN not set")
        print("Set it with: export GITHUB_TOKEN=ghp_...")
        return False
    
    # Ensure we're in the project directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Verify git repo
    result = subprocess.run(['git', 'remote', '-v'], capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        print("No remote configured. Setting up...")
        # Get owner/repo from input
        parts = repo_name.split('/')
        if len(parts) != 2:
            print(f"Invalid repo name: {repo_name}")
            return False
        owner, repo = parts
        
        # Add remote with token
        remote_url = f"https://{token}@github.com/{owner}/{repo}.git"
        subprocess.run(['git', 'remote', 'add', 'origin', remote_url])
    
    # Push
    print(f"Pushing to {repo_name}...")
    result = subprocess.run([
        'git', 'push', '-u', 'origin', 'main'
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Successfully pushed to GitHub!")
        return True
    else:
        print(f"❌ Push failed: {result.stderr}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python push_repo.py <owner>/<repo-name>")
        print("Example: python push_repo.py djohnson334/neurosymbolic-game-ai")
        sys.exit(1)
    
    success = push_repo(sys.argv[1])
    sys.exit(0 if success else 1)