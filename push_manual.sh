#!/bin/bash
# Manual GitHub Push Script
# Run this after creating a GitHub repo

set -e

REPO_NAME="${1:-djohnson334/neurosymbolic-game-ai}"

echo "Creating release package..."

# Create a clean package
PARENT_DIR=$(dirname "$(pwd)")
REPO_DIR="$PARENT_DIR/neurosymbolic-game-ai-release"

rm -rf "$REPO_DIR"
mkdir -p "$REPO_DIR"

# Copy files (excluding cache and test artifacts)
cp -r game.py heuristics.py github/ "$REPO_DIR/" 2>/dev/null || true
cp -r .github/ "$REPO_DIR/" 2>/dev/null || true

# Copy core files
for f in README.md distill.py evaluate.py quick_eval.py results.npz game_visualization.png push_repo.py; do
    if [ -f "$f" ]; then
        cp "$f" "$REPO_DIR/"
    fi
done

echo "Package created at: $REPO_DIR"
echo ""
echo "Next steps:"
echo "1. Create a new repo on GitHub: $REPO_NAME"
echo "2. Clone it or set it as remote"
echo "3. Run: cd $REPO_DIR && git add . && git commit -m 'Add project' && git push"
echo ""
echo "Or use the push_repo.py script with your token:"
echo "  export GITHUB_TOKEN=ghp_your_token_here"
echo "  python3 push_repo.py $REPO_NAME"