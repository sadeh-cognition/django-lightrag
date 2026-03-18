#!/bin/bash

# Pre-commit hook that runs ruff formatter and stages changed files
set -e

# Get list of staged Python files
python_files=$(git diff --cached --name-only --diff-filter=ACM | grep '\.py$' || true)

if [ -z "$python_files" ]; then
    echo "No Python files to format."
    exit 0
fi

echo "Running ruff formatter on staged Python files..."

# Run ruff format on the staged files
for file in $python_files; do
    if [ -f "$file" ]; then
        echo "Formatting $file..."
        ruff format "$file"
        
        # Stage the formatted file
        git add "$file"
        echo "Staged $file"
    fi
done

echo "Ruff formatting completed and files staged."
