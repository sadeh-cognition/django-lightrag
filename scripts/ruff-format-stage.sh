#!/bin/bash

# Pre-commit hook that runs ruff formatter on staged Python files and restages
# the full batch once formatting is complete.
set -euo pipefail

python_files=("$@")

if [ "${#python_files[@]}" -eq 0 ]; then
    mapfile -t python_files < <(git diff --cached --name-only --diff-filter=ACMR | grep '\.py$' || true)
fi

if [ "${#python_files[@]}" -eq 0 ]; then
    echo "No Python files to format."
    exit 0
fi

echo "Running ruff formatter on staged Python files..."

for file in "${python_files[@]}"; do
    if [ -f "$file" ]; then
        echo "Formatting $file..."
        ruff format "$file"
    fi
done

git add -- "${python_files[@]}"

echo "Ruff formatting completed and files staged."
