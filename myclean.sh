#!/bin/bash

rm -rf .gluonkit .pytest_cache
find . -path ./.venv -prune -o -name "__pycache__" -type d -prune -exec rm -rf {} +