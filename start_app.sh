#!/bin/bash

# Lab2 Stock Analysis System - Startup Script
# Course: DAP391m - Data Analytics Project

echo " Starting Lab2 Stock Analysis System..."
echo " Course: DAP391m - Data Analytics Project"
echo " Lab: Advanced Stock Analysis with Explainable AI"
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo " Virtual environment not found!"
    echo "Please ensure .venv directory exists in project root."
    exit 1
fi

# Check if source code exists
if [ ! -f "src/streamlit_app.py" ]; then
    echo " Main application not found!"
    echo "Please ensure src/streamlit_app.py exists."
    exit 1
fi

echo " Environment check passed"
echo " Starting Streamlit application on port 8507..."
echo ""
echo " Access the application at: http://localhost:8507"
echo " Login credentials: demo / demo123"
echo ""
echo " To stop the application: Press Ctrl+C"
echo "================================================"

# Start the Streamlit application
./.venv/bin/python -m streamlit run src/streamlit_app.py --server.port 8507