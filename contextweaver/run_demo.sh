#!/bin/bash
# run_demo.sh

echo "🚀 Starting ContextWeaver Demo..."
echo ""

# Activate virtual environment
source venv/bin/activate

# Run Streamlit
streamlit run app/streamlit_app.py --server.port 8501

echo ""
echo "✅ Demo running at http://localhost:8501"