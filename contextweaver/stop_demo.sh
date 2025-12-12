#!/bin/bash
# stop_demo.sh

echo "🛑 Stopping ContextWeaver Demo..."

# Find and kill Streamlit process
lsof -ti:8501 | xargs kill -9

echo "✅ Demo stopped"
echo "📊 To restart: ./run_demo.sh"