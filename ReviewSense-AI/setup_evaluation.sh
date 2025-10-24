#!/bin/bash

# ReviewSense AI - Evaluation Setup Script
# Installs required packages for model evaluation

echo "========================================"
echo "ReviewSense AI - Evaluation Setup"
echo "========================================"

# Check if in virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Warning: Not in a virtual environment"
    echo "It's recommended to activate your venv first:"
    echo "  source venv/bin/activate"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "📦 Installing evaluation packages..."

# Install evaluation metrics
pip install evaluate rouge-score nltk sacrebleu

# Install visualization
pip install matplotlib seaborn

# Download NLTK data for METEOR
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"

echo ""
echo "✅ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "  1. Ensure all_training_results.zip is in your project directory"
echo "  2. Run: python comprehensive_model_evaluation.py"
echo ""