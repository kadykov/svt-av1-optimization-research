# Create virtual environment and install Python dependencies
install:
    python -m venv venv
    . venv/bin/activate
    pip install -r requirements.txt
