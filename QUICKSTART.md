"""
Quick Start Guide

Follow these steps to set up and run the fraud detection system.
"""

## 1. Create Virtual Environment

### Windows (PowerShell)
```powershell
# Navigate to project directory
cd fraud_detection

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

### macOS/Linux
```bash
# Navigate to project directory
cd fraud_detection

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

## 2. Install Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
```

## 3. Verify Installation

```bash
# Test imports
python -c "import pandas, numpy, sklearn, xgboost, fastapi, streamlit; print('All imports successful!')"

# Check installed packages
pip list
```

## 4. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings (optional for local development)
# nano .env  (Linux/macOS)
# or edit manually in your editor
```

## 5. Run the Application

### Terminal 1 - Backend API
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Backend will be available at:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Terminal 2 - Frontend Dashboard
```bash
cd frontend
streamlit run app.py
```

Frontend will be available at:
- Dashboard: http://localhost:8501

## 6. Deactivate Virtual Environment

```bash
deactivate
```

## Common Commands

### Install Additional Packages
```bash
pip install package-name

# Update requirements.txt
pip freeze > requirements.txt
```

### Run Tests
```bash
pytest tests/ -v
```

### Format Code
```bash
black .
flake8 .
```

### Check Type Hints
```bash
mypy .
```

## Next Steps

1. Prepare your dataset in `data/raw/`
2. Create Jupyter notebooks in `notebooks/` for EDA
3. Train your model and save to `models/`
4. Implement prediction endpoints in `backend/routes/`
5. Build dashboard pages in `frontend/pages/`
6. Write unit tests in `tests/`

## Troubleshooting

### Virtual Environment Not Activating
- Ensure you're in the correct directory
- Check Python is installed: `python --version`
- Try full path: `python -m venv venv`

### Import Errors
- Confirm venv is activated
- Reinstall packages: `pip install -r requirements.txt`
- Clear Python cache: `find . -type d -name __pycache__ -exec rm -r {} +`

### Port Already in Use
- Change port: `uvicorn main:app --port 8001`
- Kill existing process on port: `lsof -ti:8000 | xargs kill` (macOS/Linux)

## Additional Resources

- [SETUP.md](./SETUP.md) - Detailed setup guide
- [README.md](./README.md) - Project overview
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Streamlit Docs](https://docs.streamlit.io/)
