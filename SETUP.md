# Fraud Detection System - Setup Guide

## Project Overview

This is a production-ready Python environment for an AI-powered fraud detection system with:
- **Backend API**: FastAPI-based REST API for model inference
- **Frontend**: Streamlit web dashboard for interactive analysis
- **Models**: Machine learning models using scikit-learn and XGBoost
- **Data**: Training and testing datasets
- **Notebooks**: Exploratory data analysis and model development

## Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git (for version control)

## Virtual Environment Setup

### Windows (PowerShell)

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# If you encounter execution policy error, run:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Upgrade pip
python -m pip install --upgrade pip
```

### Windows (Command Prompt)

```cmd
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate.bat

# Upgrade pip
python -m pip install --upgrade pip
```

### macOS/Linux (Bash/Zsh)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip
```

## Installation

Once the virtual environment is activated:

```bash
# Install all dependencies
pip install -r requirements.txt
```

## Project Structure

```
fraud_detection/
├── backend/                 # FastAPI backend
│   ├── __init__.py
│   ├── main.py             # FastAPI application entry point
│   ├── config.py           # Configuration settings
│   ├── models/
│   │   ├── __init__.py
│   │   └── fraud_model.py  # Model inference logic
│   ├── routes/
│   │   ├── __init__.py
│   │   └── predict.py      # Prediction endpoints
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── request.py      # Pydantic request models
│   └── utils/
│       ├── __init__.py
│       └── helpers.py      # Utility functions
│
├── frontend/                # Streamlit dashboard
│   ├── __init__.py
│   ├── app.py              # Main Streamlit app
│   ├── pages/
│   │   ├── __init__.py
│   │   ├── dashboard.py    # Main dashboard
│   │   ├── predictions.py  # Prediction interface
│   │   └── analytics.py    # Analytics page
│   └── utils/
│       ├── __init__.py
│       └── api_client.py   # Backend API client
│
├── models/                  # Trained models
│   ├── fraud_model.pkl     # Serialized XGBoost model
│   ├── scaler.pkl          # Feature scaler
│   └── README.md           # Model documentation
│
├── data/                    # Datasets
│   ├── raw/
│   │   └── transactions.csv
│   ├── processed/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── validation.csv
│   └── README.md
│
├── notebooks/              # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── tests/                  # Unit tests
│   ├── __init__.py
│   ├── test_backend.py
│   └── test_models.py
│
├── .env.example            # Environment variables template
├── .gitignore              # Git ignore file
├── requirements.txt        # Python dependencies
├── SETUP.md               # This file
└── README.md              # Project documentation
```

## Dependency Categories

### Core Data Processing
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scipy**: Scientific computing

### Machine Learning
- **scikit-learn**: Traditional ML algorithms and preprocessing
- **xgboost**: Gradient boosting for classification
- **joblib**: Model serialization and parallel processing

### Visualization
- **matplotlib**: Plotting library
- **seaborn**: Statistical data visualization

### Backend API
- **fastapi**: Modern Python web framework
- **uvicorn**: ASGI web server
- **pydantic**: Data validation using Python type hints
- **python-multipart**: Form data handling

### Frontend
- **streamlit**: Rapid web app development framework

### Development Tools
- **pytest**: Unit testing framework
- **black**: Code formatting
- **flake8**: Code linting
- **pylint**: Code analysis
- **mypy**: Static type checking

### Utilities
- **python-dotenv**: Environment variable management
- **requests**: HTTP client library

## Running the Application

### Backend API

```bash
# Navigate to project root (with venv activated)
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API will be available at: `http://localhost:8000`
API documentation (Swagger UI): `http://localhost:8000/docs`

### Frontend Dashboard

```bash
# In a new terminal (with venv activated)
cd frontend
streamlit run app.py
```

Dashboard will be available at: `http://localhost:8501`

## Environment Variables

Create a `.env` file in the project root (see `.env.example`):

```
# Backend Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Model Configuration
MODEL_PATH=./models/fraud_model.pkl
SCALER_PATH=./models/scaler.pkl
THRESHOLD=0.5

# Database (if applicable)
DATABASE_URL=postgresql://user:password@localhost/fraud_detection

# Logging
LOG_LEVEL=INFO
```

## Verifying Installation

```bash
# Test imports
python -c "import pandas, numpy, sklearn, xgboost, fastapi, streamlit; print('All imports successful!')"

# Run tests
pytest tests/ -v

# Check package versions
pip list
```

## Virtual Environment Management

### Exiting the Virtual Environment
```bash
deactivate
```

### Deleting the Virtual Environment
```bash
# Windows
rmdir /s venv

# macOS/Linux
rm -rf venv
```

### Recreating from requirements.txt
```bash
# After deleting venv
python -m venv venv
# Activate venv
pip install -r requirements.txt
```

## Production Deployment

### Generate Production Requirements
```bash
# Only production packages (excluding dev tools)
pip freeze | grep -v "black\|flake8\|pylint\|pytest\|mypy" > requirements-prod.txt
```

### Docker Deployment
Consider creating a `Dockerfile` and `docker-compose.yml` for containerized deployment.

## Common Issues

### Issue: "python command not found"
**Solution**: Ensure Python is installed and added to PATH. Use `python3` on macOS/Linux.

### Issue: "venv activation script not found"
**Solution**: Ensure you're in the project root directory before running activation command.

### Issue: "Permission denied" on macOS/Linux
**Solution**: 
```bash
chmod +x venv/bin/activate
source venv/bin/activate
```

### Issue: "pip not found in virtual environment"
**Solution**: 
```bash
python -m pip install --upgrade pip
```

## Next Steps

1. Create backend and frontend application files
2. Prepare and explore your dataset
3. Develop and train ML models
4. Build API endpoints for predictions
5. Create Streamlit dashboard
6. Write unit tests
7. Set up CI/CD pipeline
8. Deploy to production

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Pandas Documentation](https://pandas.pydata.org/)
